import torch
from scene import GaussianModel
import pdb
import matplotlib.pyplot as plt

fail_cnt = 0


def WtoC(R, T, world_pos, gaussian_model):
    """
    将世界坐标系转换到相机坐标系

    Args:
        R ndarray:(3,3): 旋转矩阵.
        T ndarray:(3,): 平移矩阵.
        world_pos Tensor:(N,3): 世界坐标系下的坐标列表
        gaussian_model GaussianModel: 当前使用的高斯模型

    Returns:
        Tensor:(N,3): 完成坐标变换之后的相机坐标系下的坐标
    """
    R = torch.tensor(R, dtype=torch.float32, device=gaussian_model.get_xyz.device)
    T = torch.tensor(T, dtype=torch.float32, device=gaussian_model.get_xyz.device)
    relative_positions = world_pos - T
    return torch.matmul(relative_positions, R.T)


def CtoW(R, T, cam_pos, gaussian_model):
    """
        将世界坐标系转换到相机坐标系

        Args:
            R ndarray:(3,3): 旋转矩阵.
            T ndarray:(3,): 平移矩阵.
            cam_pos Tensor:(N,3): 相机坐标系下的坐标列表
            gaussian_model GaussianModel: 当前使用的高斯模型

        Returns:
            Tensor:(N,3): 完成坐标变换之后的世界坐标系下的坐标
        """
    R = torch.tensor(R, dtype=torch.float32, device=gaussian_model.get_xyz.device)
    T = torch.tensor(T, dtype=torch.float32, device=gaussian_model.get_xyz.device)
    return torch.matmul(cam_pos, R) + T


def PerspectiveProj(image, FoVx, FoVy, cam_pos):
    """
        完成相机坐标系到像素坐标系的转换（透视投影），返回像素坐标

        Args:
            image Tensor:(3,H,W): 渲染后的图像，提供像素平面尺寸
            FoVx float: 水平视场角
            FoVy float: 垂直视场角
            cam_pos Tensor:(N,3): 相机坐标系下的坐标列表

        Returns:
            Tensor:(N,2): 完成透视投影之后的像素坐标
    """
    image_H, image_W = image.shape[-2:]
    # 将 FoVx 和 FoVy 转换为 tensor 类型
    FoVx = torch.tensor(FoVx, dtype=torch.float32)
    FoVy = torch.tensor(FoVy, dtype=torch.float32)
    # 计算焦距
    f_x = image_W / (2 * torch.tan(FoVx / 2))
    f_y = image_H / (2 * torch.tan(FoVy / 2))
    # 假设相机的主点位于图像中心
    c_x = image_W / 2
    c_y = image_H / 2
    # 获取 transformed_positions 中的每个高斯体的坐标 (N, 3)
    X_camera = cam_pos[:, 0]
    Y_camera = cam_pos[:, 1]
    Z_camera = cam_pos[:, 2]
    # 通过投影公式转换到像素坐标
    x_pixel = f_x * X_camera / Z_camera + c_x
    y_pixel = f_y * Y_camera / Z_camera + c_y
    # 将计算得到的像素坐标合并为 (N, 2) 的张量，也是和visibility_filter中的内容是一一对应的
    return torch.stack((x_pixel, y_pixel), dim=1)


def valid_pixel_filter(image, pixel_coordinates, invDepth, mono_invdepth):
    """
        找到满足条件的像素坐标，返回一组bool元素的tensor数组
        这些坐标应当满足不超过图像边界（有些高斯体渲染后的中心不在图像内）
        也应当满足不超过先验深度输出图像的尺寸和渲染深度输出图像尺寸（这两个大概率是一样的）

        Args:
            image Tensor:(3,H,W): 渲染后的图像，提供像素平面尺寸
            pixel_coordinates Tensor:(N,2): 等待筛选的像素坐标
            invDepth Tensor:(1,H,W): 渲染后的深度
            mono_invdepth Tensor:(1,H,W): 先验深度

        Returns:
            Tensor:(N,): 筛选之后的像素坐标bool列表，不满足条件的为false
    """
    image_H, image_W = image.shape[-2:]
    # 取出 x_pixel 和 y_pixel
    x_pixel = pixel_coordinates[:, 0]
    y_pixel = pixel_coordinates[:, 1]
    # 检查 x 和 y 是否在有效的像素范围内
    valid_x = (x_pixel >= 0) & (x_pixel <= image_W)
    valid_y = (y_pixel >= 0) & (y_pixel <= image_H)
    # 检查像素坐标是否在invDepth和mono_invdepth有效范围内
    inv_depth_height, inv_depth_width = invDepth.shape[1], invDepth.shape[2]
    mono_invdepth_height, mono_invdepth_width = mono_invdepth.shape[1], mono_invdepth.shape[2]
    valid_inv_coords = (x_pixel < inv_depth_width) & (y_pixel < inv_depth_height)
    valid_mono_coords = (x_pixel < mono_invdepth_width) & (y_pixel < mono_invdepth_height)
    return valid_x & valid_y & valid_inv_coords & valid_mono_coords


def least_squares(feature, target):
    """
        最小二乘法函数
        在这里的主要功能是，找到一对k和b，使得k*先验深度+b=相机坐标系深度
        为什么这么做？因为3DGS官方代码将渲染深度和先验深度已经做了归一化，但是相机坐标系还没有归一化
        本操作使得相机坐标系和这两个深度处于同一种尺度，才能进行比较和修改

        Args:
            feature Tensor:(N,1): 一般是渲染深度（因为先验深度和渲染深度是归一化的，所以找到渲染深度和相机坐标系的关系，就等于找到先验深度和相机坐标系的关系）
            target Tensor:(N,1): 一般是相机坐标深度

        Returns:
            Tensor:(),Tensor:(): 返回一对k和b
    """
    # 构造输入矩阵 X，第一列是 toadd_inv_depth，第二列是全1，用来表示偏置项 b 形状为 (N, 2)
    X = torch.cat([feature, torch.ones_like(feature)], dim=1)
    y = target  # (N, 1) 目标向量 y 是 toadd_depth_info
    XT_X = X.T @ X  # 形状为 (2, 2)
    XT_y = X.T @ y  # 形状为 (2, 1)

    try:
        # 使用 torch.linalg.solve 代替手动求逆，保证数值稳定性
        k_b = torch.linalg.solve(XT_X, XT_y)  # 形状为 (2, 1)
    except torch._C._LinAlgError as e:
        global fail_cnt
        print("矩阵求解失败:", e)
        print("XT_X:\n", XT_X)
        print("XT_y:\n", XT_y)
        fail_cnt = fail_cnt + 1
        print(f"fail {fail_cnt} times already!")
        # input()
        return 0, 0, 0

        # pdb.set_trace()  # 进入交互式调试模式
    # 提取 k 和 b，满足k*toadd_inv_depth+b=toadd_depth_info
    return k_b[0, 0], k_b[1, 0], 1


def set_z(self, new_z, indices):
    """
        修改高斯体z坐标的方法，这个方法会被添加到GaussianModel类中

        Args:
            new_z Tensor:(N,): 新的z
            indices Tensor:(N,1,1): 高斯体编号（后面两个傻鸟维度不用担心，会squeeze）

        Returns:
            void
    """

    if indices is None:
        raise ValueError("New z indices must have length.")
    else:
        indices = indices.view(-1)
        self._xyz[indices, 2] = new_z
        # print("end")


def gs_adjustment(valid_coordinates, pixel_coordinates, visibility_filter, invDepth, mono_invdepth,
                  transformed_positions, gaussians, viewpoint_cam, radii):
    """
        根据阈值对高斯体做增加和删除操作

        Args:
            valid_coordinates :平面像素坐标


        Returns:
            void
    """
    if torch.all(~valid_coordinates):  # 或者 torch.all(tensor == False)
        # print("No valid_coordinates")
        return

    visible_gaussian_indices = visibility_filter.squeeze()
    # 深度信息有效性检查，depth_info和visibility_filter中的内容是一一对应的
    depth_info = transformed_positions[:, 2]  # 获取 Z 坐标
    min_depth = depth_info.min()

    valid_pixel_coordinates = pixel_coordinates[valid_coordinates]  # 输出有效的像素坐标
    valid_gaussian_indices = visibility_filter[
        valid_coordinates]  # 输出有效的像素坐标对应的高斯体编号（visibility_filter里面存的就是高斯体编号，所以这里索引出来的一定也是高斯体编号）
    valid_depth_info = depth_info[valid_coordinates]
    valid_depth_info = valid_depth_info.view(-1, 1)

    # 方案一：将渲染深度和depth_anything先验深度进行比对
    # 利用投影矩阵将渲染深度转化为线性，然后再尝试归一化
    epsilon = 1e-6
    linear_invdepth = -viewpoint_cam.projection_matrix[2, 2] / (
            invDepth - viewpoint_cam.projection_matrix[2, 3] + epsilon)
    linear_monoinv_depth = -viewpoint_cam.projection_matrix[2, 2] / (
            mono_invdepth - viewpoint_cam.projection_matrix[2, 3] + epsilon)
    y_coords = valid_pixel_coordinates[:, 0].to(torch.long)  # 形状为 (N,)
    x_coords = valid_pixel_coordinates[:, 1].to(torch.long)  # 形状为 (N,)

    valid_inv_depth = linear_invdepth[0, x_coords, y_coords]
    valid_monoinv_depth = linear_monoinv_depth[0, x_coords, y_coords]
    valid_inv_depth = valid_inv_depth.view(-1, 1)
    valid_monoinv_depth = valid_monoinv_depth.view(-1, 1)
    k, b, isSuccess = least_squares(valid_inv_depth, valid_depth_info)
    if isSuccess:
        valid_inv_depth = k * valid_inv_depth + b
        valid_monoinv_depth = k * valid_monoinv_depth + b
    else:
        print("failed to calculate least_square")
        return
    # 深度可视化，这段基本可以验证viewpoint_cam.invdepthmap[0]是COLMAP重建过程中的相机视角深度值，不然不会这么清晰
    # invdepthmap_np = viewpoint_cam.invdepthmap[0].cpu().numpy()  # (892, 1600)
    # plt.figure(figsize=(10, 5))
    # plt.imshow(1/invdepthmap_np, cmap='jet')  # 选择合适的 colormap，如 'gray'、'jet'
    # plt.colorbar(label="Inverse Depth")
    # plt.title("Inverse Depth Map Visualization")
    # plt.axis("off")  # 关闭坐标轴
    # plt.show()

    valid_depth_diff = valid_inv_depth - valid_monoinv_depth  # 计算 valid_inv_depth 和 valid_monoinv_depth之间的差值
    # print(f"depth_diff.max():{torch.max(valid_depth_diff)}")
    # 确保非空
    if valid_depth_diff.numel() > 0:
        # print(f"depth_diff.max():{torch.max(valid_depth_diff)}")
        indices = torch.nonzero((valid_depth_diff > 8) | (valid_depth_diff < -8))[:, 0]  # 找出差值大于8
        if indices.size(0) > 0:
            # 这里筛选出来的toadd_gaussian_indices是绝对的高斯编号
            toadd_gaussian_indices = valid_gaussian_indices[indices]
            # 这里的toadd_depth_info已经是相机坐标系下的深度了
            toadd_depth_info = valid_depth_info[indices]
            toadd_inv_depth = valid_inv_depth[indices]
            toadd_monoinv_depth = valid_monoinv_depth[indices]

            visible_gaussian_indices, _ = torch.sort(visible_gaussian_indices)
            # 这一步是求出了toadd_gaussian_indices所列编号在visible_gaussian_indices中的下标数值。因为toadd_gaussian_indices是高斯编号
            # 因为visible_gaussian_indices和transformed_positions长度一致，所以下标同一。因此我们可以用这个下标去索引所有的相机坐标系下要改的高斯体坐标
            # searchsorted的意思是找到toadd_gaussian_indices内容在visible_gaussian_indices对应的下标，需要保证两个都有序
            matching_indices = torch.searchsorted(visible_gaussian_indices, toadd_gaussian_indices)

            # toadd_gaussian_xyz = gaussians.get_xyz[toadd_gaussian_indices]
            toadd_gaussian_xyz = transformed_positions[matching_indices]  # 利用matching_indices索引
            toadd_gaussian_xyz = toadd_gaussian_xyz.squeeze()
            # 易错点，检查是否变成了一维张量 (3,)，如果是，可以使用 unsqueeze 恢复为二维张量
            if toadd_gaussian_xyz.ndimension() == 1:
                toadd_gaussian_xyz = toadd_gaussian_xyz.unsqueeze(0)  # 恢复为 (1, 3)
            # print(toadd_gaussian_xyz.shape)
            toadd_x_coords = toadd_gaussian_xyz[:, 0]  # 提取相机坐标系的 x 坐标
            toadd_y_coords = toadd_gaussian_xyz[:, 1]  # 提取相机坐标系的 y 坐标
            # 这里求出来的toadd_z_coords，是相机坐标系下应该有的深度。需要变换为世界坐标系的深度，才能修改进高斯体
            toadd_z_coords = toadd_monoinv_depth
            toadd_z_coords = toadd_z_coords.squeeze()
            if toadd_z_coords.ndimension() == 0:  # 变成标量
                toadd_z_coords = toadd_z_coords.unsqueeze(0)  # 恢复成形状为 (1,)
            toadd_cam_xyz = torch.stack((toadd_x_coords, toadd_y_coords, toadd_z_coords),
                                        dim=-1)  # 新的 (N, 3) 坐标
            toadd_world_xyz = CtoW(viewpoint_cam.R, viewpoint_cam.T, toadd_cam_xyz, gaussians)

            GaussianModel.set_z = set_z

            new_z = toadd_world_xyz[:, 2].squeeze()
            new_z_mask = new_z >= 0
            if torch.all(~new_z_mask):
                print("No new_z is >0, considering throwing all new_z!")
            else:
                new_z = new_z[new_z_mask]  # 仍然是 (M,)
                toadd_gaussian_indices = toadd_gaussian_indices[new_z_mask]  # 仍然是 (M,1)
                print(new_z.shape)
                gaussians.set_z(new_z, toadd_gaussian_indices)  # 这里因为要改高斯体了，所以需要使用绝对的编号
            # for idx, ta_gs_xyz in enumerate(toadd_gaussian_xyz):
            #     # 打印编号和深度信息TransWld_Coor和Wld_Coor应该相近，Cam_Coor高斯体相机坐标系坐标，
            #     print(f"Gaussian number {toadd_gaussian_indices[idx]} - Cam_Coor: {ta_gs_xyz} - "
            #           f"TransWld_Coor: {toadd_world_xyz[idx]} - Wld_"
            #           f"Coor: {gaussians.get_xyz[toadd_gaussian_indices[idx]]}")

            # for xyz in toadd_new_xyz:
            #     # 传入了高斯体的半径变量
            #     gaussians.add_point_using_closest_colmap(xyz, radii)
            # print("end")
