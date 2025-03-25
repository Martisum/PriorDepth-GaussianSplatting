import torch
from torch.cuda import device

from scene import GaussianModel
import pdb
import matplotlib.pyplot as plt
import numpy as np

fail_cnt = 0
EPSILON = 1e-6  # 极小量
MAX_LENGTH_TABLE = 10_0000  # 最小二乘法数据上限

#---------- 总长为图像像素总数（二维） ----------
Linear_InvDepth = torch.zeros(())  # 线性化的渲染深度
Linear_MonoDepth = torch.zeros(())  # 线性化的先验深度，这里还是二维的

#---------- 总长为全体高斯体 ----------
Pixel_Coordinate = torch.zeros(())  # 高斯体中心坐标对应的像素坐标
Cam_Coordinate = torch.zeros(())  # 高斯体转化为相机坐标系下的坐标

#---------- 总长为有效高斯体 ----------
VALID_GS_IDX = torch.zeros(())  # 有效的高斯编号
Norm_InvDepth = torch.zeros(())  # 这个是一维的，存的是坐标对应的深度。这个是最小二乘之后转为相机系的深度
Norm_MonoDepth = torch.zeros(())
Norm_Pix_x = torch.zeros(())  # 像素坐标的坐标值
Norm_Pix_y = torch.zeros(())

Feature_Target_Table = torch.zeros((MAX_LENGTH_TABLE, 2), dtype=torch.float32,
                                   device='cuda')  # 存储了历史上所有符合条件的数对，尽量扩充最小二乘法的数据
FT_Index = 0  # 特征数据表需要维护的下标
LEA_k, LEA_b = 0, 0  # 最小二乘法计算出的k和b


def initialize():
    GaussianModel.set_z = set_z


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
            None 生成一个Tensor:(N,2)的完成透视投影之后的像素坐标
    """
    global Pixel_Coordinate
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
    Pixel_Coordinate = torch.stack((x_pixel, y_pixel), dim=1)


def valid_pixel_filter(image, invDepth, mono_invdepth, visible_gaussian_indices):
    """
        找到满足条件的像素坐标，返回一组bool元素的tensor数组
        这些坐标应当满足不超过图像边界（有些高斯体渲染后的中心不在图像内）
        也应当满足不超过先验深度输出图像的尺寸和渲染深度输出图像尺寸（这两个大概率是一样的）
        同时需要满足可见性

        Args:
            image Tensor:(3,H,W): 渲染后的图像，提供像素平面尺寸
            invDepth Tensor:(1,H,W): 渲染后的深度
            mono_invdepth Tensor:(1,H,W): 先验深度
            visible_gaussian_indices Tensor:(M,): 高斯体可见性

        Returns:
            None 生成一个筛选出来准备处理的高斯编号
    """
    global VALID_GS_IDX
    image_H, image_W = image.shape[-2:]
    # 取出 x_pixel 和 y_pixel
    x_pixel = Pixel_Coordinate[:, 0]
    y_pixel = Pixel_Coordinate[:, 1]
    # 检查 x 和 y 是否在有效的像素范围内
    valid_x = (x_pixel >= 0) & (x_pixel <= image_W)
    valid_y = (y_pixel >= 0) & (y_pixel <= image_H)
    # 检查像素坐标是否在invDepth和mono_invdepth有效范围内
    inv_depth_height, inv_depth_width = invDepth.shape[1], invDepth.shape[2]
    mono_invdepth_height, mono_invdepth_width = mono_invdepth.shape[1], mono_invdepth.shape[2]
    valid_inv_coords = (x_pixel < inv_depth_width) & (y_pixel < inv_depth_height)
    valid_mono_coords = (x_pixel < mono_invdepth_width) & (y_pixel < mono_invdepth_height)
    valid_depth = Cam_Coordinate[:, 2] > 0  # 要求深度都是为正数
    valid_mask = valid_x & valid_y & valid_inv_coords & valid_mono_coords & valid_depth

    VALID_GS_IDX = torch.nonzero(valid_mask, as_tuple=True)[0]
    VALID_GS_IDX = VALID_GS_IDX[torch.isin(VALID_GS_IDX, visible_gaussian_indices)]


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


def opacity_modulate(gaussians, invDepth, mono_invdepth, pixel_coordinates, valid_coordinates, gs_idx):
    """
        根据渲染深度和先验深度，对对应的高斯体做不透明度调制
        此模块应该放在高斯优化模块之后

        Args:
            gaussians GaussianModel:高斯模型
            invDepth Tensor:(1,H,W):渲染深度
            mono_invdepth Tensor:(1,H,W):先验深度
            pixel_coordinates Tensor:(N,2):


        Returns:
            void
    """
    """
    新方案的主要思路：
    目前已经完成了对于渲染深度小于先验深度，且
    先看能不能找到删除高斯体的函数，要是不行，就直接把高斯体的不透明度给改成全透明，相当于删除了
    不能使用移动高斯体的方案，因为每个视图的先验深度并不一样，这么做可能会把高斯体移来移去。也就是说，先验深度后面的高斯体不能动，因为有可能这个高斯体就是正确的
    但是先验深度前面的高斯体肯定是错误的，要删掉。
    一定要验证一下densification_postfix函数有没有真正添加了高斯体
    
    VALID_GS_IDX筛选合适的高斯体
    """
    print("none")


def linearization(depth_image, proj_matrix):
    """
           将仿射变换下的depth_image深度转化为线形坐标系的线性深度

           Args:
               depth_image GaussianModel:高斯模型
               proj_matrix Tensor:(1,H,W):渲染深度

           Returns:

    """
    global EPSILON
    # return -proj_matrix[2, 2] / (depth_image - proj_matrix[2, 3] + EPSILON)
    return 1 / (depth_image + EPSILON)
    # return depth_image


def update_feature_target_table(tmp_pair):
    """
        循环队列更新 Feature_Target_Table，确保其大小不会超过 max_length。
        如果超过上限，则从头开始覆盖最早的数据。

        Args:
            tmp_pair (Tensor): (N, 2) 的新增数据

        Returns:
            None,更新Feature_Target_Table和FT_Index
    """
    global Feature_Target_Table, FT_Index, MAX_LENGTH_TABLE
    add_length = tmp_pair.shape[0]  # 新增数据的行数
    if add_length > MAX_LENGTH_TABLE:  # MAX_LENGTH_TABLE比较小的时候可能会出现tmp_pair本身就超过了MAX_LENGTH_TABLE，导致出现错误
        tmp_pair = tmp_pair[:MAX_LENGTH_TABLE, :]  # 仅保留前 MAX_LENGTH_TABLE 个数据
        add_length = MAX_LENGTH_TABLE  # 重新计算插入长度

    if FT_Index + add_length <= MAX_LENGTH_TABLE:
        Feature_Target_Table[FT_Index:FT_Index + add_length, :] = tmp_pair
        FT_Index = FT_Index + add_length
    else:
        # 先填充到末尾，再从头开始覆盖
        end_part = MAX_LENGTH_TABLE - FT_Index
        Feature_Target_Table[FT_Index:MAX_LENGTH_TABLE, :] = tmp_pair[:end_part, :]  # 填充到末尾
        start_part = add_length - end_part
        Feature_Target_Table[:add_length - end_part, :] = tmp_pair[end_part:, :]  # 从头覆盖
        FT_Index = start_part  # 确保 index 重新回到合法范围


def depth_normalization():
    """
        利用最小二乘归一化深度，输出归一化深度的表

        Args:
           NULL

        Returns:
            void
    """
    global LEA_k, LEA_b, Feature_Target_Table, Pixel_Coordinate, VALID_GS_IDX, Norm_MonoDepth, Norm_InvDepth, Norm_Pix_x, Norm_Pix_y
    Norm_Pix_x = Pixel_Coordinate[VALID_GS_IDX][:, 1].to(torch.long)  # [VALID_GS_IDX]筛选出合适高斯体的像素坐标，[:, 1]取横坐标，再转整形
    Norm_Pix_y = Pixel_Coordinate[VALID_GS_IDX][:, 0].to(torch.long)
    valid_inv_depth = Linear_InvDepth[0, Norm_Pix_x, Norm_Pix_y].view(-1, 1)  # 索引出渲染深度的具体值
    valid_monoinv_depth = Linear_MonoDepth[0, Norm_Pix_x, Norm_Pix_y].view(-1, 1)

    device = torch.device('cuda:0')
    Feature_Target_Table = Feature_Target_Table.to(device)  # to(device)转移到相同设备
    tmp_pair = torch.cat((valid_inv_depth, Cam_Coordinate[:, 2][VALID_GS_IDX].unsqueeze(dim=1)),
                         dim=1)  # [:, 2]取相机系Z坐标，[VALID_GS_IDX]取合适高斯体，unsqueeze(dim=1)增加一个维度
    update_feature_target_table(tmp_pair)
    # 最小二乘法求出k和b，最后计算出归一化的深度
    LEA_k, LEA_b, isSuccess = least_squares(Feature_Target_Table[:, 0:1], Feature_Target_Table[:, 1:2])
    LEA_k,LEA_b=1,0
    if isSuccess:
        Norm_InvDepth = LEA_k * valid_inv_depth + LEA_b
        Norm_MonoDepth = LEA_k * valid_monoinv_depth + LEA_b
    # print("end")


def visualize_inv_depth(invDepth):
    """
    可视化逆深度图

    参数:
    invDepth (numpy.ndarray): 逆深度的二维张量
    """
    # 如果 invDepth 是 PyTorch 张量，转换为 NumPy
    if isinstance(invDepth, torch.Tensor):
        invDepth = invDepth.detach().cpu().numpy()  # 先移到 CPU 再转 NumPy

    # 处理 NaN 和无穷值
    invDepth = np.nan_to_num(invDepth, nan=0.0, posinf=0.0, neginf=0.0)

    # 归一化到 0-1 之间，防止数值过大或过小影响可视化
    # min_val, max_val = np.min(invDepth), np.max(invDepth)
    # if max_val > min_val:  # 避免除零错误
    #     norm_invDepth = (invDepth - min_val) / (max_val - min_val)
    # else:
    #     norm_invDepth = np.zeros_like(invDepth)  # 全部相同值则显示纯色

    # 绘制图像
    plt.figure(figsize=(8, 6))
    plt.imshow(invDepth, cmap='jet', interpolation='nearest')
    plt.colorbar(label="Linear_MonoDepth Depth")
    plt.title("Linear_MonoDepth Depth Visualization")
    plt.axis("off")  # 关闭坐标轴
    plt.show(block=False)  # 非阻塞显示
    plt.pause(0.1)  # 短暂暂停，确保窗口刷新
    input("Press Enter to continue...")  # 等待手动关闭
    plt.close()  # 关闭窗口

def plot_invdepth_vs_z(Norm_InvDepth, Cam_Z):
    """
    绘制 Norm_InvDepth 和相机坐标 Z 之间的散点图
    :param Norm_InvDepth: 归一化的逆深度，一维张量或数组
    :param Cam_Z: 相机坐标系中的 Z 值，一维张量或数组
    """
    # 转换为 NumPy 数组，确保可以进行绘图
    Norm_InvDepth = Norm_InvDepth.cpu().numpy() if hasattr(Norm_InvDepth, 'cpu') else np.array(Norm_InvDepth)
    Cam_Z = Cam_Z.cpu().numpy() if hasattr(Cam_Z, 'cpu') else np.array(Cam_Z)

    # 确保数据不是空的
    if len(Norm_InvDepth) == 0 or len(Cam_Z) == 0:
        print("Warning: Empty input data!")
        return

    # 取前100个点
    Norm_InvDepth = Norm_InvDepth[:100]
    Cam_Z = Cam_Z[:100]

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(Norm_InvDepth, Cam_Z, s=10, alpha=0.6, c='blue', edgecolors='none')
    plt.xlabel("Normalized Inverse Depth")
    plt.ylabel("Camera Coordinate Z")
    plt.title("Scatter Plot of Norm_InvDepth vs. Cam_Z")
    plt.grid(True, linestyle='--', alpha=0.5)

    # 尝试非阻塞显示
    plt.show(block=False)
    plt.pause(0.1)


def floatingObj_prune(gaussians, cam_extent, radii):
    """
        删除近相机漂浮物

        Args:
            invDepth :渲染深度图
            mono_invdepth :先验深度图
            gaussians :高斯模型，包含所有高斯体的信息
            cam_extent :场景相机尺度，用于动态调整阈值

        Returns:
            void
    """
    global VALID_GS_IDX, Norm_Pix_x, Norm_Pix_y, Norm_InvDepth, Norm_MonoDepth
    if VALID_GS_IDX.numel() == 0:
        # print("no VALID_GS_IDX!")
        return
    diff_mask = ((Norm_MonoDepth - Norm_InvDepth) > 0.5*cam_extent).squeeze()
    tmp_diff = Cam_Coordinate[:, 2][VALID_GS_IDX].squeeze() - 0.6 * radii[VALID_GS_IDX]

    # plot_invdepth_vs_z(Norm_InvDepth.squeeze()[:1000], Cam_Coordinate[:, 2][VALID_GS_IDX].squeeze()[:1000])
    # input()

    # if tmp_diff.min() < 0:
    #     print("jijiji")
    diff_mask = torch.logical_and(diff_mask, tmp_diff < Norm_InvDepth.squeeze())
    if diff_mask.sum() == 0:  # 全为false则无需继续
        return

    selected_gs_idx = VALID_GS_IDX[diff_mask.squeeze()]
    prune_filter = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device=selected_gs_idx.device)
    prune_filter[selected_gs_idx] = True
    gaussians.prune_points(prune_filter)
    print(selected_gs_idx.shape)
    """
    思路：
    找到符合如下条件的高斯体中心位置，并删除：
        A.渲染深度要小于先验深度（足够说明先验深度之前存在高斯体的遮挡）
        B.渲染深度要在高斯体中或者后（渲染深度如果在高斯体中或者在此高斯体后面，足以说明这个高斯体遮挡了渲染，需要删除）
    为什么算法有效？
        假设不满足条件A，那么应该是这个位置存在未补足的细节，而不是选择删除高斯体，应该用其他方法处理。
        假设不满足条件B，说明渲染深度在高斯体的前面，那么我们无法判断这个高斯体是否正确。在其它视角中，这个高斯体可能是正确的
        问：如果先验深度面前存在大量遮挡高斯体，你如何保证删除？
        答：可以肯定的是，每一次迭代都会删掉一定数量的遮挡高斯体。只要迭代的次数足够多，就可以不断地删除这些遮挡的高斯体
        问：你如何保证不会错误删除？
        答：如果枚举到的高斯体的深度比渲染深度还要浅，并且先验深度又远大于渲染深度，那么可以认为渲染深度在计算的过程中一定统计了这个高斯体的不透明度，
            并且这个渲染深度是错的。错误的原因，是这个高斯体一定为阻碍渲染深度做出了贡献，那么它肯定就是近相机漂浮物之一，需要删除
    """


def gs_adjustment(invDepth, mono_invdepth, gaussians, viewpoint_cam, radii):
    """
        根据阈值对高斯体做增加和删除操作

        Args:
            invDepth :渲染深度图
            mono_invdepth :先验深度图
            gaussians :高斯模型，包含所有高斯体的信息
            viewpoint_cam :场景相机模型

        Returns:
            void
    """
    global EPSILON, Linear_InvDepth, Linear_MonoDepth, VALID_GS_IDX, Feature_Target_Table, LEA_k, LEA_b
    if VALID_GS_IDX.numel() == 0:
        # print("no VALID_GS_IDX!")
        return

    if Cam_Coordinate[:, 2][VALID_GS_IDX].min() < 0:
        print(f"There is an unexpected depth {Cam_Coordinate[:, 2].min()} exists!")
        input()
        return

    Linear_InvDepth = linearization(invDepth, viewpoint_cam.projection_matrix)  # 将非线性的深度线性化
    Linear_MonoDepth = linearization(mono_invdepth, viewpoint_cam.projection_matrix)

    valid_pix_X = Pixel_Coordinate[VALID_GS_IDX][:, 1].to(torch.long)  # [VALID_GS_IDX]筛选出合适高斯体的像素坐标，[:, 1]取横坐标，再转整形
    valid_pix_Y = Pixel_Coordinate[VALID_GS_IDX][:, 0].to(torch.long)
    valid_inv_depth = Linear_InvDepth[0, valid_pix_X, valid_pix_Y].view(-1, 1)  # 索引出渲染深度的具体值
    valid_monoinv_depth = Linear_MonoDepth[0, valid_pix_X, valid_pix_Y].view(-1, 1)

    device = torch.device('cuda:0')
    Feature_Target_Table = Feature_Target_Table.to(device)  # to(device)转移到相同设备
    tmp_pair = torch.cat((valid_inv_depth, Cam_Coordinate[:, 2][VALID_GS_IDX].unsqueeze(dim=1)),
                         dim=1)  # [:, 2]取相机系Z坐标，[VALID_GS_IDX]取合适高斯体，unsqueeze(dim=1)增加一个维度
    update_feature_target_table(tmp_pair)
    LEA_k, LEA_b, isSuccess = least_squares(Feature_Target_Table[:, 0:1], Feature_Target_Table[:, 1:2])
    if isSuccess:
        valid_inv_depth = LEA_k * valid_inv_depth + LEA_b
        valid_monoinv_depth = LEA_k * valid_monoinv_depth + LEA_b
    else:
        print("failed to calculate least_square")
        return

    # 这里添加不透明度调制

    abs_diff_mask = (torch.abs(Cam_Coordinate[:, 2][VALID_GS_IDX].unsqueeze(dim=1) - valid_monoinv_depth) > (
            radii[VALID_GS_IDX] + 5).unsqueeze(dim=1)).squeeze(
        1)
    if abs_diff_mask.sum() == 0:  # 全为false则无需继续
        return
    VALID_GS_IDX = VALID_GS_IDX[abs_diff_mask]  # 更新需要更改的高斯体的下标
    valid_monoinv_depth = valid_monoinv_depth[abs_diff_mask]  # 同样全部都筛掉那个不符合条件的

    new_cam_x_coords = Cam_Coordinate[:, 0][VALID_GS_IDX]  # [:, 0]提取相机坐标系的 x 坐标，[VALID_GS_IDX]选取要修改的高斯体
    new_cam_y_coords = Cam_Coordinate[:, 1][VALID_GS_IDX]
    new_cam_z_coords = valid_monoinv_depth.view(-1)
    try:
        new_cam_xyz = torch.stack((new_cam_x_coords, new_cam_y_coords, new_cam_z_coords),
                                  dim=-1)  # 新的 (N, 3) 坐标
    except Exception as e:
        input()
        print("aaa")
    new_world_xyz = CtoW(viewpoint_cam.R, viewpoint_cam.T, new_cam_xyz, gaussians)

    new_z = new_world_xyz[:, 2].squeeze()
    print(new_z.shape)
    gaussians.set_z(new_z, VALID_GS_IDX)  # 这里因为要改高斯体了，所以需要使用绝对的编号


initialize()  # 文件初始化
