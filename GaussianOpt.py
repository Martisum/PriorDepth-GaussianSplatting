import torch


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
