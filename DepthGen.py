import os
import numpy as np
from external.scripts import read_write_model
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from PIL import Image
from tqdm import tqdm
import pickle  # 专门拿来存储字典的库

Depth_Sparse = {}  # 稀疏深度矩阵（字典）
Depth_Dense = {}  # 稠密深度矩阵（ZoeDepth）（也是一个字典）
IMG_LENGTH = 0  # 处理的图片总数
Opt_ST_Dict = {}  # 存储优化后的s t数值，键是image_id
Image_W = 1920
Image_H = 1080
Is_Create_DepthMatrix = False  # 检测是否已经存在了先验稠密深度矩阵


def sparse_depth_gen(source_path, sparse_model_path, camera='OPENCV', use_gpu=1):
    global Depth_Sparse, IMG_LENGTH

    # 获取当前工作路径，找到COLMAP
    current_path = os.getcwd()
    colmap_path = os.path.join(current_path, 'external', r'COLMAP-3.8-windows-cuda\colmap.bat')

    # 这一步创建\distorted\sparse目录
    os.makedirs(source_path + r"\distorted\sparse", exist_ok=True)
    ## 单个图像的特征提取
    # 使用特征提取器，提供database.db的存储路径（没有database会按照此路径创建）
    # 提供图像路径，相机类型（默认OPENCV），选择SIFT使用GPU提取特征
    feat_extracton_cmd = colmap_path + " feature_extractor " \
                                       "--database_path " + source_path + "/distorted/database.db \
           --image_path " + source_path + "/input \
           --ImageReader.single_camera 1 \
           --ImageReader.camera_model " + camera + " \
           --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        print(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## 图像间特征匹配
    # 使用穷举匹配器，提供database路径，选择匹配方法是SIFT
    # 结果也是存储在database.db中，文件总数和第一步一样，没有变化
    feat_matching_cmd = colmap_path + " exhaustive_matcher \
        --database_path " + source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        print(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### 捆绑调整（Bundle adjustment）
    # 使用 SfM 对数据集进行稀疏 3D 重建/映射 执行特征提取和匹配
    # 默认的 Mapper 容差不必要地大，减小它可以加快捆绑调整步骤。
    # 执行完这个一步，sparse会多出一个文件夹0，内中含有cameras，images，points3D三个文件，和一个COLMAP工程配置文件project.ini
    # 这三个文件包含了相机位姿，点云位置，images还不知道包含了啥
    mapper_cmd = (colmap_path + " mapper \
        --database_path " + source_path + "/distorted/database.db \
        --image_path " + source_path + "/input \
        --output_path " + source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        print(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)
    # 读取重建的模型
    cameras, images, points3D = read_write_model.read_model(sparse_model_path, ext=".bin")
    # 使用字典构造器构造一个image_id和图片名对应的一个字典，image_id是键，image.name是值，image_id的变化和每个for循环的image_id变化相关
    image_id_to_name = {image_id: image.name for image_id, image in images.items()}

    IMG_LENGTH = len(images)
    # 稀疏深度矩阵字典键值初始化
    for image_id, image in images.items():
        Depth_Sparse[image_id] = {}

    # 获取点云坐标
    for point_id, point in points3D.items():
        print(f"Point ID: {point_id}")
        print(f"Coordinates: {point.xyz}")
        print(f"RGB Color: {point.rgb}")
        print(f"Track Length: {len(point.image_ids)}\n")  # 在几个图片中可以观测到此点
        print(f"error: {point.error}")
        # 查询此点对应哪些图片的像素位置
        print("------Pixel Locate Start------")
        # point.point2D_idxs是这个点在各个图片投影点的索引，不是点的编号
        # point.image_ids是这个点投影到的图片的id
        for image_id, point2D_idx in zip(point.image_ids, point.point2D_idxs):
            image = images[image_id]
            # 这一步就弄出像素特征点在哪里
            point2D = image.xys[point2D_idx]
            # 输出像素坐标
            print(f"Image ID: {image_id}")
            print(f"Image Name: {image_id_to_name[image_id]}")
            print(f"2D Pixel Coordinates: {point2D}")  # point2D.xy 是像素坐标（1920*1080）

            # 准备构建每张图片特征点的像素深度字典
            rotation_raw = image.qvec  # 四元数 (qw, qx, qy, qz)
            # 调整四元数顺序为 [qx, qy, qz, qw] ,不然求出来深度是负值
            rotation_quat = [rotation_raw[1], rotation_raw[2], rotation_raw[3], rotation_raw[0]]
            rotation = R.from_quat(rotation_quat)  # 创建旋转对象并获取旋转矩阵
            rotation_matrix = rotation.as_matrix()
            point_xyz = np.array(point.xyz).reshape(-1, 1)
            point_img = rotation_matrix @ point_xyz + np.array(image.tvec).reshape(-1, 1)  # 转换 3D 点到相机坐标系
            print(point_img[2])  # 求出当前像素的深度值

            point2D = tuple(int(round(val)) for val in point2D)  # 取整
            # point2D不可哈希，需要转换为不可变数值，即转化为元组
            Depth_Sparse[image_id][tuple(point2D)] = (int(point_img[2]), point_id)

        print("------Pixel Locate End------")

    # 获取相机内参
    for camera_id, camera in cameras.items():
        print(f"Camera {camera_id}:")
        print(f"Focal Lengths: {camera.params[0]} (fx), {camera.params[1]} (fy)")
        print(f"Principal Point: {camera.params[2]} (cx), {camera.params[3]} (cy)")


def dense_depth_gen(image_path, sparse_model_path):
    global Depth_Dense, Is_Create_DepthMatrix

    if os.path.isfile("Depth_Dense.pkl"):
        Is_Create_DepthMatrix = True
        print("Depth_Dense.pkl is already exist, still create Depth_Dense.pkl? y/n")
        isGen = input("Input y/n:")
        if isGen != 'y':
            print("Reading data from Depth_Dense.pkl...")
            with open("Depth_Dense.pkl", "rb") as f:
                Depth_Dense = pickle.load(f)
            # print(Depth_Dense)
            return

    Is_Create_DepthMatrix = False
    print("ZoeDepth_start")
    # 使用模型ZoeD_NK
    conf = get_config("zoedepth_nk", "infer")
    model_zoe_nk = build_model(conf)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_nk.to(DEVICE)

    # 读取重建的模型
    cameras, images, points3D = read_write_model.read_model(sparse_model_path, ext=".bin")
    # 创建一个进度条
    with tqdm(total=len(images), desc="ZoeDepth Progress") as pbar:
        for image_id, image in images.items():
            image_PIL = Image.open("data/myvedio2/input/" + image.name).convert("RGB")  # load
            depth_numpy = zoe.infer_pil(image_PIL)  # as numpy
            Depth_Dense[image_id] = depth_numpy
            pbar.update(1)

    print("ZoeDepth done.")


# ---------- s，t 优化器部分 ---------- #
# 定义一个尺度模糊的误差函数，接下来要把这个函数优化到最小，然后取出s t数值
def Fuzzy_error(sparse_model_path, image_id, s, t):
    error_sum = 0
    # 读取重建的模型
    cameras, images, points3D = read_write_model.read_model(sparse_model_path, ext=".bin")
    # 枚举编号为image_id图片的所有特征像素点和稀疏深度，depth_inf包含深度信息和对应的点的id
    for pixel, depth_inf in Depth_Sparse[image_id].items():
        # 注意这里pixel的下标是反过来写的，因为稀疏矩阵中（1920*1080）稠密矩阵是（1080*1920）
        # print(
        #     f"坐标 {pixel} 对应的稀疏深度值是 {depth_inf[0]} 对应稠密深度值是 {Depth_Dense[image_id][pixel[1], pixel[0]]}")
        error_sum = error_sum + ((1 / points3D[depth_inf[1]].error) * depth_inf[0] - (
                s * Depth_Dense[image_id][pixel[1], pixel[0]] + t)) ** 2  # 考虑重投影误差的error计算公式
    return error_sum


def Depth_Optimize(sparse_model_path):
    if not Is_Create_DepthMatrix:
        global Opt_ST_Dict
        # 限定s t的边界 s >= 1, t >= 1
        bounds = [(1, None), (1, None)]
        # 读取重建的模型
        cameras, images, points3D = read_write_model.read_model(sparse_model_path, ext=".bin")

        with tqdm(total=len(images), desc="Optimize Progress（最佳s t优化）") as pbar:
            for image_id, image in images.items():
                # 使用最小优化器进行优化
                result = minimize(lambda params: Fuzzy_error(sparse_model_path, image_id, params[0], params[1]),
                                  np.array([1, 1]), bounds=bounds)
                opt_s, opt_t = result.x
                Opt_ST_Dict[image_id] = (opt_s, opt_t)  # 以元组的形式存储，第一个是opt_s，第二个是opt_t
                # print(f"opt_s: {opt_s}, opt_t: {opt_t}")
                # print(f"error: {Fuzzy_error(sparse_model_path, image_id, opt_s, opt_t)}")
                pbar.update(1)
        # 更新稠密深度矩阵
        with tqdm(total=len(images)*Image_H*Image_W, desc="Matrix Saved Progress") as pbar:
            for image_id, image_depth in Depth_Dense.items():
                for h in range(0, Image_H):
                    for w in range(0, Image_W):
                        Depth_Dense[image_id][h, w] = Depth_Dense[image_id][h, w] * Opt_ST_Dict[image_id][0] + \
                                                      Opt_ST_Dict[image_id][1]
                        pbar.update(1)

        # 将字典保存为文件
        with open("Depth_Dense.pkl", "wb") as f:
            pickle.dump(Depth_Dense, f)
        print("Optimize done. Saved successfully!")
    else:
        print("Depth_Matrix is already created and optimized!")


# ---------- s，t 优化器部分 ---------- #

if __name__ == "__main__":
    folder_path = r"D:\Work\AI\learn\gaussian-splatting\gaussian-splatting\data\myvedio2"
    sparse_path = r"D:\Work\AI\learn\gaussian-splatting\gaussian-splatting\data\myvedio2\distorted\sparse\0"
    img_path = r"D:\Work\AI\learn\gaussian-splatting\gaussian-splatting\data\myvedio2\input"
    sparse_depth_gen(source_path=folder_path, sparse_model_path=sparse_path)
    dense_depth_gen(image_path=img_path, sparse_model_path=sparse_path)
    Depth_Optimize(sparse_model_path=sparse_path)
    print("over")
    # tmp = Fuzzy_error(sparse_model_path=sparse_path, image_id=1, s=1, t=1)
    # print(f"tmp={tmp}")
