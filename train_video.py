import os
import subprocess
import DepthGen

USE_DEPTHGEN = False
USE_DEPTH_ANYTHING = True  # 使用官方的先验深度

# 视频绝对路径
video_path = r"D:\Work\AI\data\yqh_xzyl\yqh_xzyl.mp4"
depth_anything_path = r"D:\Work\AI\Depth-Anything-V2-main"
# 切分帧数，每秒多少帧
fps = 2

# 获取当前工作路径
current_path = os.getcwd()
# 上一级文件夹所在路径
folder_path = os.path.dirname(video_path)
# 图片保存路径
images_path = os.path.join(folder_path, 'input')
# 创建一个路径，用来存储切分之后的图像
os.makedirs(images_path, exist_ok=True)

ffmpeg_path = os.path.join(current_path, 'external', r'ffmpeg/bin/ffmpeg.exe')

# 脚本运行
# 视频切分脚本，使用ffmpeg软件实现切割
command = f'{ffmpeg_path} -i {video_path} -qscale:v 1 -qmin 1 -vf fps={fps} {images_path}\\%04d.jpg'
subprocess.run(command, shell=True)
# # COLMAP估算相机位姿
# if USE_DEPTHGEN:
#     # 使用了先验深度，包含相机位姿估算、点云生成，无需重复特征匹配
#     DepthGen.sparse_depth_gen(source_path=folder_path, sparse_model_path=folder_path + r'\distorted\sparse\0')
#     DepthGen.dense_depth_gen(image_path=folder_path + r'\input', sparse_model_path=folder_path + r'\distorted\sparse\0')
#     DepthGen.Depth_Optimize(source_path=folder_path, sparse_model_path=folder_path + r'\distorted\sparse\0')
#     command = f'python convert.py -s {folder_path} --skip_matching'
# else:
#     command = f'python convert.py -s {folder_path}'
# subprocess.run(command, shell=True)
# # 模型训练脚本，模型会保存在output路径下
# if USE_DEPTH_ANYTHING:
#     colmap_model_path = folder_path + r'\distorted'
#     command = f'python {depth_anything_path}/run.py --encoder vitl --pred-only --grayscale --img-path {images_path} --outdir {depth_anything_path}'
#     subprocess.run(command, shell=True)
#     command = f'python utils/make_depth_scale.py --base_dir {colmap_model_path} --depths_dir {depth_anything_path}'
#     subprocess.run(command, shell=True)
#     command = f'python train.py -s {folder_path} -d {depth_anything_path}'
#     subprocess.run(command, shell=True)
# else:
#     command = f'python train.py -s {folder_path}'
#     subprocess.run(command, shell=True)
