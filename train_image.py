import os
import subprocess

# 图片集所在的绝对路径
images_path = r"D:\Work\AI\data\yqh_xzyl\input"
depth_anything_path = r"D:\Work\AI\Depth-Anything-V2-main"

# 上一级文件夹所在路径
folder_path = os.path.dirname(images_path)

# 脚本运行
command = f'python convert.py -s {folder_path}'
subprocess.run(command, shell=True)
colmap_model_path = folder_path + r'\distorted'
command = f'python {depth_anything_path}/run.py --encoder vitl --pred-only --grayscale --img-path {images_path} --outdir {depth_anything_path}'
subprocess.run(command, shell=True)
command = f'python utils/make_depth_scale.py --base_dir {colmap_model_path} --depths_dir {depth_anything_path}'
subprocess.run(command, shell=True)
command = f'python train.py -s {folder_path} -d {depth_anything_path} --eval'
subprocess.run(command, shell=True)
