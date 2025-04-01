import subprocess

isEVAL = False
# output保存路径
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-4'  # 堡垒原始
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-3'  # 堡垒优化
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-1'  # 牛角优化
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-2'  # 牛角原始
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-5'  # 堡垒原始，但是带保存进度
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\e20b7013-4'  # room的优化，但删除范围较大，未添加绝对深度约束 c607b2fd-9 M03281006-7 e20b7013-4
model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-6'  # room的原始
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\1d54904e-4'  # 调整参数后对牛角的优化，但是因为非常严格，所以几乎没有优化点，因为几乎没有近相机漂浮物
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\77b99386-7'  # 是
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\ffbf3a47-c'  # 88837801-b是干扰0 psnr=31 优化是c4797669-4 ccb5cfc9-7 0.1cam_ext 0e374e99-5
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\8b72ba27-6'  # 干扰1
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\f9f011b9-4'  # 9f4a386b-a干扰2 psnr=30 上同 优化f9f011b9-4
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\df5f6912-6'  # 干扰3 psnr=31



if isEVAL:
    command = f'python render.py -m {model_path}'
    subprocess.run(command, shell=True)
    command = f'python metrics.py -m {model_path}'
    subprocess.run(command, shell=True)

# 脚本执行
command = f'SIBR_gaussianViewer_app.exe -m {model_path}'
run_path = 'external/viewers/bin'
subprocess.run(command, shell=True, cwd=run_path)
