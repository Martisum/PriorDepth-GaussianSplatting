import subprocess

isEVAL = False
# output保存路径
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-4'  # 堡垒原始
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-3'  # 堡垒优化
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-1'  # 牛角优化
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-2'  # 牛角原始
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-5'  # 堡垒原始，但是带保存进度
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-7'  # room的优化，但删除范围较大，未添加绝对深度约束
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-6'  # room的原始
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\1d54904e-4'  # 调整参数后对牛角的优化，但是因为非常严格，所以几乎没有优化点，因为几乎没有近相机漂浮物
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\77b99386-7'  # 是
model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\88837801-b'  # 是捣乱的
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\319bcc94-e'  #

if isEVAL:
    command = f'python render.py -m {model_path}'
    subprocess.run(command, shell=True)
    command = f'python metrics.py -m {model_path}'
    subprocess.run(command, shell=True)

# 脚本执行
command = f'SIBR_gaussianViewer_app.exe -m {model_path}'
run_path = 'external/viewers/bin'
subprocess.run(command, shell=True, cwd=run_path)
