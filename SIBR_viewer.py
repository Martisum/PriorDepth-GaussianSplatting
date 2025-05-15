import subprocess

# isEVAL = True
isEVAL = False
# output保存路径
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-1'
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-2'
model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-3'
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-4'

# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\OPC-1'
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\OPC-2'
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\OPC-3'
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\OPC-4'

if isEVAL:
    command = f'python render.py -m {model_path}'
    subprocess.run(command, shell=True)
    command = f'python metrics.py -m {model_path}'
    subprocess.run(command, shell=True)

# 脚本执行
command = f'SIBR_gaussianViewer_app.exe -m {model_path}'
run_path = 'external/viewers/bin'
subprocess.run(command, shell=True, cwd=run_path)
