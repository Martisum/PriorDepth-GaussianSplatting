import subprocess

isEVAL = True
# output保存路径
model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\9131a5e0-7'
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\eval_origin'

if isEVAL:
    command = f'python render.py -m {model_path}'
    subprocess.run(command, shell=True)
    command = f'python metrics.py -m {model_path}'
    subprocess.run(command, shell=True)

# 脚本执行
command = f'SIBR_gaussianViewer_app.exe -m {model_path}'
run_path = 'external/viewers/bin'
subprocess.run(command, shell=True, cwd=run_path)
