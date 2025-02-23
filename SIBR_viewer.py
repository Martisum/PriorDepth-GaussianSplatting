import subprocess

# output保存路径
model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\02011830'
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\02091922'
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\9ded5d31-f'

# 脚本执行
command = f'SIBR_gaussianViewer_app.exe -m {model_path}'
run_path = 'external/viewers/bin'
subprocess.run(command, shell=True, cwd=run_path)
