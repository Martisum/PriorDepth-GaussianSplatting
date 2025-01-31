import subprocess

# output保存路径
model_path = r'D:\Work\AI\learn\gaussian-splatting\gaussian-splatting\output\88c8a6ef-c'

# 脚本执行
command = f'SIBR_gaussianViewer_app.exe -m {model_path}'
run_path = 'external/viewers/bin'
subprocess.run(command, shell=True, cwd=run_path)
