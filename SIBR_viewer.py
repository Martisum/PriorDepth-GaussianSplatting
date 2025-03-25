import subprocess

isEVAL = True
# output保存路径
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\eval_newopt2'  # 更改条件，考虑半径信息
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\new_opt1'  # 更改条件前，还是使用的cam_scene
model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\63ade197-6'  # 更改条件，考虑半径信息但是更换模型
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
