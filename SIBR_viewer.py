import subprocess

isEVAL = True
# output保存路径
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\eval_newopt2'  # 更改条件，考虑半径信息
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\new_opt1'  # 更改条件前，还是使用的cam_scene
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\47827f66-2'  # 堡垒原始
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\1fb7c6d3-2'  # 堡垒优化
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\b79ac61b-2'  # 牛角优化
# model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\5528ffd2-c'  # 牛角原始
model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\8e075e78-0'  # yqh

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
