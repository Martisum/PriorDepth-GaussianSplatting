# 3D Gaussian Splatting - Prior Depth
**先验深度引导的三维高斯泼溅模型使用手册**
作者：钱俊彦 
指导老师：彭时林
### 说明
本毕业设计项目基于论文 "3D Gaussian Splatting for Real-Time Radiance Field Rendering"的公开代码链接 [gaussian_splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).旨在原来模型的基础上，使用单目深度估计网络得到先验深度，从而约束重建。 


### 致谢
在完成本项目的过程中，感谢彭时林老师的敦促与指导。彭老师定期的组会和周报，让我能够及时整理和回顾当前的工作进展，让毕业设计的计划有条不紊地进行。同时，彭老师在论文结构安排、格式整理和文本撰写方面给予了很大的帮助。此外，感谢朱冬晨老师为我选择了基于三维高斯泼溅模型的场景重建研究这个有趣且实用的方向。在研究和学习的过程中，朱老师给予了很多建设性的建议和帮助。

### 如何使用本项目

#### 克隆本项目
注意，本代码包含子模块。请注意检查submodule中的子模块是否完备。可以在官方代码 [gaussian_splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)中查找补充 
```shell
# HTTPS
git clone https://github.com/Martisum/PriorDepth-GaussianSplatting.git --recursive
```

#### 基础硬件需求
- 具有计算能力 7.0+ 的 CUDA 就绪 GPU
- 24 GB VRAM（用于训练到论文的评估质量）
- 对于较小的 VRAM 配置，请参阅[gaussian_splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)解答

#### 基础软件需求
- Conda（推荐使用，以便于设置）
- PyTorch 扩展的 C++ 编译器（论文使用了适用于 Windows 的 Visual Studio 2019）
  - **注意！请必须使用vs2019！本代码需要且只能安装msvc142！参考博客[diff-gaussian-rasterization的安装](https://blog.csdn.net/qq_52915639/article/details/144890000)，其余本人已经尝试，均会导致高斯光栅化模块安装失败！**
- 适用于 PyTorch 扩展的 CUDA SDK 11，在 Visual Studio 之后安装（论文使用的是 11.8，官方代码声称11.6有问题，但是本人尝试似乎是可以运行的）
- C++ 编译器和 CUDA SDK 必须兼容

#### 本地环境部署
论文提供的默认安装方法基于 Conda 包和环境管理
```shell
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate gaussian_splatting
```
请注意，此过程假定你已安装 CUDA SDK 11，而不是 12。

提示：使用 Conda 下载软件包和创建新环境可能需要大量磁盘空间。默认情况下，Conda 将使用主系统硬盘驱动器。您可以通过指定不同的包下载位置和不同驱动器上的环境来避免这种情况：

```shell
conda config --add pkgs_dirs <Drive>/<pkg_path>
conda env create --file environment.yml --prefix <Drive>/<env_path>/gaussian_splatting
conda activate <Drive>/<env_path>/gaussian_splatting
```
如果您能负担得起磁盘空间，我们建议使用我们的环境文件来设置与我们相同的训练环境。如果您想进行修改，请注意，主要版本更改可能会影响我们方法的结果。但是，我们的（有限的）实验表明，代码库在更新的环境（Python 3.8、PyTorch 2.0.0、CUDA 12）中运行良好。确保创建一个 PyTorch 及其 CUDA 运行时版本匹配的环境，并且安装的 CUDA SDK 与 PyTorch 的 CUDA 版本没有主要版本差异。

#### 已知问题
在 Windows 上构建子模块时遇到问题（cl.exe：找不到文件或类似文件）。请考虑官方代码FAQ中此问题的解决方法。

#### 运行本代码

若试图运行基础3DGS优化器，只需使用。本文存在先验深度约束，因此需要额外进行单目深度估计网络启动。本项目集成有ZoeDepth和相对应的DepthGen.py模块（本模块已弃用，实际上最后是使用DepthAnythingV2模型）来生成单目深度估计灰度图。
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```
关于train.py的命令行参数，请参考原项目，本项目不做赘述。
若想要运行DepthAnythingV2模型（本项目已集成），请参考下方步骤：
- 克隆[Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#usage)：
```
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```
- 从[Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) 下载权重并将其放在 ```Depth-Anything-V2/checkpoints/```
- 生成深度贴图
```
python Depth-Anything-V2/run.py --encoder vitl --pred-only --grayscale --img-path <path to input images> --outdir <output path>
```
- 使用以下方法生成``` depth_params.json ```文件：
```
python utils/make_depth_scale.py --base_dir <path to colmap> --depths_dir <path to generated depths>
```
如果要使用深度正则化``` -d <path 进行深度映射>```，则应在训练时设置一个新参数。
假设您已经安装好了DepthAnythingV2模型，可以参考以下步骤运行代码，本项目已经设计了视频切分脚本，并自动使用Depth-Anything-V2模型，请使用train_video.py执行，并修改视频和单目深度估计网络路径。USE_DEPTHGEN和USE_DEPTH_ANYTHING可以选择使用哪种单目深度估计网络。

#### 数据集选择
请注意，与 MipNeRF360 类似，我们以 1-1.6K 像素范围内的分辨率定位图像。为方便起见，可以传递任意大小的输入，如果其宽度超过 1600 像素，则会自动调整其大小。论文建议保留此行为，但您可以通过设置```-r 1```来强制训练使用更高分辨率的图像。

MipNeRF360 场景由论文作者托管此处 .您可以在[此处](https://jonbarron.info/mipnerf360/)找到论文的 Tanks&Temples 和 Deep Blending 的 SfM 数据集。如果您未提供输出模型目录 （```-m```），则训练的模型将写入```output```目录内具有随机唯一名称的文件夹。此时，可以使用实时查看器查看经过训练的模型（请参阅下文）。

#### 模型评估
默认情况下，经过训练的模型使用数据集中的所有可用图像。要在保留测试集进行评估的同时训练它们，请使用```--eval``` 标志。这样，您可以呈现训练集/测试集并生成错误指标，如下所示：
```shell
python render.py -m <path to trained model> # 生成渲染
python metrics.py -m <path to trained model> # 计算评估指标
```
如果您想评估论文的预训练模型[pre-trained models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip) ，则必须下载相应的源数据集，并使用额外的 ```--source_path/-s``` 标志向 ```render.py``` 指示它们的位置。注意：预训练模型是使用发布代码库创建的。此代码库已清理并包含错误修复，因此您通过评估它们获得的指标将与论文中的指标不同。
```shell
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
python metrics.py -m <path to pre-trained model>
```
关于render.py和metrics.py的命令行参数，请参考原项目，本项目不做赘述。
#### SIBR交互式查看器
论文为本方法提供了两个交互式查看器：远程和实时。论文的查看解决方案基于 [SIBR](https://sibr.gitlabpages.inria.fr/) 框架，该框架由 GRAPHDECO 集团为多个 Novel-View 合成项目开发。

##### 硬件需求
- 支持 OpenGL 4.5 的 GPU 和驱动程序（或最新的 MESA 软件）
- 推荐 4 GB VRAM
- 具有计算能力 7.0+ 的 CUDA 就绪 GPU（仅适用于 Real-Time Viewer）

##### 软件需求
- Visual Studio 或 g++， 而不是 Clang（论文使用了适用于 Windows 的 Visual Studio 2019）
- CUDA SDK 11，在 Visual Studio 之后安装（论文用的是 11.8）
- CMake （最新版本，论文使用的是 3.24）
- 7zip（仅在 Windows 上）

##### 启动SIBR_viewer
本文提供了COLMAP和SIBR、ffmpeg等多个组件，可以很方便地使用SIBR，而无需使用官方的繁琐设定。
这些组件位于项目```external```文件夹中
SIBR 界面提供了几种导航场景的方法。默认情况下，您将从 FPS 导航器开始，您可以使用``` W、A、S、D、Q、E ```进行相机平移，使用``` I、K、J、L、U、O``` 进行旋转。或者，您可能希望使用 ```Trackball ```风格的导航器（从浮动菜单中选择）。您还可以使用``` Snap to``` 按钮从数据集捕捉到摄像机，或使用 ```Snap to closest``` 查找最近的摄像机。浮动菜单还允许您更改导航速度。您可以使用 ```Scaling Modifier ```来控制显示的高斯大小，或显示初始点云。
![aa](/assets/aa_onoff.gif)
对于使用者而言，只需要启动SIBR_viewer.py即可，注意要修改3DGS模型输出路径：
```python
import subprocess

# 选择是否开启评估
isEVAL = False
# output保存路径
model_path = r'D:\Work\AI\PriorDepth-GaussianSplatting\output\M03281006-6'

if isEVAL:
    command = f'python render.py -m {model_path}'
    subprocess.run(command, shell=True)
    command = f'python metrics.py -m {model_path}'
    subprocess.run(command, shell=True)

# 脚本执行
command = f'SIBR_gaussianViewer_app.exe -m {model_path}'
run_path = 'external/viewers/bin'
subprocess.run(command, shell=True, cwd=run_path)
```

### 制作自己的输入数据集（假设不使用本项目的COLMAP）
论文的 COLMAP 加载器在源路径位置需要以下数据集结构：

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```
对于光栅化，相机型号必须是 SIMPLE_PINHOLE 或 PINHOLE 相机。我们提供了一个转换器脚本 convert.py，用于从输入图像中提取未失真的图像和 SfM 信息。或者，您可以使用 ImageMagick 调整未失真图像的大小。这种重新缩放类似于 MipNeRF360，即在相应的文件夹中创建分辨率为原始分辨率 1/2、1/4 和 1/8 的图像。要使用它们，请先安装最新版本的 COLMAP（最好是 CUDA 驱动的）和 ImageMagick。将要使用的图像放在目录 ```<location>/input``` 中。
```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```
如果您的系统路径上有 COLMAP 和 ImageMagick（本项目已经集成在了external上），则只需运行 
```shell
python convert.py -s <location> [--resize] #If not resizing, ImageMagick is not needed
```