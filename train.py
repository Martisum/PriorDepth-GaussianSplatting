#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import torch
from random import randint

from sympy.physics.paulialgebra import epsilon

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# class Gaussian_Body:
#     def __init__(self, index: int, depth: float, pixel: tuple, radius: float, opacity: float):
#         self.index = index
#         self.depth = depth
#         self.pixel = pixel
#         self.radius = radius
#         self.opacity = opacity

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    global invDepth, mono_invdepth
    is_depth_feedback = False

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        is_depth_available = False  # 重置状态

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer,
                                       use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)[
                        "render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        # gs_body = []  # 用来存储筛选出来的对此像素有影响的所有高斯体
        # r_image_height, r_image_width = image.shape[-2:]
        # h_step, w_step = int(r_image_height * 0.1), int(r_image_width * 0.1)  # 只枚举10%的图片，全部枚举太费时间
        # for h in range(0, r_image_height, h_step):
        #     for w in range(0, r_image_width, w_step):
        #         # 筛选出对这个像素有影响的所有高斯体
        #         for gs_idx in visibility_filter:
        #             gs_idx = int(gs_idx.item())
        #             # 变换此高斯点坐标到相机坐标系，从而提取深度
        #             R_torch = torch.tensor(viewpoint_cam.R, dtype=torch.float32, device=gaussians.get_xyz.device)
        #             T_torch = torch.tensor(viewpoint_cam.T, dtype=torch.float32, device=gaussians.get_xyz.device)
        #             gs_cam_point = (R_torch @ gaussians.get_xyz[gs_idx].T + T_torch).flatten()
        #             # 检查是否存在负深度，保证变换正确
        #             if gs_cam_point[2].item() <= 0:
        #                 print(f"det(R)={torch.det(R_torch)},RT*R={R_torch.T @ R_torch}")
        #                 print(f"error z{gs_cam_point[2].item()}")
        #             # 归一化设备坐标
        #             x_n, y_n = gs_cam_point[0].item() / gs_cam_point[2].item(), gs_cam_point[1].item() / gs_cam_point[
        #                 2].item()
        #             # 求出像素平面坐标
        #             u = (x_n / (2 * np.tan(viewpoint_cam.FoVx / 2))) * r_image_width + (r_image_width / 2)
        #             v = (y_n / (2 * np.tan(viewpoint_cam.FoVy / 2))) * r_image_height + (r_image_height / 2)
        #             if u > 0 and u < r_image_width and v > 0 and v < r_image_height:
        #                 if (w - u) ** 2 + (h - v) ** 2 <= radii[gs_idx].item() ** 2:
        #                     gs_body.append(Gaussian_Body(index=gs_idx, depth=gs_cam_point[2].item(), pixel=(u, v),
        #                                                  radius=radii[gs_idx].item(),
        #                                                  opacity=gaussians.get_opacity[gs_idx].item()))
        #
        #         print("end")

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            is_depth_available = True
            # invDepth是渲染过程中出来的高斯体深度
            invDepth = render_pkg["depth"]
            # mono_invdepth是depth-anything出来的先验深度
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            # 选择是否启用深度约束
            if is_depth_feedback:
                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth_pure = 0
                Ll1depth = 0
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render,
                            (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                            dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Gaussian Optimization Module（可能这一步要放在致密化之前，因为致密化改变了高斯体个数，使得渲染结果的可见性和高斯体总数对不上）
            if is_depth_available:
                # print(f"gaussians.get_xyz shape: {gaussians.get_xyz.shape}")
                visible_gaussian_indices = visibility_filter.squeeze()
                # print(f"visible_gaussian_indices: {torch.max(visible_gaussian_indices)}")
                visible_gaussians = gaussians.get_xyz[visible_gaussian_indices]
                # 变换此高斯点坐标到相机坐标系，从而提取深度
                R_torch = torch.tensor(viewpoint_cam.R, dtype=torch.float32, device=gaussians.get_xyz.device)
                T_torch = torch.tensor(viewpoint_cam.T, dtype=torch.float32, device=gaussians.get_xyz.device)
                relative_positions = visible_gaussians - T_torch  # 先平移（从世界坐标系到相机坐标系原点）
                transformed_positions = torch.matmul(relative_positions, R_torch.T)

                # 深度信息有效性检查，depth_info和visibility_filter中的内容是一一对应的
                depth_info = transformed_positions[:, 2]  # 获取 Z 坐标
                min_depth = depth_info.min()

                # 求出像素坐标，从而找到对应先验深度
                image_H, image_W = image.shape[-2:]
                FoVx, FoVy = viewpoint_cam.FoVx, viewpoint_cam.FoVy
                # 将 FoVx 和 FoVy 转换为 tensor 类型
                FoVx = torch.tensor(FoVx, dtype=torch.float32)
                FoVy = torch.tensor(FoVy, dtype=torch.float32)
                # 计算焦距
                f_x = image_W / (2 * torch.tan(FoVx / 2))
                f_y = image_H / (2 * torch.tan(FoVy / 2))
                # 假设相机的主点位于图像中心
                c_x = image_W / 2
                c_y = image_H / 2
                # 获取 transformed_positions 中的每个高斯体的坐标 (N, 3)
                X_camera = transformed_positions[:, 0]
                Y_camera = transformed_positions[:, 1]
                Z_camera = transformed_positions[:, 2]
                # 通过投影公式转换到像素坐标
                x_pixel = f_x * X_camera / Z_camera + c_x
                y_pixel = f_y * Y_camera / Z_camera + c_y
                # 将计算得到的像素坐标合并为 (N, 2) 的张量，也是和visibility_filter中的内容是一一对应的
                pixel_coordinates = torch.stack((x_pixel, y_pixel), dim=1)

                # 像素坐标有效性检查
                # 取出 x_pixel 和 y_pixel
                x_pixel = pixel_coordinates[:, 0]
                y_pixel = pixel_coordinates[:, 1]
                # 检查 x 和 y 是否在有效的像素范围内
                valid_x = (x_pixel >= 0) & (x_pixel <= image_W)
                valid_y = (y_pixel >= 0) & (y_pixel <= image_H)
                # 检查像素坐标是否在invDepth和mono_invdepth有效范围内
                inv_depth_height, inv_depth_width = invDepth.shape[1], invDepth.shape[2]
                mono_invdepth_height, mono_invdepth_width = mono_invdepth.shape[1], mono_invdepth.shape[2]
                valid_inv_coords = (x_pixel < inv_depth_width) & (y_pixel < inv_depth_height)
                valid_mono_coords = (x_pixel < mono_invdepth_width) & (y_pixel < mono_invdepth_height)

                valid_coordinates = valid_x & valid_y & valid_inv_coords & valid_mono_coords  # 合并四个条件，得到有效的像素坐标
                valid_pixel_coordinates = pixel_coordinates[valid_coordinates]  # 输出有效的像素坐标
                valid_gaussian_indices = visibility_filter[valid_coordinates]  # 输出有效的像素坐标对应的高斯体编号
                valid_depth_info = depth_info[valid_coordinates]
                # for i, (coord, depth) in enumerate(zip(valid_pixel_coordinates, valid_depth_info)):
                #     print(f"有效像素坐标 {coord} 对应的高斯体编号: {valid_gaussian_indices[i]}, 深度: {depth}")

                # 方案一：将渲染深度和depth_anything先验深度进行比对
                y_coords = valid_pixel_coordinates[:, 0].to(torch.long)  # 形状为 (N,)
                x_coords = valid_pixel_coordinates[:, 1].to(torch.long)  # 形状为 (N,)
                epsilon = 1e-6
                valid_inv_depth = 1 / (invDepth[0, x_coords, y_coords]+epsilon)
                valid_monoinv_depth = 1 / (mono_invdepth[0, x_coords, y_coords]+epsilon)
                # 输出有效像素对应的深度倒数
                # for i, inv_depth in enumerate(valid_inv_depth):
                #     print(f"有效像素坐标 {valid_pixel_coordinates[i]} 对应的深度倒数: {1/inv_depth.item()}")
                valid_depth_diff = valid_inv_depth - valid_monoinv_depth  # 计算 valid_inv_depth 和 valid_monoinv_depth
                # 之间的差值
                # 确保非空
                if valid_depth_diff.numel() > 0:
                    # print(f"depth_diff.max():{torch.max(valid_depth_diff)}")
                    indices = torch.nonzero(valid_depth_diff > 5)  # 找出差值大于5的下标，若非空则认为这些点过远，需要在先验位置处添加高斯体
                    if indices.size(0) > 0:
                        # 这里筛选出来的toadd_gaussian_indices是绝对的高斯编号
                        toadd_gaussian_indices = valid_gaussian_indices[indices]
                        # 这里的toadd_depth_info已经是相机坐标系下的深度了
                        toadd_depth_info = valid_depth_info[indices]
                        toadd_inv_depth = valid_inv_depth[indices]
                        toadd_monoinv_depth = valid_monoinv_depth[indices]
                        # 最小二乘法求k b
                        toadd_inv_depth = torch.tensor(toadd_inv_depth, dtype=torch.float32)
                        toadd_depth_info = torch.tensor(toadd_depth_info, dtype=torch.float32)
                        # 将数据调整为 (N, 1) 形状
                        toadd_inv_depth = toadd_inv_depth.view(-1, 1)
                        toadd_depth_info = toadd_depth_info.view(-1, 1)
                        # 构造输入矩阵 X，第一列是 toadd_inv_depth，第二列是全1，用来表示偏置项 b 形状为 (N, 2)
                        X = torch.cat([toadd_inv_depth, torch.ones_like(toadd_inv_depth)], dim=1)
                        y = toadd_depth_info  # (N, 1) 目标向量 y 是 toadd_depth_info
                        XT_X = X.T @ X  # 形状为 (2, 2)
                        XT_y = X.T @ y  # 形状为 (2, 1)
                        # 使用 torch.linalg.solve 代替手动求逆，保证数值稳定性
                        k_b = torch.linalg.solve(XT_X, XT_y)  # 形状为 (2, 1)
                        # 提取 k 和 b，满足k*toadd_inv_depth+b=toadd_depth_info
                        k = k_b[0, 0]
                        b = k_b[1, 0]

                        visible_gaussian_indices, _ = torch.sort(visible_gaussian_indices)
                        # 这一步是求出了toadd_gaussian_indices所列编号在visible_gaussian_indices中的下标数值。因为toadd_gaussian_indices是高斯编号
                        # 因为visible_gaussian_indices和transformed_positions长度一致，所以下标同一。因此我们可以用这个下标去索引所有的相机坐标系下要改的高斯体坐标
                        matching_indices = torch.searchsorted(visible_gaussian_indices, toadd_gaussian_indices)

                        # toadd_gaussian_xyz = gaussians.get_xyz[toadd_gaussian_indices]
                        toadd_gaussian_xyz = transformed_positions[matching_indices]  # 利用matching_indices索引
                        toadd_gaussian_xyz = toadd_gaussian_xyz.squeeze()
                        # 易错点，检查是否变成了一维张量 (3,)，如果是，可以使用 unsqueeze 恢复为二维张量
                        if toadd_gaussian_xyz.ndimension() == 1:
                            toadd_gaussian_xyz = toadd_gaussian_xyz.unsqueeze(0)  # 恢复为 (1, 3)
                        print(toadd_gaussian_xyz.shape)
                        toadd_x_coords = toadd_gaussian_xyz[:, 0]  # 提取相机坐标系的 x 坐标
                        toadd_y_coords = toadd_gaussian_xyz[:, 1]  # 提取相机坐标系的 y 坐标
                        # 这里求出来的toadd_z_coords，是相机坐标系下应该有的深度。需要变换为世界坐标系的深度，才能修改进高斯体
                        toadd_z_coords = toadd_monoinv_depth * k + b
                        toadd_z_coords = toadd_z_coords.squeeze()
                        if toadd_z_coords.ndimension() == 0:  # 变成标量
                            toadd_z_coords = toadd_z_coords.unsqueeze(0)  # 恢复成形状为 (1,)
                        toadd_cam_xyz = torch.stack((toadd_x_coords, toadd_y_coords, toadd_z_coords),
                                                    dim=-1)  # 新的 (N, 3) 坐标
                        toadd_world_xyz = torch.matmul(toadd_cam_xyz, R_torch) + T_torch  # 先旋转再平移
                        gaussians.set_z(toadd_world_xyz[:, 2], toadd_gaussian_indices)  # 这里因为要改高斯体了，所以需要使用绝对的编号
                        # for idx, ta_gs_xyz in enumerate(toadd_gaussian_xyz):
                        #     # 打印编号和深度信息TransWld_Coor和Wld_Coor应该相近，Cam_Coor高斯体相机坐标系坐标，
                        #     print(f"Gaussian number {toadd_gaussian_indices[idx]} - Cam_Coor: {ta_gs_xyz} - "
                        #           f"TransWld_Coor: {toadd_world_xyz[idx]} - Wld_"
                        #           f"Coor: {gaussians.get_xyz[toadd_gaussian_indices[idx]]}")

                        # for xyz in toadd_new_xyz:
                        #     # 传入了高斯体的半径变量
                        #     gaussians.add_point_using_closest_colmap(xyz, radii)
                        # print("end")


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,
                                                radii)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import torch

    print(torch.version.cuda)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
