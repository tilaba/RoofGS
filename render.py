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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # ==========================================================
    # 步骤 1: Warming Up (预热)
    # ==========================================================
    print(f"[{name}] Warming up GPU...")
    warmup_iters = min(10, len(views))
    with torch.no_grad():
        for i in range(warmup_iters):
            _ = render(views[i], gaussians, pipeline, background)

    # ==========================================================
    # 步骤 2: FPS Benchmark (纯渲染性能测试)
    # ==========================================================
    print(f"[{name}] ======Benchmarking FPS (Pure Rendering)...=====")
    torch.cuda.synchronize()
    
    # 使用 CUDA Event 记录 GPU 内部真实的起始和结束时间
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    total_rasterizer_time_ms = 0.0
    for view in views:
        # print("view.image_name:", view.image_name)
        results = render(view, gaussians, pipeline, background)
        rendered_image = results["render"]
        rasterizer_time_ms = results["kernel_times"]
        total_rasterizer_time_ms += rasterizer_time_ms[0]
        # print("rasterizer_time_ms[0]", rasterizer_time_ms[0]/(1.0))
    if (len(views) > 0):
        avg_rasterizer_time_ms = total_rasterizer_time_ms / len(views)
        fps = 1000.0 / avg_rasterizer_time_ms
        print(f"Average FPS (based on rasterizer kernel time): {fps:.2f}")
        
    # ==========================================================
    # 步骤 3: 保存图片 (独立循环，不计入 FPS)
    # # ==========================================================
    print(f"[{name}] Saving images to disk...")
    with torch.no_grad():
    
        for idx, view in enumerate(tqdm(views, desc="Saving results")):
            rendering = render(view, gaussians, pipeline, background)["render"]
            
            # 保存渲染图
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            
            # 只有第一次运行或需要对比时才保存 GT
            if not os.path.exists(os.path.join(gts_path, '{0:05d}'.format(idx) + ".png")):
                gt = view.original_image[0:3, :, :]
                torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # for cam in scene.getTestCameras():
        #     cam.image_height = 512
        #     cam.image_width = 512

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)