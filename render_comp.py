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
from scene import Scene as OriginalScene
import os
import sys
import json
import time
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render as original_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

# 导入 FlashGS 所需依赖
import flash_gaussian_splatting


# =========================================================================
# FlashGS 组件重新封装（加前缀避免命名冲突）
# =========================================================================
class FlashGS_Scene:
    def __init__(self, device):
        self.device = device
        self.num_vertex = 0
        self.position = None
        self.shs = None
        self.opacity = None
        self.cov3d = None

    def loadPly(self, scene_path):
        self.num_vertex, self.position, self.shs, self.opacity, self.cov3d = flash_gaussian_splatting.ops.loadPly(scene_path)
        print(f"[FlashGS] Loaded PLY with num_vertex = {self.num_vertex}")
        self.position = self.position.to(self.device)
        self.shs = self.shs.to(self.device)
        self.opacity = self.opacity.to(self.device)
        self.cov3d = self.cov3d.to(self.device)

class FlashGS_Camera:
    def __init__(self, camera_json):
        self.id = camera_json['id']
        self.img_name = camera_json['img_name']
        self.width = camera_json['width']
        self.height = camera_json['height']
        self.position = torch.tensor(camera_json['position'])
        self.rotation = torch.tensor(camera_json['rotation'])
        self.focal_x = camera_json['fx']
        self.focal_y = camera_json['fy']
        self.zFar = 100.0
        self.zNear = 0.01

class FlashGS_Rasterizer:
    def __init__(self, scene, MAX_NUM_RENDERED, MAX_NUM_TILES):
        self.gaussian_keys_unsorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_unsorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)
        self.gaussian_keys_sorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_sorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)

        self.MAX_NUM_RENDERED = MAX_NUM_RENDERED
        self.MAX_NUM_TILES = MAX_NUM_TILES
        self.SORT_BUFFER_SIZE = flash_gaussian_splatting.ops.get_sort_buffer_size(MAX_NUM_RENDERED)
        self.list_sorting_space = torch.zeros(self.SORT_BUFFER_SIZE, device=scene.device, dtype=torch.int8)
        self.ranges = torch.zeros((MAX_NUM_TILES, 2), device=scene.device, dtype=torch.int32)
        self.curr_offset = torch.zeros(1, device=scene.device, dtype=torch.int32)

        self.points_xy = torch.zeros((scene.num_vertex, 2), device=scene.device, dtype=torch.float32)
        self.rgb_depth = torch.zeros((scene.num_vertex, 4), device=scene.device, dtype=torch.float32)
        self.conic_opacity = torch.zeros((scene.num_vertex, 4), device=scene.device, dtype=torch.float32)

    def forward(self, scene, camera, bg_color):
        self.curr_offset.fill_(0)
        flash_gaussian_splatting.ops.preprocess(
            scene.position, scene.shs, scene.opacity, scene.cov3d,
            camera.width, camera.height, 16, 16,
            camera.position, camera.rotation,
            camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
            self.points_xy, self.rgb_depth, self.conic_opacity,
            self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
            self.curr_offset
        )
        
        num_rendered = int(self.curr_offset.cpu()[0])
        if num_rendered >= self.MAX_NUM_RENDERED:
            raise RuntimeError("Too many k-v pairs! Increase MAX_NUM_RENDERED.")
        t2 = time.time()
        flash_gaussian_splatting.ops.sort_gaussian(
            num_rendered, camera.width, camera.height, 16, 16,
            self.list_sorting_space,
            self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
            self.gaussian_keys_sorted, self.gaussian_values_sorted
        )
        torch.cuda.synchronize()
        t3 = time.time()
        elapsed_ms = (t3 - t2) * 1000
        print("===flash sort_gaussian %f ms ====" % elapsed_ms)

        t4 = time.time()
        out_color = torch.zeros((camera.height, camera.width, 3), device=scene.device, dtype=torch.int8)
        flash_gaussian_splatting.ops.render_16x16(
            num_rendered, camera.width, camera.height,
            self.points_xy, self.rgb_depth, self.conic_opacity,
            self.gaussian_keys_sorted, self.gaussian_values_sorted,
            self.ranges, bg_color, out_color
        )
        torch.cuda.synchronize()
        t5 = time.time()
        elapsed_ms = (t5 - t4) * 1000
        print("===render_16x16 %f ms ====" % elapsed_ms)

        return out_color

# =========================================================================
# 核心集成性能测试函数
# =========================================================================
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    print(f"\n==================== [Benchmark Start: {name}] ====================")
    num_views = len(views)
    print(f"[INFO] Evaluating {num_views} views.")

    if num_views == 0:
        print(f"\n[INFO] Skipping {name} set because it contains 0 views/cameras.")
        return
    
    device = torch.device('cuda:0')
    
    # ---------------------------------------------------------------------
    # 准备工作: 对齐并加载 FlashGS 对应的资源
    # ---------------------------------------------------------------------
    scene_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    camera_path = os.path.join(model_path, "cameras.json")
    
    flash_scene = FlashGS_Scene(device)
    flash_scene.loadPly(scene_path)
    
    with open(camera_path, 'r') as f:
        cameras_json = json.loads(f.read())
    
    # 构建 FlashGS 相机字典进行对齐
    flash_cam_dict = {cam['img_name']: FlashGS_Camera(cam) for cam in cameras_json}
    aligned_flash_cameras = []
    
    for view in views:
        img_name = view.image_name
        # 兼容处理文件名可能带或不带扩展名的情况
        if img_name in flash_cam_dict:
            aligned_flash_cameras.append(flash_cam_dict[img_name])
        else:
            base_name = os.path.splitext(img_name)[0]
            if base_name in flash_cam_dict:
                aligned_flash_cameras.append(flash_cam_dict[base_name])
            else:
                # 模糊匹配退路
                matched = False
                for k, v in flash_cam_dict.items():
                    if k in img_name or img_name in k:
                        aligned_flash_cameras.append(v)
                        matched = True
                        break
                if not matched:
                    raise ValueError(f"Cannot find matched FlashGS camera for 3DGS view: {img_name}")

    # 初始化 FlashGS 静态内存分配光栅化器
    MAX_NUM_RENDERED = 2 ** 27
    MAX_NUM_TILES = 2 ** 20
    flash_rasterizer = FlashGS_Rasterizer(flash_scene, MAX_NUM_RENDERED, MAX_NUM_TILES)
    # flash_bg_color = torch.zeros(3, dtype=torch.float32, device=device) # 黑色背景对应 [0,0,0]
    flash_bg_color = torch.zeros(3, dtype=torch.float32)



    # ==========================================================
    # 步骤 1: 统一预热 (Warming Up)
    # ==========================================================
    print(f"[{name}] Warming up both renderers...")
    warmup_iters = min(10, num_views)
    with torch.no_grad():
        for i in range(warmup_iters):
            _ = original_render(views[i], gaussians, pipeline, background)
            _ = flash_rasterizer.forward(flash_scene, aligned_flash_cameras[i], flash_bg_color)
    torch.cuda.synchronize()

    # ==========================================================
    # 步骤 2: 单循环混合性能测试 (Interleaved Benchmark)
    # ==========================================================
    print(f"[{name}] ====== Benchmarking Both Renderers in a Single Loop ======")
    
    # 初始化累计时间
    orig_total_time_ms = 0.0
    flash_total_time_ms = 0.0
    
    # 预先创建 CUDA 时间记录器，避免在循环中重复创建造成开销
    starter_orig = torch.cuda.Event(enable_timing=True)
    ender_orig = torch.cuda.Event(enable_timing=True)
    starter_flash = torch.cuda.Event(enable_timing=True)
    ender_flash = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # 使用 zip 将原生 3DGS 视角和 FlashGS 相机对齐，合入同一个循环
        for idx, (view, flash_cam) in enumerate(zip(views, aligned_flash_cameras)):
            
            # --- 2.1 原生 3DGS 渲染与计时 ---
            print("====speedy gs====\n")
            torch.cuda.synchronize()  # 确保显卡队列清空
            starter_orig.record()
            _ = original_render(view, gaussians, pipeline, background)["render"]
            ender_orig.record()
            torch.cuda.synchronize()  # 阻塞等待当前帧渲染完毕
            orig_total_time_ms += starter_orig.elapsed_time(ender_orig)

            # --- 2.2 FlashGS 渲染与计时 ---
            print("====flash gs====\n")
            torch.cuda.synchronize()  # 确保显卡队列清空
            starter_flash.record()
            _ = flash_rasterizer.forward(flash_scene, flash_cam, flash_bg_color)
            ender_flash.record()
            torch.cuda.synchronize()  # 阻塞等待当前帧渲染完毕
            flash_total_time_ms += starter_flash.elapsed_time(ender_flash)
            
            # 每 20 帧打印一次实时战况（可选，不需要可以删掉）
            if (idx + 1) % 20 == 0 or (idx + 1) == num_views:
                print(f" Progress: {idx + 1}/{num_views} frames checked...")

    # 计算最终 FPS
    orig_fps = 1000.0 * num_views / orig_total_time_ms
    flash_fps = 1000.0 * num_views / flash_total_time_ms

    # ==========================================================
    # 性能对比报告
    # ==========================================================
    print(f"\n==================== [Performance Report: {name}] ====================")
    print(f"Number of Rendered Frames : {num_views}")
    print("-" * 60)
    print(f"Original 3DGS Renderer    : Total Time: {orig_total_time_ms:.2f} ms | Avg Per Frame: {orig_total_time_ms/num_views:.2f} ms | FPS: {orig_fps:.2f}")
    print(f"FlashGS Renderer          : Total Time: {flash_total_time_ms:.2f} ms | Avg Per Frame: {flash_total_time_ms/num_views:.2f} ms | FPS: {flash_fps:.2f}")
    print("-" * 60)
    speedup = flash_fps / orig_fps if orig_fps > 0 else 0
    print(f"Result: FlashGS is {speedup:.2f}x as fast as Original 3DGS (Strict Frame-by-Frame Test).")
    print("===================================================================\n")

    # ==========================================================
    # 性能测试 1: 原生 3DGS 渲染测试
    # ==========================================================
    # print(f"[{name}] 1/2 Warming up Original 3DGS...")
    # warmup_iters = min(10, num_views)
    # with torch.no_grad():
    #     for i in range(warmup_iters):
    #         _ = original_render(views[i], gaussians, pipeline, background)
    # torch.cuda.synchronize()

    # print(f"[{name}] 1/2 Benchmarking Original 3DGS Pure Rendering...")
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # torch.cuda.synchronize()
    # starter.record()
    # with torch.no_grad():
    #     for view in views:
    #         _ = original_render(view, gaussians, pipeline, background)["render"]
    # ender.record()
    # torch.cuda.synchronize()
    
    # orig_total_time_ms = starter.elapsed_time(ender)
    # orig_fps = 1000.0 * num_views / orig_total_time_ms

    # # ==========================================================
    # # 性能测试 2: FlashGS 渲染测试
    # # ==========================================================
    # print(f"[{name}] 2/2 Warming up FlashGS...")
    # with torch.no_grad():
    #     for i in range(warmup_iters):
    #         _ = flash_rasterizer.forward(flash_scene, aligned_flash_cameras[i], flash_bg_color)
    # torch.cuda.synchronize()

    # print(f"[{name}] 2/2 Benchmarking FlashGS Pure Rendering...")
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # torch.cuda.synchronize()
    # starter.record()
    # with torch.no_grad():
    #     for flash_cam in aligned_flash_cameras:
    #         _ = flash_rasterizer.forward(flash_scene, flash_cam, flash_bg_color)
    # ender.record()
    # torch.cuda.synchronize()
    
    # flash_total_time_ms = starter.elapsed_time(ender)
    # flash_fps = 1000.0 * num_views / flash_total_time_ms

    # # ==========================================================
    # # 输出最终性能对比报告
    # # ==========================================================
    # print(f"\n==================== [Performance Report: {name}] ====================")
    # print(f"Number of Rendered Frames : {num_views}")
    # print("-" * 60)
    # print(f"Original 3DGS Renderer    : Total Time: {orig_total_time_ms:.2f} ms | Avg Per Frame: {orig_total_time_ms/num_views:.2f} ms | FPS: {orig_fps:.2f}")
    # print(f"FlashGS Renderer          : Total Time: {flash_total_time_ms:.2f} ms | Avg Per Frame: {flash_total_time_ms/num_views:.2f} ms | FPS: {flash_fps:.2f}")
    # print("-" * 60)
    # speedup = flash_fps / orig_fps if orig_fps > 0 else 0
    # print(f"Result: FlashGS is {speedup:.2f}x as fast as Original 3DGS.")
    # print("===================================================================\n")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = OriginalScene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    
    print("Evaluating Performance for model: " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)