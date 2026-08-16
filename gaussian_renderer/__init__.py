import torch
import math
import time
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


import torch

def quantile_histogram(tensor, q, bins=4096):
    flat = tensor.detach().flatten().float()
    if flat.numel() == 0:
        return torch.tensor(0.0, device=tensor.device)
    min_val = flat.min()
    max_val = flat.max()
    if min_val == max_val:
        return min_val
    hist = torch.histc(flat, bins=bins, min=min_val, max=max_val)
    cumsum = torch.cumsum(hist, dim=0)
    total = cumsum[-1]
    target = q * total
    idx = torch.searchsorted(cumsum, target)
    idx = idx.clamp(0, bins - 1)
    bin_width = (max_val - min_val) / bins
    left_edge = min_val + idx * bin_width
    cum_before = cumsum[idx - 1] if idx > 0 else 0
    bin_count = hist[idx]
    frac = (target - cum_before) / bin_count if bin_count > 0 else 0.0
    return left_edge + frac * bin_width


def quantize_sh_to_int8_two_scale(float_sh, quantile=0.9998):
    orig_shape = float_sh.shape
    P = orig_shape[0]

    if float_sh.dim() == 2 and float_sh.shape[1] == 48:
        float_sh_3d = float_sh.reshape(P, 16, 3)
    elif float_sh.dim() == 3 and float_sh.shape[1] == 3 and float_sh.shape[2] != 3:
        float_sh_3d = float_sh.transpose(1, 2).contiguous()
    elif float_sh.dim() == 3 and float_sh.shape[2] == 3:
        float_sh_3d = float_sh
    else:
        raise ValueError(f"Unsupported shape: {orig_shape}")

    x_abs = float_sh_3d[:, :, 0].abs().flatten()
    y_abs = float_sh_3d[:, :, 1].abs().flatten()
    z_abs = float_sh_3d[:, :, 2].abs().flatten()

    if quantile < 1.0:
        max_abs_x = quantile_histogram(x_abs, quantile)
        max_abs_y = quantile_histogram(y_abs, quantile)
        max_abs_z = quantile_histogram(z_abs, quantile)
    else:
        max_abs_x = x_abs.max()
        max_abs_y = y_abs.max()
        max_abs_z = z_abs.max()

    scale_x = max_abs_x / 127.0
    scale_y = max_abs_y / 127.0
    scale_z = max_abs_z / 127.0

    scale_vec = torch.tensor([scale_x, scale_y, scale_z], dtype=float_sh.dtype, device=float_sh.device)
    quantized_3d = torch.round(float_sh_3d / scale_vec)
    quantized_3d = torch.clamp(quantized_3d, -127, 127)
    int8_3d = quantized_3d.to(torch.int8)

    if orig_shape == float_sh_3d.shape:
        int8_sh = int8_3d
    elif len(orig_shape) == 2 and orig_shape[1] == 48:
        int8_sh = int8_3d.reshape(P, 48)
    else:
        int8_sh = int8_3d.transpose(1, 2).contiguous()

    scales_half = torch.tensor([[scale_x.item(), scale_y.item(), scale_z.item()]], dtype=torch.half, device=float_sh.device)
    return int8_sh, scales_half


def quantize_sh_to_int8_degree_wise(float_sh, quantile=0.9998):
    orig_shape = float_sh.shape
    P = orig_shape[0]

    # 1. 形状标准化为 (P, 16, 3)
    if float_sh.dim() == 2 and float_sh.shape[1] == 48:
        float_sh_3d = float_sh.reshape(P, 16, 3)
    elif float_sh.dim() == 3 and float_sh.shape[1] == 3 and float_sh.shape[2] != 3:
        float_sh_3d = float_sh.transpose(1, 2).contiguous()
    elif float_sh.dim() == 3 and float_sh.shape[2] == 3:
        float_sh_3d = float_sh
    else:
        raise ValueError(f"Unsupported shape: {orig_shape}")

    # 2. 定义 degree 范围 (0..3 阶共 16 个基函数)
    degree_ranges = [(0, 1), (1, 4), (4, 9), (9, 16)]  # [start, end)

    scales = torch.zeros(4, 3, dtype=float_sh.dtype, device=float_sh.device)

    # 3. 对每个 degree、每个通道计算 max_abs 或分位数
    for d, (start, end) in enumerate(degree_ranges):
        for c in range(3):
            vals = float_sh_3d[:, start:end, c].abs().flatten()
            if quantile < 1.0:
                max_abs = quantile_histogram(vals, quantile)
            else:
                max_abs = vals.max()
            scales[d, c] = max_abs / 127.0

    # 4. 广播 scales 到 (16, 3) 以便一次性量化
    scale_mat = torch.zeros(16, 3, dtype=float_sh.dtype, device=float_sh.device)
    for d, (start, end) in enumerate(degree_ranges):
        scale_mat[start:end, :] = scales[d:d+1, :]

    quantized_3d = torch.round(float_sh_3d / scale_mat)
    quantized_3d = torch.clamp(quantized_3d, -127, 127)
    int8_3d = quantized_3d.to(torch.int8)

    # 5. 恢复原始形状
    if orig_shape == float_sh_3d.shape:
        int8_sh = int8_3d
    elif len(orig_shape) == 2 and orig_shape[1] == 48:
        int8_sh = int8_3d.reshape(P, 48)
    else:
        int8_sh = int8_3d.transpose(1, 2).contiguous()

    # 6. 输出 12 个 half scale（拍平）
    scales_half = scales.flatten().half()  # shape (12,)
    return int8_sh, scales_half



def quantize_sh_to_int8(shs_float):
    if shs_float.dim() == 3:
        P = shs_float.shape[0]
        shs_flat = shs_float.reshape(P, 48)
    else:
        shs_flat = shs_float
        P = shs_flat.shape[0]

    r_vals = shs_flat[:, 0::3]
    g_vals = shs_flat[:, 1::3]
    b_vals = shs_flat[:, 2::3]

    max_r  = torch.max(torch.abs(r_vals), dim=1).values
    max_gb = torch.max(torch.maximum(torch.abs(g_vals), torch.abs(b_vals)), dim=1).values

    eps = torch.tensor(1e-8, device=shs_flat.device, dtype=torch.float32)
    max_r  = torch.maximum(max_r, eps)
    max_gb = torch.maximum(max_gb, eps)

    scale_r  = max_r  / 127.0
    scale_gb = max_gb / 127.0

    shs_int8 = torch.empty_like(shs_flat, dtype=torch.int8)
    shs_int8[:, 0::3] = torch.clamp(torch.round(r_vals / scale_r[:, None]),  -127, 127).to(torch.int8)
    shs_int8[:, 1::3] = torch.clamp(torch.round(g_vals / scale_gb[:, None]), -127, 127).to(torch.int8)
    shs_int8[:, 2::3] = torch.clamp(torch.round(b_vals / scale_gb[:, None]), -127, 127).to(torch.int8)

    scales = torch.stack([scale_r, scale_gb], dim=1)
    scales_half = scales.half()
    return shs_int8, scales_half



def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,
           scores=None, scaling_modifier=1.0, override_color=None):

    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    scales = pc.get_scaling  # (N, 3)
    # # 在相机坐标系下，视线方向（Z）上的最大延伸近似为 max_scale 或某种投影
    # # 简单做法：每个高斯取 scale 的最大值作为半长轴的保守估计
    xyz = pc.get_xyz
    R = torch.as_tensor(viewpoint_camera.R, dtype=xyz.dtype, device=xyz.device)
    t = torch.as_tensor(viewpoint_camera.T, dtype=xyz.dtype, device=xyz.device)
    pts_cam = (R @ xyz.T + t.unsqueeze(1)).T
    max_scale_per_gaussian = scales.max(dim=1).values   # (N,)
    # 取 center_depth + 3 * max_scale，3σ覆盖 99.7%
    extended_depth = pts_cam[:, 2] + 8.0 * max_scale_per_gaussian
    max_depth = extended_depth.max().item() * 2
    max_depth = min(max(max_depth, 35), 115)
    # print("max_depth:", max_depth)
    

    screenspace_points = torch.zeros_like(
        pc.get_xyz,
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda"
    ) + 0

    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    if scores is None:
        scores = torch.zeros_like(opacity)

    scales = None
    rotations = None
    cov3D_precomp = None
    sh_scales = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier).half()
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )

            dir_pp = (
                pc.get_xyz
                - viewpoint_camera.camera_center.repeat(
                    pc.get_features.shape[0], 1
                )
            )

            dir_pp_normalized = dir_pp / dir_pp.norm(
                dim=1,
                keepdim=True
            )

            sh2rgb = eval_sh(
                pc.active_sh_degree,
                shs_view,
                dir_pp_normalized
            )

            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        else:
            # if pipe.use_int8_sh:  # 假设有个开关
            shs_float = pc.get_features   # (P,16,3)
            shs, sh_scales = quantize_sh_to_int8_degree_wise(shs_float)
            # else:
            #     shs = pc.get_features
            #     sh_scales = None

    else:
        colors_precomp = override_color

    # =========================================================
    # Rasterizer timing
    # =========================================================

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    rendered_image, radii, kernel_times = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scores=None,
        scales=None,
        cov3D_precomp=cov3D_precomp,
        sh_scales = sh_scales,
        max_depth = max_depth
    )

    end_event.record()

    torch.cuda.synchronize()

    rasterizer_time_ms = start_event.elapsed_time(end_event)

    # =========================================================

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "kernel_times": kernel_times,
        "rasterizer_time_ms": rasterizer_time_ms
    }