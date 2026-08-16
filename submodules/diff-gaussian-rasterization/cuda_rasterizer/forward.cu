/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <math_functions.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h> 
namespace cg = cooperative_groups;



__device__ __forceinline__ glm::vec3 computeColorFromSH_D3_int8(
    int idx, int deg, glm::vec3 pos, glm::vec3 campos,
    const int8_t* shs, const __half* scales_half)
{
    const int8_t* sh_ptr = shs + idx * 48;
    int8_t local_sh[48];
    const int4* src = reinterpret_cast<const int4*>(sh_ptr);
    int4* dst = reinterpret_cast<int4*>(local_sh);
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];

    float s[12];
    const __half* hs = scales_half;
    for (int i = 0; i < 12; ++i) s[i] = __half2float(hs[i]);

    auto sh = [&](int i, int ch) {
        int d = (i == 0) ? 0 : (i <= 3) ? 1 : (i <= 8) ? 2 : 3;
        return (float)local_sh[i*3 + ch] * s[d*3 + ch];
    };

    glm::vec3 dir = pos - campos;
    float inv = rsqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    dir *= inv;
    float x = dir.x, y = dir.y, z = dir.z;

    float r = SH_C0 * sh(0,0) + 0.5f;
    float g = SH_C0 * sh(0,1) + 0.5f;
    float b = SH_C0 * sh(0,2) + 0.5f;

    float t = SH_C1 * y;
    r -= t * sh(1,0); g -= t * sh(1,1); b -= t * sh(1,2);
    t = SH_C1 * z;
    r += t * sh(2,0); g += t * sh(2,1); b += t * sh(2,2);
    t = SH_C1 * x;
    r -= t * sh(3,0); g -= t * sh(3,1); b -= t * sh(3,2);

    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, yz = y*z, xz = x*z;

    float t2;
    t2 = SH_C2[0] * xy;
    r += t2 * sh(4,0); g += t2 * sh(4,1); b += t2 * sh(4,2);
    t2 = SH_C2[1] * yz;
    r += t2 * sh(5,0); g += t2 * sh(5,1); b += t2 * sh(5,2);
    t2 = SH_C2[2] * (3.0f*zz - 1.0f);
    r += t2 * sh(6,0); g += t2 * sh(6,1); b += t2 * sh(6,2);
    t2 = SH_C2[3] * xz;
    r += t2 * sh(7,0); g += t2 * sh(7,1); b += t2 * sh(7,2);
    t2 = SH_C2[4] * (xx - yy);
    r += t2 * sh(8,0); g += t2 * sh(8,1); b += t2 * sh(8,2);

    float t3;
    t3 = SH_C3[0] * y * (3.0f*xx - yy);
    r += t3 * sh(9,0); g += t3 * sh(9,1); b += t3 * sh(9,2);
    t3 = SH_C3[1] * xy * z;
    r += t3 * sh(10,0); g += t3 * sh(10,1); b += t3 * sh(10,2);
    t3 = SH_C3[2] * y * (5.0f*zz - 1.0f);
    r += t3 * sh(11,0); g += t3 * sh(11,1); b += t3 * sh(11,2);
    t3 = SH_C3[3] * z * (5.0f*zz - 3.0f);
    r += t3 * sh(12,0); g += t3 * sh(12,1); b += t3 * sh(12,2);
    t3 = SH_C3[4] * x * (5.0f*zz - 1.0f);
    r += t3 * sh(13,0); g += t3 * sh(13,1); b += t3 * sh(13,2);
    t3 = SH_C3[5] * z * (xx - yy);
    r += t3 * sh(14,0); g += t3 * sh(14,1); b += t3 * sh(14,2);
    t3 = SH_C3[6] * x * (xx - 3.0f*yy);
    r += t3 * sh(15,0); g += t3 * sh(15,1); b += t3 * sh(15,2);

    return glm::max(glm::vec3(r, g, b), 0.0f);
}



// Forward version of 2D covariance matrix computation
__device__ __forceinline__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}



// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void __forceinline__ computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

template<int C>
__global__ void preprocessCUDA_Optimized_Fuse(
    int P, int D, int M,
    const float* __restrict__ orig_points,
    const half* __restrict__ scales,
    const float scale_modifier,
    const glm::vec4* __restrict__ rotations,
    const float* __restrict__ opacities,
    const int8_t* __restrict__ shs,
    bool* __restrict__ clamped,
    const half* __restrict__ cov3D_precomp,
    const float* __restrict__ colors_precomp,
    const float* __restrict__ viewmatrix,
    const float* __restrict__ projmatrix,
    const glm::vec3* __restrict__ cam_pos,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    int* __restrict__ radii,
    float2* __restrict__ points_xy_image,
    float* __restrict__ depths,
    float* __restrict__ cov3Ds,
    half* __restrict__ colors,
    half* __restrict__ conic_opacity,
    const dim3 grid,
    uint32_t* __restrict__ g_global_offset,
    uint32_t* __restrict__ gaussian_keys_unsorted,
    uint32_t* __restrict__ gaussian_values_unsorted,
    const float max_depth,
    bool prefiltered)
{
    auto idx = cg::this_grid().thread_rank();
    const int tid = threadIdx.x;

    __shared__ uint32_t s_base;
    __shared__ uint32_t warp_sums[8]; // 用于 256 线程 (8个 Warp) 的快速 Scan

    uint32_t my_tiles = 0;
    float3 p_orig, p_view;
    float2 point_image;
    float4 con_o;
    
    // 用于 64-bit 向量化写入
    union {
        half h[4];
        float2 f2;
    } vec_conic_opacity;

    bool valid = false;

    // ==================== 1. 计算与剔除阶段 ====================
    if (idx < P)
    {
        p_orig = { 
            __ldg(&orig_points[3 * idx]), 
            __ldg(&orig_points[3 * idx + 1]), 
            __ldg(&orig_points[3 * idx + 2]) 
        };

        if (in_frustum_opt(idx, p_orig, viewmatrix, projmatrix, prefiltered, p_view))
        {
            float opacity = __ldg(&opacities[idx]);
            const half* cov3D_src = cov3D_precomp + idx * 6;
            
            float cov3D_half[6];
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                cov3D_half[i] = __half2float(__ldg(&cov3D_src[i]));
            }

            float4 p_hom = transformPoint4x4(p_orig, projmatrix);
            float p_w = 1.0f / (p_hom.w + 0.0000001f);
            float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

            float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D_half, viewmatrix);
            float det = cov.x * cov.z - cov.y * cov.y;
            
            if (det > 0.0f)
            {
                point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

                float det_inv = __frcp_rn(det); 
                float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
                
                vec_conic_opacity.h[0] = __float2half(conic.x);
                vec_conic_opacity.h[1] = __float2half(conic.y);
                vec_conic_opacity.h[2] = __float2half(conic.z);
                vec_conic_opacity.h[3] = __float2half(opacity);

                con_o = { 
                    __half2float(vec_conic_opacity.h[0]), 
                    __half2float(vec_conic_opacity.h[1]),
                    __half2float(vec_conic_opacity.h[2]), 
                    __half2float(vec_conic_opacity.h[3]) 
                };

                my_tiles = duplicateToTilesTouched(
                    point_image, con_o, grid,
                    0, 0, 0, nullptr, nullptr);

                valid = (my_tiles > 0);
            }
        }
    }

    // ==================== 2. Native Warp-Synchronous Prefix Sum (Bypassing CUB) ====================
    uint32_t lane_id = tid % 32;
    uint32_t warp_id = tid / 32;
    uint32_t val = my_tiles;

    // Warp 内 Inclusive Scan
    #pragma unroll
    for (int i = 1; i <= 16; i *= 2) {
        uint32_t n = __shfl_up_sync(0xffffffff, val, i, 32);
        if (lane_id >= i) val += n;
    }

    // 将每个 Warp 的总和写入 Shared Memory
    if (lane_id == 31) warp_sums[warp_id] = val;
    __syncthreads();

    // 由第一个 Warp 对 warp_sums 进行 Scan
    if (warp_id == 0) {
        uint32_t w_sum = (lane_id < 8) ? warp_sums[lane_id] : 0;
        #pragma unroll
        for (int i = 1; i <= 4; i *= 2) {
            uint32_t n = __shfl_up_sync(0xffffffff, w_sum, i, 32);
            if (lane_id >= i) w_sum += n;
        }
        if (lane_id < 8) warp_sums[lane_id] = w_sum;
    }
    __syncthreads();

    // 计算全局 local offset (Exclusive sum) 和 Block 总和
    uint32_t block_total = warp_sums[7];
    uint32_t local_off = val - my_tiles; // 当前线程在 Warp 内的 exclusive offset
    if (warp_id > 0) {
        local_off += warp_sums[warp_id - 1]; // 加上前面 Warp 的总和
    }

    // ==================== 3. 申请全局内存偏移 ====================
    if (block_total == 0) return;

    if (tid == 0) {
        s_base = atomicAdd(g_global_offset, block_total);
    }
    __syncthreads(); 

    uint32_t global_off = s_base + local_off;

    // ==================== 4. 写出数据阶段 ====================
    if (valid)
    {
        duplicateToTilesTouched(
            point_image, con_o, grid,
            idx, global_off, p_view.z,
            gaussian_keys_unsorted,
            gaussian_values_unsorted,
            max_depth);

        glm::vec3 result = computeColorFromSH_D3_int8(
            idx, D, glm::vec3(p_orig.x, p_orig.y, p_orig.z), *cam_pos, shs, scales);
        
        // 修正点 1: 不使用 half3，直接顺序写入指针，编译器会自动优化
        half* color_ptr = colors + idx * 3;
        color_ptr[0] = __float2half(result.x);
        color_ptr[1] = __float2half(result.y);
        color_ptr[2] = __float2half(result.z);

        depths[idx] = p_view.z;
        points_xy_image[idx] = point_image;

        // 保持向量化写入优化
        float2* dst_vec = reinterpret_cast<float2*>(conic_opacity + idx * 4);
        *dst_vec = vec_conic_opacity.f2;
    }
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color, const float* depths = nullptr,
    float* __restrict__ invdepth = nullptr)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}



#define TILE_THREADS_Y (BLOCK_Y / 2)         // 线程块 Y 方向线程数
#define BLOCK_SIZE_ILP (BLOCK_X * TILE_THREADS_Y) // 每轮加载的高斯数 = 128


#define EXP_SCALE                  12102203.0f
#define EXP_BIAS                   1064866816.0f
#define LOG_ALPHA_THRESHOLD        logf(1.0f / 255.0f)
#define SCALED_LOG_ALPHA_THRESHOLD  ((LOG_ALPHA_THRESHOLD) * (EXP_SCALE) + (EXP_BIAS))

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_SIZE_ILP)  // 总线程数 128
renderCUDA_OPT_V4_ILP2_OPT1(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ points_xy_image,
    const half* __restrict__ features,
    const half* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    const float* depths = nullptr,
    float* __restrict__ invdepth = nullptr)
{
    auto block = cg::this_thread_block();

    // 基础索引：Block 覆盖的像素区域仍为 BLOCK_X × BLOCK_Y
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { blockIdx.x * BLOCK_X, blockIdx.y * BLOCK_Y };

    // 每个线程负责的两个像素
    uint2 pix1 = { pix_min.x + threadIdx.x, pix_min.y + 2 * threadIdx.y };
    uint2 pix2 = { pix_min.x + threadIdx.x, pix_min.y + 2 * threadIdx.y + 1 };

    uint32_t pix1_id = W * pix1.y + pix1.x;
    uint32_t pix2_id = W * pix2.y + pix2.x;

    bool inside1 = pix1.x < W && pix1.y < H;
    bool inside2 = pix2.x < W && pix2.y < H;
    bool done = !inside1 && !inside2;  // 两个像素都不在图像内才彻底跳过

    // 线程与 Warp 索引（总 128 线程 = 4 个 Warp）
    uint32_t tid = threadIdx.y * BLOCK_X + threadIdx.x;  // 0..127

    uint2 range = ranges[blockIdx.y * horizontal_blocks + blockIdx.x];
    const int rounds = (range.y - range.x + BLOCK_SIZE_ILP - 1) / BLOCK_SIZE_ILP;
    int toDo = range.y - range.x;

    // 共享内存：大小为 BLOCK_SIZE（128）
    __shared__ float3  collected_feat[BLOCK_SIZE_ILP];
    __shared__ float4  collected_quad_const[BLOCK_SIZE_ILP];
    __shared__ float2  collected_linear[BLOCK_SIZE_ILP];

    // 双像素混合状态
    float T1 = 1.0f, T2 = 1.0f;
    float Color1[CHANNELS] = { 0.0f };
    float Color2[CHANNELS] = { 0.0f };

    // 预先计算两个像素的局部坐标（相对于 tile 左上角）
    float local_x = (float)threadIdx.x;          // 两个像素相同
    float local_y1 = (float)(2 * threadIdx.y);   // 第一个像素的 Y 偏移

    float x_2 = local_x * local_x;
    float y1_2 = local_y1 * local_y1;
    float xy1 = local_x * local_y1;
    float y_diff_const = 2.0f * local_y1 + 1.0f;

    // 遍历高斯批次
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE_ILP)
    {
        // 块内统一提前退出：两个像素的 T 都极小或都出界
        if (__syncthreads_and((T1 < 0.01f && T2 < 0.01f) || done))
            break;

        // --- 每个线程加载一个高斯到共享内存 ---
        int progress = i * BLOCK_SIZE_ILP + tid;
        if (progress < range.y - range.x)
        {
            int coll_id = point_list[range.x + progress];
            float2 g_xy = points_xy_image[coll_id];
            const half* hptr = conic_opacity + coll_id * 4;
            float4 con_o = make_float4(
                __half2float(hptr[0]),
                __half2float(hptr[1]),
                __half2float(hptr[2]),
                __half2float(hptr[3])
            );

            // float3 feat = reinterpret_cast<const float3*>(features)[coll_id];
            const half* h_feat = features + coll_id * CHANNELS;
            float3 feat = make_float3(
                __half2float(h_feat[0]),
                __half2float(h_feat[1]),
                __half2float(h_feat[2])
            );

            float cx = g_xy.x;
            float cy = g_xy.y;
            float A = -0.5f * con_o.x;
            float B = con_o.y;
            float C = -0.5f * con_o.z;
            float D = __logf(con_o.w);

            // 预计算与像素无关的二次型系数（相对于 tile 左上角）
            float local_cx = cx - (float)pix_min.x;
            float local_cy = cy - (float)pix_min.y;
            float const_term = A * local_cx * local_cx - B * local_cx * local_cy + C * local_cy * local_cy + D;
            float lin_x = -2.0f * A * local_cx + B * local_cy;
            float lin_y = -2.0f * C * local_cy + B * local_cx;

            // ---- 优化点 2：将 fast_exp_bit 的乘加操作融合进二次型系数 ----
            float hoisted_A     = A * EXP_SCALE;
            float hoisted_B     = B * EXP_SCALE;
            float hoisted_C     = C * EXP_SCALE;
            float hoisted_lin_x = lin_x * EXP_SCALE;
            float hoisted_lin_y = lin_y * EXP_SCALE;
            // 常数项需要应用完整的线性变换：Scale + Bias
            float hoisted_const = const_term * EXP_SCALE + EXP_BIAS;

            collected_quad_const[tid] = { hoisted_A, -hoisted_B, hoisted_C, hoisted_const };
            collected_linear[tid]     = { hoisted_lin_x, hoisted_lin_y };
            collected_feat[tid]       = make_float3(feat.x, feat.y, feat.z);
        }
        
        int batch_size = min(BLOCK_SIZE_ILP, toDo);
        bool all_saturated1 = (T1 < 0.01f && T2 < 0.01f);
        block.sync();
        if (__all_sync(0xFFFFFFFF, all_saturated1 || done)) continue;

        #pragma unroll 8
        for (int j = 0; j < batch_size; j++)
        {
            float4 quad_const = collected_quad_const[j];
            float2 linear     = collected_linear[j];
            float3 f          = collected_feat[j];
    
            float A = quad_const.x;
            float B = quad_const.y;
            float C = quad_const.z;
            float const_term = quad_const.w;

            // ---- 优化点 3：直接计算出应用了线性变换的 Scaled Power ----
            float scaled_power1 = A * x_2 + C * y1_2 + B * xy1 + linear.x * local_x + linear.y * local_y1 + const_term;
            float scaled_power2 = scaled_power1 + C * y_diff_const + B * local_x + linear.y;

            // 比较判定使用对应缩放后的阈值
            bool is_cuiling1 = scaled_power1 > SCALED_LOG_ALPHA_THRESHOLD;
            bool is_cuiling2 = scaled_power2 > SCALED_LOG_ALPHA_THRESHOLD;
            
            // ---- 优化点 4：跳过多余函数调用，直接进行位解释 ----
            float alpha1 = is_cuiling1 ? __int_as_float(__float2int_rn(scaled_power1)) : 0.0f;
            float alpha2 = is_cuiling2 ? __int_as_float(__float2int_rn(scaled_power2)) : 0.0f;


            float weighted_alpha1 = alpha1 * T1;
            float weighted_alpha2 = alpha2 * T2;
            
            Color1[0] += f.x * weighted_alpha1;
            Color1[1] += f.y * weighted_alpha1;
            Color1[2] += f.z * weighted_alpha1;

            Color2[0] += f.x * weighted_alpha2;
            Color2[1] += f.y * weighted_alpha2;
            Color2[2] += f.z * weighted_alpha2;
            
            T1 -= weighted_alpha1;
            T2 -= weighted_alpha2;
        }
    }

    // --- 写回两个像素 ---
    if (inside1)
    {
        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[ch * H * W + pix1_id] = Color1[ch] + T1 * bg_color[ch];
    }
    if (inside2)
    {
        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[ch * H * W + pix2_id] = Color2[ch] + T2 * bg_color[ch];
    }
}



template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_SIZE_ILP)
renderCUDA_OPT_V4_ILP4_OPT2(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ points_xy_image,
    const half* __restrict__ features,
    const half* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    const float* depths = nullptr,
    float* __restrict__ invdepth = nullptr)
{
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { blockIdx.x * BLOCK_X, blockIdx.y * BLOCK_Y };
    const uint32_t tid = threadIdx.y * BLOCK_X + threadIdx.x;

    // 展开4个像素坐标
    const uint2 pix0 = { pix_min.x + threadIdx.x, pix_min.y + 4 * threadIdx.y };
    const uint2 pix1 = { pix_min.x + threadIdx.x, pix_min.y + 4 * threadIdx.y + 1 };
    const uint2 pix2 = { pix_min.x + threadIdx.x, pix_min.y + 4 * threadIdx.y + 2 };
    const uint2 pix3 = { pix_min.x + threadIdx.x, pix_min.y + 4 * threadIdx.y + 3 };

    const uint32_t pix0_id = W * pix0.y + pix0.x;
    const uint32_t pix1_id = W * pix1.y + pix1.x;
    const uint32_t pix2_id = W * pix2.y + pix2.x;
    const uint32_t pix3_id = W * pix3.y + pix3.x;

    const bool inside0 = (pix0.x < W) && (pix0.y < H);
    const bool inside1 = (pix1.x < W) && (pix1.y < H);
    const bool inside2 = (pix2.x < W) && (pix2.y < H);
    const bool inside3 = (pix3.x < W) && (pix3.y < H);
    const bool done = !inside0 && !inside1 && !inside2 && !inside3;

    uint2 range = ranges[blockIdx.y * horizontal_blocks + blockIdx.x];
    const int rounds = (range.y - range.x + BLOCK_SIZE_ILP - 1) / BLOCK_SIZE_ILP;
    int toDo = range.y - range.x;

    __shared__ float3  collected_feat[BLOCK_SIZE_ILP];
    __shared__ float4  collected_quad_const[BLOCK_SIZE_ILP];
    __shared__ float2  collected_linear[BLOCK_SIZE_ILP];

    float T0 = 1.0f, T1 = 1.0f, T2 = 1.0f, T3 = 1.0f;
    float C0[CHANNELS] = {0.0f}, C1[CHANNELS] = {0.0f}, C2[CHANNELS] = {0.0f}, C3[CHANNELS] = {0.0f};

    const float local_x = (float)threadIdx.x;
    const float x_2 = local_x * local_x;
    const float local_y0 = (float)(4 * threadIdx.y);
    const float y0_2 = local_y0 * local_y0;
    const float xy0 = local_x * local_y0;

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE_ILP) {
        // 循环开头同步检查饱和/出界（原始位置）
        bool all_saturated_start = (T0 < 0.01f && T1 < 0.01f && T2 < 0.01f && T3 < 0.01f) || done;
        if (__syncthreads_and(all_saturated_start))
            break;

        int progress = i * BLOCK_SIZE_ILP + tid;
        if (progress < range.y - range.x) {
            int coll_id = point_list[range.x + progress];
            float2 g_xy = __ldg(points_xy_image + coll_id);

            const half* hptr = conic_opacity + coll_id * 4;
            uint64_t conic_raw = __ldg(reinterpret_cast<const uint64_t*>(hptr));
            const half* conic_arr = reinterpret_cast<const half*>(&conic_raw);
            float4 con_o = make_float4(
                __half2float(conic_arr[0]), __half2float(conic_arr[1]),
                __half2float(conic_arr[2]), __half2float(conic_arr[3]));

            const half* h_feat = features + coll_id * CHANNELS;
            uint32_t feat_raw = __ldg(reinterpret_cast<const uint32_t*>(h_feat));
            const half* feat_half = reinterpret_cast<const half*>(&feat_raw);
            float feat_x = __half2float(feat_half[0]);
            float feat_y = __half2float(feat_half[1]);
            float feat_z = __half2float(*(h_feat + 2));

            float cx = g_xy.x, cy = g_xy.y;
            float A = -0.5f * con_o.x;
            float B = con_o.y;
            float C = -0.5f * con_o.z;
            float D = __logf(con_o.w);

            float local_cx = cx - (float)pix_min.x;
            float local_cy = cy - (float)pix_min.y;
            float const_term = A*local_cx*local_cx - B*local_cx*local_cy + C*local_cy*local_cy + D;
            float lin_x = -2.0f*A*local_cx + B*local_cy;
            float lin_y = -2.0f*C*local_cy + B*local_cx;

            float hoisted_A = A * EXP_SCALE;
            float hoisted_B = B * EXP_SCALE;
            float hoisted_C = C * EXP_SCALE;
            float hoisted_lin_x = lin_x * EXP_SCALE;
            float hoisted_lin_y = lin_y * EXP_SCALE;
            float hoisted_const = const_term * EXP_SCALE + EXP_BIAS;

            collected_quad_const[tid] = { hoisted_A, -hoisted_B, hoisted_C, hoisted_const };
            collected_linear[tid]     = { hoisted_lin_x, hoisted_lin_y };
            collected_feat[tid]       = { feat_x, feat_y, feat_z };
        }

        int batch_size = min(BLOCK_SIZE_ILP, toDo);
        bool all_saturated_now = (T0 < 0.01f && T1 < 0.01f && T2 < 0.01f && T3 < 0.01f);
        block.sync();
        if (__all_sync(0xFFFFFFFF, all_saturated_now || done)) continue;

        #pragma unroll 4
        for (int j = 0; j < batch_size; j++) {
            float4 quad_const = collected_quad_const[j];
            float2 linear     = collected_linear[j];
            float3 f          = collected_feat[j];

            float A_coef = quad_const.x, B_coef = quad_const.y;
            float C_coef = quad_const.z, const_term = quad_const.w;

            float sp0 = A_coef*x_2 + C_coef*y0_2 + B_coef*xy0 + linear.x*local_x + linear.y*local_y0 + const_term;
            float delta = C_coef * (2.0f*local_y0 + 1.0f) + B_coef*local_x + linear.y;

            // 像素0
            bool cull0 = sp0 > SCALED_LOG_ALPHA_THRESHOLD;
            float alpha0 = cull0 ? __int_as_float(__float2int_rn(sp0)) : 0.0f;
            float wa0 = alpha0 * T0;
            C0[0] += f.x * wa0; C0[1] += f.y * wa0; C0[2] += f.z * wa0;
            T0 -= wa0;

            // 像素1
            float sp1 = sp0 + delta;
            delta += 2.0f * C_coef;
            bool cull1 = sp1 > SCALED_LOG_ALPHA_THRESHOLD;
            float alpha1 = cull1 ? __int_as_float(__float2int_rn(sp1)) : 0.0f;
            float wa1 = alpha1 * T1;
            C1[0] += f.x * wa1; C1[1] += f.y * wa1; C1[2] += f.z * wa1;
            T1 -= wa1;

            // 像素2
            float sp2 = sp1 + delta;
            delta += 2.0f * C_coef;
            bool cull2 = sp2 > SCALED_LOG_ALPHA_THRESHOLD;
            float alpha2 = cull2 ? __int_as_float(__float2int_rn(sp2)) : 0.0f;
            float wa2 = alpha2 * T2;
            C2[0] += f.x * wa2; C2[1] += f.y * wa2; C2[2] += f.z * wa2;
            T2 -= wa2;

            // 像素3
            float sp3 = sp2 + delta;
            bool cull3 = sp3 > SCALED_LOG_ALPHA_THRESHOLD;
            float alpha3 = cull3 ? __int_as_float(__float2int_rn(sp3)) : 0.0f;
            float wa3 = alpha3 * T3;
            C3[0] += f.x * wa3; C3[1] += f.y * wa3; C3[2] += f.z * wa3;
            T3 -= wa3;
        }
    }

    // 写回
    if (inside0) { for (int ch=0; ch<CHANNELS; ch++) out_color[ch*H*W + pix0_id] = C0[ch] + T0 * bg_color[ch]; }
    if (inside1) { for (int ch=0; ch<CHANNELS; ch++) out_color[ch*H*W + pix1_id] = C1[ch] + T1 * bg_color[ch]; }
    if (inside2) { for (int ch=0; ch<CHANNELS; ch++) out_color[ch*H*W + pix2_id] = C2[ch] + T2 * bg_color[ch]; }
    if (inside3) { for (int ch=0; ch<CHANNELS; ch++) out_color[ch*H*W + pix3_id] = C3[ch] + T3 * bg_color[ch]; }
}





void FORWARD::render_only(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const half* colors,
	const half* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	dim3 grid_dim((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1); // 恢复为 16
	dim3 block_dim(BLOCK_X, BLOCK_Y/2, 1);                               // 保持 16x8

	// printf("W is %d\n", W);
	// printf("H is %d\n", H);

	renderCUDA_OPT_V4_ILP2_OPT1<NUM_CHANNELS> << <grid_dim, block_dim>> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);

}



void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	dim3 grid_dim((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1); // 恢复为 16
	dim3 block_dim(BLOCK_X, BLOCK_Y/2, 1);                               // 保持 16x8

	// printf("W is %d\n", W);
	// printf("H is %d\n", H);

	renderCUDA<NUM_CHANNELS> << <grid, block>> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);

}

void FORWARD::preprocess_fuse(int P, int D, int M,
    const float* means3D,
    const half* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const int8_t* shs,
    bool* clamped,
    const half* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    int* radii,
    float2* means2D,
    float* depths,
    float* cov3Ds,
    half* rgb,
    half* conic_opacity,
    const dim3 grid,
    uint32_t* g_global_offset,
    uint32_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    const float max_depth,
    bool prefiltered)
{
    preprocessCUDA_Optimized_Fuse<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
        P, D, M,
        means3D,
        scales,
        scale_modifier,
        rotations,
        opacities,
        shs,
        clamped,
        cov3D_precomp,
        colors_precomp,
        viewmatrix,
        projmatrix,
        cam_pos,
        W, H,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        radii,
        means2D,
        depths,
        cov3Ds,
        rgb,
        conic_opacity,
        grid,
        g_global_offset,
        gaussian_keys_unsorted,
        gaussian_values_unsorted,
        max_depth,
        prefiltered
    );
}
