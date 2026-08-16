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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


// ================= CUDA TIMER =================
#define CUDA_TIMER_INIT                         \
    cudaEvent_t start, stop;                  \
    cudaEventCreate(&start);                  \
    cudaEventCreate(&stop);                   \
    float ms = 0.0f;

#define CUDA_TIMER_START cudaEventRecord(start);

#define CUDA_TIMER_STOP(name)                                      \
    cudaEventRecord(stop);                                        \
    cudaEventSynchronize(stop);                                   \
    cudaEventElapsedTime(&ms, start, stop);                       \
    printf("[CUDA TIMER] %-25s : %8.4f ms\n", name, ms);

#define CUDA_TIMER_DESTROY        \
    cudaEventDestroy(start);     \
    cudaEventDestroy(stop);

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint32_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	half* con_o,
  uint32_t* tiles_touched,
	dim3 grid, const float max_depth)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (tiles_touched[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    // Update unsorted arrays with Gaussian idx for every tile that
    // Gaussian touches
    
    const half* hptr = con_o + idx * 4;
    int2 loaded_val = *reinterpret_cast<const int2*>(hptr);
    const half2* h2_ptr = reinterpret_cast<const half2*>(&loaded_val);
    float2 f01 = __half22float2(h2_ptr[0]);
    float2 f23 = __half22float2(h2_ptr[1]);
    float4 con_o_f = make_float4(f01.x, f01.y, f23.x, f23.y);

    duplicateToTilesTouched(
        points_xy[idx], con_o_f, grid,
        idx, off, depths[idx],
        gaussian_keys_unsorted,
        gaussian_values_unsorted, max_depth);
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint32_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	// uint64_t key = point_list_keys[idx];
    uint32_t key = point_list_keys[idx];
	uint32_t currtile = key >> QUANT_DEPTH_BITS;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> QUANT_DEPTH_BITS;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	// obtain(chunk, geom.clamped, P * 3, 128);
	// obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, 4 * P, 128);
    obtain(chunk, geom.conic_opacity_bwd, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
    obtain(chunk, geom.rgb_bwd, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
    std::function<char* (size_t)> geometryBuffer,
    std::function<char* (size_t)> binningBuffer,
    std::function<char* (size_t)> imageBuffer,
    const int P, int D, int M,
    const float* background,
    const int width, int height,
    const float* means3D,
    const int8_t* shs,
    const float* colors_precomp,
    const float* opacities,
    const half* scales,
    const float max_depth,
    const float scale_modifier,
    const float* rotations,
    const half* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    const float tan_fovx, float tan_fovy,
    const bool prefiltered,
    float* kernel_times,
    float* out_color,
    int* radii,
    bool debug)
{
    // CUDA_TIMER_INIT
    
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventRecord(total_start);

    //CUDA_TIMER_START
    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    size_t chunk_size = required<GeometryState>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    if (radii == nullptr)
        radii = geomState.internal_radii;

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X,
                   (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    size_t img_chunk_size = required<ImageState>(width * height);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

    //CUDA_TIMER_STOP("imageBuffer + geometryBuffer")

    if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
        throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");

    // =========================
    // preprocess
    // =========================
//     //CUDA_TIMER_START
//     CHECK_CUDA(FORWARD::preprocess(
//         P, D, M,
//         means3D,
//         scales,
//         scale_modifier,
//         (glm::vec4*)rotations,
//         opacities,
//         shs,
//         geomState.clamped,
//         cov3D_precomp,
//         colors_precomp,
//         viewmatrix, projmatrix,
//         (glm::vec3*)cam_pos,
//         width, height,
//         focal_x, focal_y,
//         tan_fovx, tan_fovy,
//         radii,
//         geomState.means2D,
//         geomState.depths,
//         geomState.cov3D,
//         geomState.rgb,
//         geomState.conic_opacity,
//         tile_grid,
//         geomState.tiles_touched,
//         prefiltered
//     ), debug)
//    //CUDA_TIMER_STOP("preprocess")

//     // =========================
//     // prefix scan
//     // =========================
//     //CUDA_TIMER_START
//     CHECK_CUDA(cub::DeviceScan::InclusiveSum(
//         geomState.scanning_space,
//         geomState.scan_size,
//         geomState.tiles_touched,
//         geomState.point_offsets,
//         P), debug)
//    //CUDA_TIMER_STOP("prefix scan")

//     //CUDA_TIMER_START

//     int num_rendered;
//     CHECK_CUDA(cudaMemcpy(&num_rendered,
//         geomState.point_offsets + P - 1,
//         sizeof(int),
//         cudaMemcpyDeviceToHost), debug);

//     size_t binning_chunk_size = required<BinningState>(num_rendered);
//     char* binning_chunkptr = binningBuffer(binning_chunk_size);
//     BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);
//    //CUDA_TIMER_STOP("binningBuffer allocation");


//     // =========================
//     // duplicate
//     // =========================
//     // CUDA_TIMER_START
//     duplicateWithKeys<<<(P + 255) / 256, 256>>>(
//         P,
//         geomState.means2D,
//         geomState.depths,
//         geomState.point_offsets,
//         binningState.point_list_keys_unsorted,
//         binningState.point_list_unsorted,
//         geomState.conic_opacity,
//         geomState.tiles_touched,
//         tile_grid, max_depth);
//     CHECK_CUDA(cudaGetLastError(), debug)
//    CUDA_TIMER_STOP("duplicate")

    // 全局原子计数器，初始为 0
uint32_t* g_global_offset;
cudaMalloc(&g_global_offset, sizeof(uint32_t));
cudaMemset(g_global_offset, 0, sizeof(uint32_t));

// 预估最大可能 tile 数，分配键值缓冲区（避免溢出）
const int MAX_TILES_PER_GAUSSIAN = 32;   // 可根据场景调整，如 64/128/256
int max_tiles_guess = max(P * MAX_TILES_PER_GAUSSIAN, 320000 * MAX_TILES_PER_GAUSSIAN);
// printf("P is %d\n", P);



size_t binning_chunk_size = required<BinningState>(max_tiles_guess);
char* binning_chunkptr = binningBuffer(binning_chunk_size);
BinningState binningState = BinningState::fromChunk(binning_chunkptr, max_tiles_guess);
CHECK_CUDA(FORWARD::preprocess_fuse(
    P, D, M,
    means3D,
    scales,
    scale_modifier,
    (glm::vec4*)rotations,
    opacities,
    shs,
    geomState.clamped,
    cov3D_precomp,
    colors_precomp,
    viewmatrix, projmatrix,
    (glm::vec3*)cam_pos,
    width, height,
    focal_x, focal_y,
    tan_fovx, tan_fovy,
    radii,
    geomState.means2D,
    geomState.depths,
    geomState.cov3D,
    geomState.rgb,
    geomState.conic_opacity,
    tile_grid,
    g_global_offset,
    binningState.point_list_keys_unsorted,
    binningState.point_list_unsorted,
    max_depth,
    prefiltered
), debug);

int num_rendered;
cudaMemcpy(&num_rendered, g_global_offset, sizeof(int), cudaMemcpyDeviceToHost);
cudaFree(g_global_offset);




    int bit = getHigherMsb(tile_grid.x * tile_grid.y);

    // =========================
    // radix sort
    // =========================
    //CUDA_TIMER_START
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space,
        binningState.sorting_size,
        binningState.point_list_keys_unsorted,
        binningState.point_list_keys,
        binningState.point_list_unsorted,
        binningState.point_list,
        num_rendered, 0, QUANT_DEPTH_BITS  + bit), debug)
    // printf("bit is %d\n", bit);
   //CUDA_TIMER_STOP("radix sort")
    
   //CUDA_TIMER_START
    CHECK_CUDA(cudaMemset(imgState.ranges, 0,
        tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
    //CUDA_TIMER_STOP("cudaMemset")

    // =========================
    // tile range
    // =========================
    //CUDA_TIMER_START
    // printf("total gaussians is%d\n", P);
    // printf("contributing gaussians is%d\n", num_rendered);
    if (num_rendered > 0)
    {
        identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binningState.point_list_keys,
            imgState.ranges);
        CHECK_CUDA(cudaGetLastError(), debug)
    }
   //CUDA_TIMER_STOP("tile range")

    // =========================
    // render
    // =========================
    const half* feature_ptr = geomState.rgb;
    
    //CUDA_TIMER_START
    // cudaEventRecord(total_stop);
    // cudaEventSynchronize(total_stop);
    // float ms1;
    // cudaEventElapsedTime(&ms1, total_start, total_stop);


    CHECK_CUDA(FORWARD::render_only(
        tile_grid, block,
        imgState.ranges,
        binningState.point_list,
        width, height,
        geomState.means2D,
        feature_ptr,
        geomState.conic_opacity,
        imgState.accum_alpha,
        imgState.n_contrib,
        background,
        out_color), debug)
   //CUDA_TIMER_STOP("render")

    // =========================
    // total
    // =========================
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    float total_ms;
    cudaEventElapsedTime(&total_ms, total_start, total_stop);



    //  printf("[CUDA TIMER] %-25s : %8.4f ms\n", "render", total_ms - ms1);

    // printf("\n================ CUDA PROFILE ================\n");
    // printf("[CUDA TOTAL] forward total: %.4f ms\n", total_ms);
    // printf("=============================================\n\n");

    // // CUDA_TIMER_DESTROY
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    kernel_times[0] = total_ms;

    return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dG2,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb_bwd;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity_bwd,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
    dL_dG2), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}
