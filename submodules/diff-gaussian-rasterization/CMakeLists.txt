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

cmake_minimum_required(VERSION 3.10)

project(DiffRast LANGUAGES CUDA CXX)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -O2 -arch=sm_75")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(CudaRasterizer
	cuda_rasterizer/backward.h
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.h
	cuda_rasterizer/forward.cu
	cuda_rasterizer/auxiliary.h
	cuda_rasterizer/rasterizer_impl.cu
	cuda_rasterizer/rasterizer_impl.h
	cuda_rasterizer/rasterizer.h
)

set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_include_directories(CudaRasterizer PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_include_directories(CudaRasterizer PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_executable(test_rasterizer main.cpp)
target_link_libraries(test_rasterizer PRIVATE CudaRasterizer)
set_target_properties(test_rasterizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
