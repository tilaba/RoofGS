#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#include "forward.h"

using namespace std;

/* ================= 工具函数 ================= */

template<typename T>
bool load_bin(const string& filename, vector<T>& out)
{
    ifstream fin(filename, ios::binary | ios::ate);
    if (!fin.is_open()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    size_t bytes = fin.tellg();
    if (bytes % sizeof(T) != 0) {
        cerr << "Size mismatch for " << filename << endl;
        return false;
    }

    size_t count = bytes / sizeof(T);
    fin.seekg(0, ios::beg);

    out.resize(count);
    fin.read(reinterpret_cast<char*>(out.data()), bytes);
    fin.close();

    cout << "Loaded " << filename << " (" << count << " elements)" << endl;
    return true;
}

template<typename T>
T* copy_to_gpu(const vector<T>& h)
{
    if (h.empty()) return nullptr;
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template<typename T>
static void save_bin(const std::string& filename, const T* data, size_t count) {
    if (count == 0) return;
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }
    ofs.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    ofs.close();
    std::cout << "Saved " << filename << " (" << count << " elements)" << std::endl;
}

template<typename T>
static void save_gpu_bin(const std::string& filename, const T* d_ptr, size_t count) {
    if (count == 0 || d_ptr == nullptr) return;
    std::vector<T> h_data(count);
    cudaMemcpy(h_data.data(), d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    save_bin(filename, h_data.data(), count);
}


/* ================= main ================= */

int main()
{
    /* ---------- 1. 加载输入 ---------- */

    vector<uint2>    ranges;
    vector<uint32_t> point_list;
    vector<float2>   means2D;
    vector<float>    colors;
    vector<float4>   conic_opacity;
    vector<float>    final_T;
    vector<uint32_t> n_contrib;
    vector<float>    background(3, 0.0f);
    vector<float>    depths;
    vector<float>    out_depths;

    load_bin("/home/yluo/Turbo3DGS/data/render_ranges.bin", ranges);
    load_bin("/home/yluo/Turbo3DGS/data/render_point_list.bin", point_list);
    load_bin("/home/yluo/Turbo3DGS/data/render_means2D.bin", means2D);
    load_bin("/home/yluo/Turbo3DGS/data/render_colors.bin", colors);
    load_bin("/home/yluo/Turbo3DGS/data/render_conic_opacity.bin", conic_opacity);
    load_bin("/home/yluo/Turbo3DGS/data/render_accum_alpha.bin", final_T);
    load_bin("/home/yluo/Turbo3DGS/data/render_n_contrib.bin", n_contrib);
    load_bin("/home/yluo/Turbo3DGS/data/render_depths.bin", depths);
    load_bin("/home/yluo/Turbo3DGS/data/d_out_depth.bin", out_depths);

    /* ---------- 2. 加载参考输出 ---------- */

    vector<float> ref_out_color;
    load_bin("/home/yluo/Turbo3DGS/data/render_out_coler.bin", ref_out_color);

    /* ---------- 3. 拷贝到 GPU ---------- */

    uint2*    d_ranges        = copy_to_gpu(ranges);
    uint32_t* d_point_list    = copy_to_gpu(point_list);
    float2*   d_means2D       = copy_to_gpu(means2D);
    float*    d_colors        = copy_to_gpu(colors);
    float4*   d_conic_opacity = copy_to_gpu(conic_opacity);
    float*    d_final_T       = copy_to_gpu(final_T);
    uint32_t* d_n_contrib     = copy_to_gpu(n_contrib);
    float*    d_bg            = copy_to_gpu(background);
    float*    d_depths        = copy_to_gpu(depths);

    /* ---------- 4. 输出 buffer ---------- */

    int W = 1332;
    int H = 876;

    size_t out_color_elems = W * H * 3;

    float* d_out_color = nullptr;
    float* d_out_depth = nullptr;

    cudaMalloc(&d_out_color, out_color_elems * sizeof(float));
    cudaMalloc(&d_out_depth, W * H * sizeof(float));

    /* ---------- 5. grid / block ---------- */

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    /* ---------- 6. warm-up ---------- */

    for (int i = 0; i < 5; ++i) {
        FORWARD::render(
            grid, block,
            d_ranges,
            d_point_list,
            W, H,
            d_means2D,
            d_colors,
            d_conic_opacity,
            d_final_T,
            d_n_contrib,
            d_bg,
            d_out_color);
    }
    cudaDeviceSynchronize();

    /* ---------- 7. CUDA Event 计时 ---------- */

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITER = 10;
    float total_ms = 0.f;

    for (int i = 0; i < ITER; ++i) {
        cudaEventRecord(start);

        FORWARD::render(
            grid, block,
            d_ranges,
            d_point_list,
            W, H,
            d_means2D,
            d_colors,
            d_conic_opacity,
            d_final_T,
            d_n_contrib,
            d_bg,
            d_out_color);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;

        cout << "Iter " << i << ": " << ms << " ms" << endl;
    }

    cout << "Average: " << total_ms / ITER << " ms" << endl;

    /* ---------- 8. 拷回结果 ---------- */

    vector<float> out_color(out_color_elems);
    cudaMemcpy(out_color.data(), d_out_color,
               out_color_elems * sizeof(float),
               cudaMemcpyDeviceToHost);

    /* ---------- 8.5 深度结果拷回 ---------- */

    vector<float> out_depth(H * W);
    cudaMemcpy(out_depth.data(), d_out_depth,
            H * W * sizeof(float),
            cudaMemcpyDeviceToHost);

    /* ---------- 9. 对比结果 ---------- */

    double max_err = 0.0;
    double mean_err = 0.0;

    for (size_t i = 0; i < out_color_elems; ++i) {
        double diff = fabs(out_color[i] - ref_out_color[i]);
        max_err = max(max_err, diff);
        mean_err += diff;
    }
    mean_err /= out_color_elems;

    cout << "\n===== out_color compare =====" << endl;
    cout << "Max error : " << max_err << endl;
    cout << "Mean error: " << mean_err << endl;



    /* ---------- 9. depth 对比 ---------- */
    double max_err_depth = 0.0;
    double mean_err_depth = 0.0;
    double valid_count = 0;

    for (size_t i = 0; i < H * W; ++i) 
    {
        float ref = out_depths[i];
        float val = out_depth[i];

        // 可选：处理 invalid depth（避免 inf / 0 干扰）
        if (ref == 0.0f && val == 0.0f)
            continue;

        double diff = fabs((double)val - (double)ref);

        max_err_depth = max(max_err_depth, diff);
        mean_err_depth += diff;
        valid_count += 1.0;
    }

    if (valid_count > 0)
        mean_err_depth /= valid_count;

    cout << "\n===== out_depth compare =====" << endl;
    cout << "Max error : " << max_err_depth << endl;
    cout << "Mean error: " << mean_err_depth << endl;


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_ranges);
    cudaFree(d_point_list);
    cudaFree(d_means2D);
    cudaFree(d_colors);
    cudaFree(d_conic_opacity);
    cudaFree(d_final_T);
    cudaFree(d_n_contrib);
    cudaFree(d_bg);
    cudaFree(d_depths);
    cudaFree(d_out_color);
    cudaFree(d_out_depth);

    return 0;
}
