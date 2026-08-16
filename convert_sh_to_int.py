import numpy as np
from plyfile import PlyData, PlyElement
import json

# ================= 配置 =================
input_path = "../gaussian-splatting/output/tandt/train/point_cloud/iteration_30000/point_cloud.ply"
output_ply = input_path.replace(".ply", "_int8sh_fixed_scale.ply")
output_json = input_path.replace(".ply", "_int8sh_fixed_scale_meta.json")

FIXED_MAX_ABS = 10.0          # 假设所有通道的 max_abs = 10
FIXED_SCALE = FIXED_MAX_ABS / 127.0   # = 10/127 ≈ 0.07874

# ================= 读取原始 PLY =================
plydata = PlyData.read(input_path)
vertex = plydata['vertex']

# 找出所有球谐属性名（保持原始顺序）
sh_names = [p.name for p in vertex.properties
            if p.name.startswith('f_dc_') or p.name.startswith('f_rest_')]
print(f"共 {len(sh_names)} 个球谐属性（0阶 + 高阶）")

# 提取全部 SH 数据，形状: (属性数, 顶点数)
sh_data = np.array([vertex[name] for name in sh_names], dtype=np.float32)
N = sh_data.shape[1]
print(f"顶点数: {N}")

# ================= 固定 scale 量化 =================
# 量化公式: q = round( x / scale )  clamp to [-127, 127]
quantized = np.clip(np.round(sh_data / FIXED_SCALE), -127, 127).astype(np.int8)

# 验证反量化误差
restored = quantized.astype(np.float32) * FIXED_SCALE
max_err = np.max(np.abs(sh_data - restored))
print(f"固定 scale={FIXED_SCALE:.6f}  max_abs=10.0  最大反量化误差: {max_err:.6f}")

# ================= 构建新顶点数组 =================
new_dtype = []
for prop in vertex.properties:
    name = prop.name
    if name in sh_names:
        new_dtype.append((name, 'int8'))
    else:
        new_dtype.append((name, vertex[name].dtype))

new_vertex = np.empty(N, dtype=new_dtype)
for i, (name, _) in enumerate(new_dtype):
    if name in sh_names:
        idx = sh_names.index(name)
        new_vertex[name] = quantized[idx]
    else:
        new_vertex[name] = vertex[name].copy()

# ================= 保存 PLY（comment 中记录 scale） =================
comments = [
    f"int8_sh_fixed_scale: {FIXED_SCALE:.10f}",
    "Decode: float_value = int8_value * scale",
    "max_abs_assumed = 10.0"
]
el = PlyElement.describe(new_vertex, 'vertex')
PlyData([el], text=False, comments=comments).write(output_ply)
print(f"已保存 int8 SH PLY: {output_ply}")

# ================= 保存 JSON 元数据（方便 CUDA 加载） =================
meta = {
    "scale": float(FIXED_SCALE),
    "assumed_max_abs": 10.0,
    "note": "For all 3 channels, same fixed scale = 10/127"
}
with open(output_json, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"已保存 meta: {output_json}")