#!/usr/bin/env python3
"""列出 PLY 文件顶点属性名称、类型，并可预览前几行数值。"""
import numpy as np
from plyfile import PlyData

def inspect_ply(filename, num_preview=3):
    ply = PlyData.read(filename)
    vert = ply['vertex']
    
    print(f"文件: {filename}")
    print(f"顶点总数: {vert.count}")
    print("\n顶点属性列表:")
    for name in vert.data.dtype.names:
        dtype_str = str(vert.data.dtype[name])
        print(f"  {name}: {dtype_str}")

    if num_preview > 0:
        print(f"\n前 {min(num_preview, vert.count)} 个顶点的数据:")
        for name in vert.data.dtype.names:
            vals = vert[name][:num_preview]
            if np.issubdtype(vert.data.dtype[name], np.unsignedinteger):
                # 以十六进制打印无符号整数，便于查看 half 的位模式
                print(f"  {name}: {vals}  (hex: {[hex(v) for v in vals]})")
            else:
                print(f"  {name}: {vals}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python inspect_ply.py <点云ply文件>")
    else:
        inspect_ply(sys.argv[1])