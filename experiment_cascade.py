"""
级联量化综合实验：比较多条量化路径的 MSE 失真。

"16" 代表原始浮点数（视为无损 16-bit 源），
后续数字代表每个坐标的量化 bit 数（b=1,2,4,8）。
"""

import numpy as np
import torch
from turboquant import TurboQuantMSE, normalize

# ── 实验参数 ──────────────────────────────────────────────
d    = 512
n    = 3000
seed = 42
BITS = [1, 2, 4, 8]          # 支持的量化 bit 宽
# ─────────────────────────────────────────────────────────

torch.manual_seed(seed)
x = torch.randn(n, d)
x, _ = normalize(x)

# 预建所有量化器（避免重复构造）
print("初始化量化器...")
quantizers = {}
for b in BITS:
    quantizers[b] = TurboQuantMSE(d=d, b=b, seed=seed + b)
    print(f"  b={b} 完成")

def cascade(x_in, bit_path):
    """沿 bit_path（列表）依次量化-反量化，返回最终重建向量。"""
    x_cur = x_in
    for b in bit_path:
        _, x_cur = quantizers[b].quant_dequant(x_cur)
    return x_cur

def mse(x_orig, x_recon):
    return ((x_orig - x_recon) ** 2).sum(dim=-1).mean().item()

# ── 定义所有路径 ──────────────────────────────────────────
paths = {
    # 直接量化
    "16→1":           [1],
    "16→2":           [2],
    "16→4":           [4],
    "16→8":           [8],
    # 二级级联 (→1)
    "16→2→1":         [2, 1],
    "16→4→1":         [4, 1],
    "16→8→1":         [8, 1],
    # 二级级联 (→2)
    "16→4→2":         [4, 2],
    "16→8→2":         [8, 2],
    # 二级级联 (→4)
    "16→8→4":         [8, 4],
    # 三级级联
    "16→4→2→1":       [4, 2, 1],
    "16→8→4→1":       [8, 4, 1],
    "16→8→4→2":       [8, 4, 2],
    # 四级级联
    "16→8→4→2→1":     [8, 4, 2, 1],
}

# ── 运行并输出结果 ────────────────────────────────────────
print(f"\nd={d}, n={n} 个单位向量\n")
print(f"{'路径':<20}  {'MSE':>10}  {'最终bit':>8}  {'直接同bit MSE':>14}  {'倍率':>6}")
print("─" * 68)

direct_mse = {}
for b in BITS:
    x_recon = cascade(x, [b])
    direct_mse[b] = mse(x, x_recon)

for name, path in paths.items():
    x_recon = cascade(x, path)
    m = mse(x, x_recon)
    final_b = path[-1]
    d_ref   = direct_mse[final_b]
    ratio   = m / d_ref
    print(f"{name:<20}  {m:>10.5f}  {final_b:>8}  {d_ref:>14.5f}  {ratio:>6.3f}x")

# ── 理论参考 ──────────────────────────────────────────────
print("\n理论上界 sqrt(3π)/2 · 4^{-b}：")
for b in BITS:
    ub = np.sqrt(3 * np.pi) / 2 * 4**(-b)
    print(f"  b={b}: {ub:.5f}")
