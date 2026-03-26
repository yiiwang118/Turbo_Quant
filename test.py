"""
Test TurboQuant implementation against theoretical bounds from the paper.

Theorem 1 (MSE bound):   Dmse <= sqrt(3*pi)/2 * 4^{-b}
Theorem 2 (inner prod):  Dprod <= sqrt(3*pi^2)/d * 4^{-b}, unbiased
Lower bounds:            Dmse >= 4^{-b},  Dprod >= 1/d * 4^{-b}
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from turboquant import TurboQuantMSE, TurboQuantProd, normalize


def test_mse(d: int = 512, n: int = 2000, bits: list = [1, 2, 3, 4, 5], seed: int = 42):
    """Verify MSE distortion bounds for TurboQuantMSE."""
    print(f"\n{'='*60}")
    print(f"TurboQuantMSE  d={d}, n={n} vectors")
    print(f"{'='*60}")
    print(f"{'b':>4}  {'Dmse':>10}  {'upper_bound':>12}  {'lower_bound':>12}  {'ratio':>8}")
    print("-" * 55)

    torch.manual_seed(seed)
    x = torch.randn(n, d)
    x, _ = normalize(x)  # unit norm

    results = {}
    for b in bits:
        qmse = TurboQuantMSE(d=d, b=b, seed=seed + b)
        _, x_hat = qmse.quant_dequant(x)
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()

        upper = np.sqrt(3 * np.pi) / 2 * 4 ** (-b)
        lower = 4 ** (-b)
        ratio = mse / lower

        print(f"{b:>4}  {mse:>10.5f}  {upper:>12.5f}  {lower:>12.5f}  {ratio:>8.3f}")
        results[b] = mse

    return results


def test_inner_product(d: int = 512, n: int = 2000, bits: list = [1, 2, 3, 4, 5], seed: int = 42):
    """Verify inner product distortion and unbiasedness for TurboQuantProd."""
    print(f"\n{'='*60}")
    print(f"TurboQuantProd  d={d}, n={n} vectors")
    print(f"{'='*60}")
    print(f"{'b':>4}  {'Dprod':>10}  {'upper_bound':>12}  {'lower_bound':>12}  {'bias':>10}")
    print("-" * 60)

    torch.manual_seed(seed)
    x = torch.randn(n, d)
    x, _ = normalize(x)
    y = torch.randn(n, d)  # query vectors (not normalized)

    results = {}
    for b in bits:
        qprod = TurboQuantProd(d=d, b=b, seed=seed + b)

        # Estimate inner products
        ip_true = (y * x).sum(dim=-1)  # (n,)
        ip_est = qprod.inner_product_estimate(x, y)  # (n,)

        error = ip_est - ip_true
        dprod = (error ** 2).mean().item()
        bias = error.mean().item()

        # Bounds use ||y||^2 averaged
        y_norm_sq = (y ** 2).sum(dim=-1).mean().item()
        upper = np.sqrt(3 * np.pi ** 2) / d * y_norm_sq * 4 ** (-b)
        lower = y_norm_sq / d * 4 ** (-b)

        print(f"{b:>4}  {dprod:>10.6f}  {upper:>12.6f}  {lower:>12.6f}  {bias:>10.6f}")
        results[b] = dprod

    return results


def test_bias_mse_vs_prod(d: int = 512, n: int = 5000, b: int = 2, seed: int = 42):
    """Show that TurboQuantMSE is biased but TurboQuantProd is unbiased."""
    print(f"\n{'='*60}")
    print(f"Bias comparison  d={d}, b={b}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    x = torch.randn(n, d)
    x, _ = normalize(x)
    y = torch.randn(n, d)

    ip_true = (y * x).sum(dim=-1)

    # MSE quantizer used for inner product
    qmse = TurboQuantMSE(d=d, b=b, seed=seed)
    _, x_hat = qmse.quant_dequant(x)
    ip_mse = (y * x_hat).sum(dim=-1)

    # Prod quantizer
    qprod = TurboQuantProd(d=d, b=b, seed=seed)
    ip_prod = qprod.inner_product_estimate(x, y)

    bias_mse = (ip_mse - ip_true).mean().item()
    bias_prod = (ip_prod - ip_true).mean().item()

    print(f"TurboQuantMSE  bias = {bias_mse:+.6f}  (expected nonzero for low b)")
    print(f"TurboQuantProd bias = {bias_prod:+.6f}  (expected ~0)")


def test_batch(d: int = 128, n: int = 10000, b: int = 4, seed: int = 0):
    """Quick sanity check with batched inputs."""
    print(f"\n{'='*60}")
    print(f"Batch sanity check  d={d}, n={n}, b={b}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    x = torch.randn(n, d)
    x, norms = normalize(x)

    qmse = TurboQuantMSE(d=d, b=b, seed=seed)
    idx, x_hat = qmse.quant_dequant(x)
    mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
    upper = np.sqrt(3 * np.pi) / 2 * 4 ** (-b)
    print(f"MSE quantizer: Dmse={mse:.6f}  upper_bound={upper:.6f}  OK={mse <= upper * 1.5}")

    qprod = TurboQuantProd(d=d, b=b, seed=seed)
    y = torch.randn(n, d)
    ip_true = (y * x).sum(dim=-1)
    ip_est = qprod.inner_product_estimate(x, y)
    bias = (ip_est - ip_true).mean().item()
    print(f"Prod quantizer: bias={bias:.6f}  (should be ~0)")


def plot_distortion_vs_bits(d: int = 512, n: int = 3000, bits: list = [1, 2, 3, 4, 5]):
    """Plot MSE and inner product distortion vs bit-width alongside bounds."""
    torch.manual_seed(0)
    x = torch.randn(n, d)
    x, _ = normalize(x)
    y = torch.randn(n, d)

    mse_vals, prod_vals = [], []
    for b in bits:
        qmse = TurboQuantMSE(d=d, b=b, seed=b)
        _, x_hat = qmse.quant_dequant(x)
        mse_vals.append(((x - x_hat) ** 2).sum(dim=-1).mean().item())

        qprod = TurboQuantProd(d=d, b=b, seed=b)
        ip_true = (y * x).sum(dim=-1)
        ip_est = qprod.inner_product_estimate(x, y)
        prod_vals.append(((ip_est - ip_true) ** 2).mean().item())

    bits_arr = np.array(bits)
    upper_mse = np.sqrt(3 * np.pi) / 2 * 4.0 ** (-bits_arr)
    lower_mse = 4.0 ** (-bits_arr)
    y_norm_sq = (y ** 2).sum(dim=-1).mean().item()
    upper_prod = np.sqrt(3 * np.pi ** 2) / d * y_norm_sq * 4.0 ** (-bits_arr)
    lower_prod = y_norm_sq / d * 4.0 ** (-bits_arr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogy(bits, mse_vals, 'o-', label='TurboQuantMSE', color='blue')
    ax1.semilogy(bits, upper_mse, '--', label='Upper bound √(3π)/2 · 4^{-b}', color='red')
    ax1.semilogy(bits, lower_mse, '--', label='Lower bound 4^{-b}', color='green')
    ax1.set_xlabel('Bit-width b')
    ax1.set_ylabel('MSE Distortion')
    ax1.set_title('MSE vs Bit-width')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(bits, prod_vals, 's-', label='TurboQuantProd', color='purple')
    ax2.semilogy(bits, upper_prod, '--', label='Upper bound', color='red')
    ax2.semilogy(bits, lower_prod, '--', label='Lower bound', color='green')
    ax2.set_xlabel('Bit-width b')
    ax2.set_ylabel('Inner Product Distortion')
    ax2.set_title('Inner Product Error vs Bit-width')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('distortion_vs_bits.png', dpi=150)
    print("\nSaved distortion_vs_bits.png")


if __name__ == "__main__":
    D = 512
    N = 3000

    test_batch(d=128, n=10000, b=4)
    mse_results = test_mse(d=D, n=N)
    prod_results = test_inner_product(d=D, n=N)
    test_bias_mse_vs_prod(d=D, n=N, b=2)

    try:
        plot_distortion_vs_bits(d=D, n=N)
    except Exception as e:
        print(f"Plot skipped: {e}")
