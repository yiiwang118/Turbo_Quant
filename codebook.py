"""
Offline precomputation of Lloyd-Max optimal codebooks for TurboQuant.

After random rotation, each coordinate follows a Beta(1/2, (d-1)/2) distribution,
which in high dimensions converges to N(0, 1/d). We solve the continuous k-means
(Lloyd-Max) problem for this distribution to obtain optimal scalar quantizer centroids.
"""

import numpy as np
from scipy import stats, optimize
from typing import Tuple


def beta_pdf_normalized(x: np.ndarray, d: int) -> np.ndarray:
    """PDF of a single coordinate of a uniform random unit vector in R^d.

    The coordinate follows Beta(1/2, (d-1)/2) scaled to [-1, 1],
    i.e., if u ~ Uniform(S^{d-1}), then u_1 ~ Beta(1/2,(d-1)/2) on [-1,1].
    For large d this converges to N(0, 1/d).
    """
    if d >= 100:
        # Gaussian approximation is accurate for d >= 100
        std = 1.0 / np.sqrt(d)
        return stats.norm.pdf(x, loc=0, scale=std)
    else:
        # Use exact Beta(1/2, (d-1)/2) distribution scaled to [-1, 1]
        # If u_1 ~ Beta(a, b) on [0,1], then t = 2*u_1 - 1 ~ scaled Beta on [-1,1]
        # PDF: f(t) = f_beta((t+1)/2) / 2
        a, b = 0.5, (d - 1) / 2.0
        t = (x + 1) / 2.0  # map [-1,1] -> [0,1]
        # clip to avoid boundary issues
        t = np.clip(t, 1e-10, 1 - 1e-10)
        return stats.beta.pdf(t, a, b) / 2.0


def lloyd_max_iteration(
    centroids: np.ndarray,
    pdf_func,
    x_min: float,
    x_max: float,
    n_quad: int = 10000,
) -> np.ndarray:
    """One iteration of Lloyd-Max algorithm.

    Given centroids, compute Voronoi boundaries, then update centroids
    as conditional means within each Voronoi cell.
    """
    k = len(centroids)
    centroids = np.sort(centroids)

    # Voronoi boundaries: midpoints between adjacent centroids
    boundaries = np.concatenate([
        [x_min],
        (centroids[:-1] + centroids[1:]) / 2,
        [x_max]
    ])

    x = np.linspace(x_min, x_max, n_quad)
    dx = x[1] - x[0]
    px = pdf_func(x)

    new_centroids = np.zeros(k)
    for i in range(k):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (x >= lo) & (x < hi)
        mass = np.sum(px[mask]) * dx
        if mass > 0:
            new_centroids[i] = np.sum(x[mask] * px[mask]) * dx / mass
        else:
            new_centroids[i] = centroids[i]

    return new_centroids


def compute_codebook(
    k: int,
    d: int,
    n_iter: int = 500,
    tol: float = 1e-10,
    x_range: float = 5.0,
) -> np.ndarray:
    """Compute Lloyd-Max optimal codebook with k centroids for dimension d.

    Args:
        k: number of centroids (= 2^b for b-bit quantization)
        d: vector dimension (determines coordinate distribution)
        n_iter: maximum iterations
        tol: convergence tolerance
        x_range: range [-x_range/sqrt(d), x_range/sqrt(d)] for integration

    Returns:
        centroids: sorted array of shape (k,)
    """
    std = 1.0 / np.sqrt(d)
    x_min = -x_range * std
    x_max = x_range * std

    pdf_func = lambda x: beta_pdf_normalized(x, d)

    # Initialize centroids uniformly in [-3*std, 3*std]
    centroids = np.linspace(-3 * std, 3 * std, k)

    for _ in range(n_iter):
        new_centroids = lloyd_max_iteration(centroids, pdf_func, x_min, x_max)
        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    return np.sort(new_centroids)


def compute_mse_cost(centroids: np.ndarray, d: int, x_range: float = 5.0) -> float:
    """Compute MSE cost C(f_X, b) for the given codebook."""
    k = len(centroids)
    std = 1.0 / np.sqrt(d)
    x_min = -x_range * std
    x_max = x_range * std
    boundaries = np.concatenate([
        [x_min],
        (centroids[:-1] + centroids[1:]) / 2,
        [x_max]
    ])

    n_quad = 20000
    x = np.linspace(x_min, x_max, n_quad)
    dx = x[1] - x[0]
    px = beta_pdf_normalized(x, d)

    cost = 0.0
    for i in range(k):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (x >= lo) & (x < hi)
        cost += np.sum((x[mask] - centroids[i]) ** 2 * px[mask]) * dx

    return float(cost)


def build_all_codebooks(
    max_bits: int = 8,
    d: int = 1024,
) -> dict:
    """Precompute codebooks for b = 1 to max_bits.

    Returns dict mapping bit-width b -> centroids array of shape (2^b,).
    """
    codebooks = {}
    for b in range(1, max_bits + 1):
        k = 2 ** b
        print(f"Computing codebook for b={b} (k={k})...")
        centroids = compute_codebook(k, d)
        mse = compute_mse_cost(centroids, d)
        # C(f_X, b) is per-coordinate cost; full MSE = d * C
        print(f"  per-coord MSE = {mse:.6f}, full MSE ≈ {d * mse:.4f}")
        print(f"  theoretical upper bound = {np.sqrt(3 * np.pi) / 2 / 4**b:.6f}")
        codebooks[b] = centroids
    return codebooks


if __name__ == "__main__":
    import json

    d = 1024
    codebooks = build_all_codebooks(max_bits=8, d=d)

    # Save as numpy file
    save_dict = {f"b{b}": v for b, v in codebooks.items()}
    np.savez("codebooks.npz", **save_dict)
    print("\nSaved codebooks.npz")

    # Print summary
    print("\nCodebook summary:")
    for b, c in codebooks.items():
        print(f"  b={b}: {len(c)} centroids, range=[{c[0]:.4f}, {c[-1]:.4f}]")
