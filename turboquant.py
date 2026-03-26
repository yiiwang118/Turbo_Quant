"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
Zandieh et al. 2025

Two algorithms:
  - TurboQuantMSE:  minimizes MSE distortion (Algorithm 1)
  - TurboQuantProd: unbiased inner product estimator (Algorithm 2)
"""

import numpy as np
import torch
from typing import Tuple, Optional
from codebook import compute_codebook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_rotation(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate a random rotation matrix via QR decomposition."""
    A = torch.randn(d, d, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    return Q  # (d, d) orthogonal matrix


def load_or_compute_codebook(b: int, d: int) -> torch.Tensor:
    """Load codebook from file or compute on the fly."""
    import os
    path = os.path.join(os.path.dirname(__file__), "codebooks.npz")
    if os.path.exists(path):
        data = np.load(path)
        key = f"b{b}"
        if key in data:
            return torch.from_numpy(data[key].astype(np.float32))
    # Compute on the fly
    centroids = compute_codebook(2 ** b, d)
    return torch.from_numpy(centroids.astype(np.float32))


# ---------------------------------------------------------------------------
# TurboQuantMSE  (Algorithm 1)
# ---------------------------------------------------------------------------

class TurboQuantMSE:
    """MSE-optimal TurboQuant (Algorithm 1).

    Quantizes x ∈ S^{d-1} to b bits per coordinate.

    Setup (offline, once):
      - Random rotation matrix Π ∈ R^{d×d}
      - Lloyd-Max codebook {c_1, ..., c_{2^b}} for Beta/Gaussian distribution

    Quant(x):
      y = Π x
      idx_j = argmin_k |y_j - c_k|  for each j
      return idx  (b-bit integers, shape [d])

    DeQuant(idx):
      ỹ_j = c_{idx_j}
      x̃ = Π^T ỹ
      return x̃
    """

    def __init__(
        self,
        d: int,
        b: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        self.d = d
        self.b = b
        self.device = device
        self.dtype = dtype

        if seed is not None:
            torch.manual_seed(seed)

        # Rotation matrix Π: (d, d)
        self.Pi = random_rotation(d, device, dtype)  # (d, d)

        # Codebook: (2^b,)
        centroids = load_or_compute_codebook(b, d).to(device=device, dtype=dtype)
        self.centroids = centroids  # (K,) where K = 2^b

    def quant(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize vectors.

        Args:
            x: (..., d) unit-norm vectors

        Returns:
            idx: (..., d) int32 indices into codebook
        """
        # Rotate: (..., d)
        y = x @ self.Pi.T  # (..., d)

        # Find nearest centroid for each coordinate
        # y: (..., d), centroids: (K,)
        # Expand for broadcasting: (..., d, 1) vs (K,)
        diff = y.unsqueeze(-1) - self.centroids  # (..., d, K)
        idx = diff.abs().argmin(dim=-1)           # (..., d)
        return idx.to(torch.int32)

    def dequant(self, idx: torch.Tensor) -> torch.Tensor:
        """Dequantize indices back to vectors.

        Args:
            idx: (..., d) int32 indices

        Returns:
            x_hat: (..., d) reconstructed vectors
        """
        # Look up centroids
        y_hat = self.centroids[idx.long()]  # (..., d)

        # Rotate back
        x_hat = y_hat @ self.Pi  # (..., d)
        return x_hat

    def quant_dequant(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize then dequantize, returning both idx and reconstruction."""
        idx = self.quant(x)
        x_hat = self.dequant(idx)
        return idx, x_hat


# ---------------------------------------------------------------------------
# TurboQuantProd  (Algorithm 2)
# ---------------------------------------------------------------------------

class TurboQuantProd:
    """Inner-product-optimal TurboQuant (Algorithm 2).

    Combines TurboQuantMSE (b-1 bits) with QJL on the residual (1 bit),
    yielding an unbiased inner product estimator at b bits total.

    Setup (offline, once):
      - TurboQuantMSE with bit-width (b-1)
      - Random Gaussian projection matrix S ∈ R^{d×d}

    Quant(x):
      idx = Quant_mse(x)
      r = x - DeQuant_mse(idx)          # residual
      qjl = sign(S r)                    # 1-bit QJL
      return (idx, qjl, ||r||_2)

    DeQuant(idx, qjl, gamma):
      x̃_mse = DeQuant_mse(idx)
      x̃_qjl = sqrt(π/2)/d * gamma * S^T qjl
      return x̃_mse + x̃_qjl
    """

    def __init__(
        self,
        d: int,
        b: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        assert b >= 1, "bit-width b must be >= 1"
        self.d = d
        self.b = b
        self.device = device
        self.dtype = dtype

        if seed is not None:
            torch.manual_seed(seed)

        # MSE quantizer with (b-1) bits
        # For b=1, we use 0-bit MSE (just zero vector) + full QJL
        self.b_mse = max(b - 1, 0)
        if self.b_mse > 0:
            self.mse = TurboQuantMSE(d, self.b_mse, device=device, dtype=dtype)
        else:
            self.mse = None

        # QJL random projection matrix S: (d, d), entries ~ N(0,1)
        self.S = torch.randn(d, d, device=device, dtype=dtype)  # (d, d)

    def quant(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Quantize vectors for inner product estimation.

        Args:
            x: (..., d) unit-norm vectors

        Returns:
            idx:   (..., d) int32 codebook indices (None if b_mse=0)
            qjl:   (..., d) int8 sign bits {-1, +1}
            gamma: (...,) residual norms ||r||_2
        """
        if self.mse is not None:
            idx = self.mse.quant(x)
            x_hat_mse = self.mse.dequant(idx)
            r = x - x_hat_mse  # residual (..., d)
        else:
            idx = None
            r = x  # full vector is residual

        # QJL: sign(S r)
        # r: (..., d), S: (d, d)
        Sr = r @ self.S.T  # (..., d)
        qjl = torch.sign(Sr).to(torch.int8)  # (..., d), values in {-1, 0, +1}
        # Handle exact zeros (rare) -> assign +1
        qjl = torch.where(qjl == 0, torch.ones_like(qjl), qjl)

        # Residual norm
        gamma = torch.linalg.norm(r, dim=-1)  # (...,)

        return idx, qjl, gamma

    def dequant(
        self,
        idx: Optional[torch.Tensor],
        qjl: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize for inner product estimation.

        Args:
            idx:   (..., d) int32 indices (or None)
            qjl:   (..., d) int8 sign bits
            gamma: (...,) residual norms

        Returns:
            x_hat: (..., d) reconstructed vector
        """
        # MSE part
        if self.mse is not None and idx is not None:
            x_hat_mse = self.mse.dequant(idx)  # (..., d)
        else:
            x_hat_mse = torch.zeros(
                *qjl.shape[:-1], self.d, device=self.device, dtype=self.dtype
            )

        # QJL part: sqrt(π/2)/d * gamma * S^T qjl
        # qjl: (..., d), S: (d, d)
        scale = (np.pi / 2) ** 0.5 / self.d
        St_qjl = qjl.to(self.dtype) @ self.S  # (..., d)
        # gamma: (...,) -> (..., 1) for broadcasting
        x_hat_qjl = scale * gamma.unsqueeze(-1) * St_qjl  # (..., d)

        return x_hat_mse + x_hat_qjl

    def quant_dequant(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: quantize then dequantize."""
        idx, qjl, gamma = self.quant(x)
        return self.dequant(idx, qjl, gamma)

    def inner_product_estimate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate <y, x> from quantized x.

        Args:
            x: (..., d) vectors to quantize
            y: (..., d) query vectors

        Returns:
            ip_est: (...,) inner product estimates
        """
        x_hat = self.quant_dequant(x)
        return (y * x_hat).sum(dim=-1)


# ---------------------------------------------------------------------------
# Utility: normalize vectors to unit sphere
# ---------------------------------------------------------------------------

def normalize(x: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize rows of x to unit norm. Returns (x_normalized, norms)."""
    norms = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)
    return x / norms, norms.squeeze(-1)
