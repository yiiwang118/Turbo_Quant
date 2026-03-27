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
    """Generate a Haar-distributed random orthogonal matrix via QR decomposition.

    Plain QR gives a Q whose distribution is not uniform on O(d). The standard
    fix is to multiply each column of Q by the sign of the corresponding diagonal
    entry of R, which forces R to have positive diagonal and makes Q Haar-distributed.
    """
    A = torch.randn(d, d, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    signs = torch.diag(R).sign()
    Q = Q * signs.unsqueeze(0)
    return Q


def load_or_compute_codebook(b: int, d: int) -> torch.Tensor:
    """Load codebook from file or compute on the fly."""
    import os
    path = os.path.join(os.path.dirname(__file__), "codebooks.npz")
    if os.path.exists(path):
        data = np.load(path)
        key = f"b{b}_d{d}"
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

    def asymmetric_inner_product(
        self,
        queries: torch.Tensor,
        idx: Optional[torch.Tensor],
        qjl: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Batched asymmetric inner product <queries, compressed_keys>.

        Computes attention scores directly from the compressed key representation
        without decompressing. Estimator (unbiased):

            <q, k> ≈ <q, k_mse> + γ · √(π/2)/d · <S q, qjl_k>

        Args:
            queries : (..., S_q, d)
            idx     : (..., S_k, d) int32 MSE indices, or None when b_mse=0
            qjl     : (..., S_k, d) int8 QJL sign bits
            gamma   : (..., S_k)    residual L2 norms

        Returns:
            scores  : (..., S_q, S_k)
        """
        # term1: <q, k_mse>  shape (..., S_q, S_k)
        if self.mse is not None and idx is not None:
            k_mse = self.mse.dequant(idx)                          # (..., S_k, d)
            term1 = torch.matmul(queries, k_mse.transpose(-2, -1))
        else:
            term1 = queries.new_zeros(*queries.shape[:-2], queries.shape[-2], qjl.shape[-2])

        # term2: γ · √(π/2)/d · <S q, qjl>  shape (..., S_q, S_k)
        q_proj = torch.matmul(queries, self.S.T)                   # (..., S_q, d)
        qjl_ip = torch.matmul(q_proj, qjl.to(queries.dtype).transpose(-2, -1))
        scale = (np.pi / 2) ** 0.5 / self.d
        term2 = scale * qjl_ip * gamma.unsqueeze(-2)               # (..., S_q, S_k)

        return term1 + term2


# ---------------------------------------------------------------------------
# Utility: normalize vectors to unit sphere
# ---------------------------------------------------------------------------

def normalize(x: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize rows of x to unit norm. Returns (x_normalized, norms)."""
    norms = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)
    return x / norms, norms.squeeze(-1)


# ---------------------------------------------------------------------------
# TurboQuantKVCache
# ---------------------------------------------------------------------------

class TurboQuantKVCache:
    """KV cache backed by TurboQuant.

    Keys   → TurboQuantProd  (unbiased inner product for attention scores)
    Values → TurboQuantMSE   (MSE reconstruction for weighted sum)

    Handles non-unit-norm inputs: normalises before quantisation and restores
    the original scale during score computation / reconstruction.

    Usage::

        cache = TurboQuantKVCache(d_key=128, d_value=128, b_key=4, b_value=4)
        cache.append(keys, values)          # (..., S, d) — arbitrary norm
        scores = cache.attention_scores(q)  # (..., S_q, S_k) — no decompression
        vals   = cache.get_values()         # (..., S_k, d_value)
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        b_key: int = 4,
        b_value: int = 4,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        self.d_key   = d_key
        self.d_value = d_value
        self.device  = device
        self.dtype   = dtype

        self.key_q = TurboQuantProd(d_key,   b_key,   device=device, dtype=dtype, seed=seed)
        self.val_q = TurboQuantMSE (d_value, b_value, device=device, dtype=dtype, seed=seed + 1)

        self._key_cache: list = []   # dicts: {idx, qjl, gamma, norms}
        self._val_cache: list = []   # dicts: {idx, norms}

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Quantise and store new KV pairs.

        Args:
            keys   : (..., S, d_key)
            values : (..., S, d_value)
        """
        # Keys — normalise to unit sphere first
        k_norms = torch.linalg.norm(keys,   dim=-1, keepdim=True).clamp(min=1e-8)
        idx, qjl, gamma = self.key_q.quant(keys / k_norms)
        self._key_cache.append({
            "idx": idx, "qjl": qjl, "gamma": gamma,
            "norms": k_norms.squeeze(-1),
        })

        # Values — same normalise-before-quantise pattern
        v_norms = torch.linalg.norm(values, dim=-1, keepdim=True).clamp(min=1e-8)
        val_idx = self.val_q.quant(values / v_norms)
        self._val_cache.append({
            "idx": val_idx,
            "norms": v_norms.squeeze(-1),
        })

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """Compute <q, k> for all cached keys without decompressing K.

        Args:
            queries : (..., S_q, d_key)

        Returns:
            scores  : (..., S_q, S_k_total)
        """
        parts = []
        for entry in self._key_cache:
            # Scores against unit-norm keys: (..., S_q, S_k)
            s = self.key_q.asymmetric_inner_product(
                queries, entry["idx"], entry["qjl"], entry["gamma"]
            )
            # Restore original key scale: broadcast (..., 1, S_k)
            s = s * entry["norms"].unsqueeze(-2)
            parts.append(s)
        return torch.cat(parts, dim=-1)

    def get_values(self) -> torch.Tensor:
        """Reconstruct all cached values.

        Returns:
            values : (..., S_k_total, d_value)
        """
        parts = []
        for entry in self._val_cache:
            v = self.val_q.dequant(entry["idx"])           # unit-norm reconstruction
            v = v * entry["norms"].unsqueeze(-1)           # restore scale
            parts.append(v)
        return torch.cat(parts, dim=-2)

    def memory_bits(self) -> dict:
        """Estimate storage cost in bits."""
        b_mse = self.key_q.b_mse
        b_val = self.val_q.b

        key_bits = val_bits = fp16_bits = 0
        for e in self._key_cache:
            n = e["gamma"].numel()          # number of key vectors
            key_bits += e["idx"].numel() * b_mse if e["idx"] is not None else 0
            key_bits += e["qjl"].numel()    # 1 bit each
            key_bits += n * 16              # gamma  (fp16)
            key_bits += n * 16              # norms  (fp16)
            fp16_bits += n * self.d_key * 16

        for e in self._val_cache:
            n = e["norms"].numel()
            val_bits  += e["idx"].numel() * b_val
            val_bits  += n * 16             # norms  (fp16)
            fp16_bits += n * self.d_value * 16

        total = key_bits + val_bits
        return {
            "key_bits":          key_bits,
            "val_bits":          val_bits,
            "total_bits":        total,
            "fp16_bits":         fp16_bits,
            "compression_ratio": fp16_bits / total if total > 0 else 0.0,
        }

    def __len__(self) -> int:
        return sum(e["gamma"].numel() for e in self._key_cache)
