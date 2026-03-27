"""LongBench-specific press implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from benchmark.kvpress.base_press import BasePress
from turboquant import TurboQuantMSE, TurboQuantProd


@dataclass
class TurboQuantPress(BasePress):
    """Quantize+dequantize KV cache with TurboQuant to measure quality impact."""

    b_key: int = 4
    b_value: int = 4
    seed: int = 42

    _key_qs: dict[int, TurboQuantProd] = field(default_factory=dict, init=False, repr=False)
    _val_qs: dict[int, TurboQuantMSE] = field(default_factory=dict, init=False, repr=False)

    @property
    def compression_ratio(self) -> float:
        return 0.0

    @compression_ratio.setter
    def compression_ratio(self, value: float) -> None:
        _ = value

    def _key_q(self, dim: int, device: torch.device, dtype: torch.dtype) -> TurboQuantProd:
        if dim not in self._key_qs:
            self._key_qs[dim] = TurboQuantProd(
                dim,
                self.b_key,
                device=device,
                dtype=dtype,
                seed=self.seed,
            )
        return self._key_qs[dim]

    def _val_q(self, dim: int, device: torch.device, dtype: torch.dtype) -> TurboQuantMSE:
        if dim not in self._val_qs:
            self._val_qs[dim] = TurboQuantMSE(
                dim,
                self.b_value,
                device=device,
                dtype=dtype,
                seed=self.seed + 1,
            )
        return self._val_qs[dim]

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: Optional[torch.Tensor],
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = module, hidden_states, attentions, kwargs

        bsz, num_kv_heads, seq_len, dim = keys.shape
        device, dtype = keys.device, keys.dtype

        if self.b_key > 0:
            key_quantizer = self._key_q(dim, device, dtype)
            key_flat = keys.reshape(-1, dim)
            key_norms = key_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            idx, qjl, gamma = key_quantizer.quant(key_flat / key_norms)
            key_hat = key_quantizer.dequant(idx, qjl, gamma) * key_norms
            keys = key_hat.reshape(bsz, num_kv_heads, seq_len, dim)

        if self.b_value > 0:
            value_quantizer = self._val_q(dim, device, dtype)
            value_flat = values.reshape(-1, dim)
            value_norms = value_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            value_idx = value_quantizer.quant(value_flat / value_norms)
            value_hat = value_quantizer.dequant(value_idx) * value_norms
            values = value_hat.reshape(bsz, num_kv_heads, seq_len, dim)

        return keys, values

