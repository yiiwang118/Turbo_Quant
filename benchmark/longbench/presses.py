# TurboQuant KV-cache press (simulation / accuracy-measurement mode)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from benchmark.kvpress.base_press import BasePress
from turboquant import TurboQuantMSE, TurboQuantProd  # type: ignore[import]


@dataclass
class TurboQuantPress(BasePress):
    """Quant → dequant keys/values in-place during prefill.

    Keeps the cache in the model's native float dtype, so there is no
    memory saving — but the quantisation noise is real and measurable.
    Set b_key or b_value to 0 to skip quantising that half.
    """

    b_key:   int = 4   # bits for keys  via TurboQuantProd (inner-product optimal)
    b_value: int = 4   # bits for values via TurboQuantMSE  (MSE optimal)
    seed:    int = 42

    # Per-dimension quantiser cache (created lazily on first use)
    _key_qs: dict[int, TurboQuantProd] = field(default_factory=dict, init=False, repr=False)
    _val_qs: dict[int, TurboQuantMSE]  = field(default_factory=dict, init=False, repr=False)

    # Satisfy BasePress interface — bit-width controls compression, not a ratio
    @property
    def compression_ratio(self) -> float:  return 0.0
    @compression_ratio.setter
    def compression_ratio(self, _: float) -> None: pass

    def _key_q(self, d: int, device: torch.device, dtype: torch.dtype) -> TurboQuantProd:
        if d not in self._key_qs:
            self._key_qs[d] = TurboQuantProd(d, self.b_key, device=device, dtype=dtype, seed=self.seed)
        return self._key_qs[d]

    def _val_q(self, d: int, device: torch.device, dtype: torch.dtype) -> TurboQuantMSE:
        if d not in self._val_qs:
            self._val_qs[d] = TurboQuantMSE(d, self.b_value, device=device, dtype=dtype, seed=self.seed + 1)
        return self._val_qs[d]

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: Optional[torch.Tensor],
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, nkv, seq, d = keys.shape
        device, dtype    = keys.device, keys.dtype

        if self.b_key > 0:
            kq      = self._key_q(d, device, dtype)
            k_flat  = keys.reshape(-1, d)
            k_norms = k_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            # Normalise → quant → dequant → restore scale
            idx, qjl, gamma = kq.quant(k_flat / k_norms)
            keys = (kq.dequant(idx, qjl, gamma) * k_norms).reshape(bsz, nkv, seq, d)

        if self.b_value > 0:
            vq      = self._val_q(d, device, dtype)
            v_flat  = values.reshape(-1, d)
            v_norms = v_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v_idx   = vq.quant(v_flat / v_norms)
            values  = (vq.dequant(v_idx) * v_norms).reshape(bsz, nkv, seq, d)

        return keys, values
