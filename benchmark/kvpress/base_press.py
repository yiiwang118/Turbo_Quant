# Adapted from autokv/kvpress/presses/base_press.py
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _is_quantized(layer: Any) -> bool:
    return hasattr(layer, "_quantized_keys") and hasattr(layer, "_dequantize")


def extract_keys_and_values(cache: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    layer = cache.layers[layer_idx]
    if _is_quantized(layer):
        return layer._dequantize(layer._quantized_keys), layer._dequantize(layer._quantized_values)
    return layer.keys, layer.values


def set_keys_and_values(cache: Any, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    layer = cache.layers[layer_idx]
    if _is_quantized(layer) and hasattr(layer, "_quantize"):
        layer._quantized_keys   = layer._quantize(keys,   axis=getattr(layer, "axis_key",   -1))
        layer._quantized_values = layer._quantize(values, axis=getattr(layer, "axis_value", -1))
        layer.keys   = torch.zeros(0, dtype=keys.dtype,   device=keys.device)
        layer.values = torch.zeros(0, dtype=values.dtype, device=values.device)
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = keys.shape[2]
        return
    layer.keys   = keys
    layer.values = values


# ── Base class ────────────────────────────────────────────────────────────────

@dataclass
class BasePress:
    """Base class for KV-cache compression methods.

    Subclasses implement compress(); the hook machinery here handles
    cache extraction / write-back and prefill-only gating.
    """

    def post_init_from_model(self, model: PreTrainedModel) -> None:
        """Optional: initialise from model config before hooks are attached."""

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward_hook(self, module: nn.Module, _input: tuple, kwargs: dict, output: Any) -> Any:
        hidden_states = kwargs.get("hidden_states") or (_input[0] if _input else None)
        cache         = kwargs.get("past_key_values")
        layer_idx     = getattr(module, "layer_idx", None)

        if hidden_states is None or cache is None or layer_idx is None:
            return output

        # Skip decode steps — only compress during prefill
        cache_pos = kwargs.get("cache_position")
        if cache_pos is not None and cache_pos[-1] > hidden_states.shape[1]:
            return output

        keys, values = extract_keys_and_values(cache, layer_idx)
        attentions   = output[1] if isinstance(output, (list, tuple)) and len(output) > 1 else None
        keys, values = self.compress(module, hidden_states, keys, values, attentions, kwargs)
        set_keys_and_values(cache, layer_idx, keys, values)
        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel):
        self.post_init_from_model(model)

        # Unwrap to the bare transformer stack
        lm = model
        if hasattr(lm, "model"):          lm = lm.model
        if hasattr(lm, "language_model"): lm = lm.language_model

        if not hasattr(lm, "layers"):
            raise ValueError(f"Cannot find transformer layers in {type(model)}")

        hooks: list = []
        try:
            for i, layer in enumerate(lm.layers):
                attn = getattr(layer, "self_attn", None)
                if attn is None or getattr(attn, "is_sliding", False):
                    continue
                if not hasattr(attn, "layer_idx"):
                    attn.layer_idx = i
                if hasattr(lm, "rotary_emb"):
                    try: attn.rotary_emb = lm.rotary_emb
                    except Exception: pass
                hooks.append(attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for h in hooks:
                h.remove()
