"""Minimal local BasePress implementation adapted from autokv/kvpress."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

import torch
from torch import nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def _cache_layer(cache: Any, layer_idx: int) -> Any:
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise ValueError("Unsupported cache object: missing `layers`.")
    return layers[layer_idx]


def _is_quantized_cache_layer(cache_layer: Any) -> bool:
    return (
        hasattr(cache_layer, "_quantized_keys")
        and hasattr(cache_layer, "_quantized_values")
        and hasattr(cache_layer, "_dequantize")
    )


def extract_keys_and_values(cache: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract key/value tensors from one cache layer (quantized or plain)."""
    cache_layer = _cache_layer(cache, layer_idx)

    if _is_quantized_cache_layer(cache_layer):
        keys = cache_layer._dequantize(cache_layer._quantized_keys)
        values = cache_layer._dequantize(cache_layer._quantized_values)
        return keys, values

    return cache_layer.keys, cache_layer.values


def set_keys_and_values(cache: Any, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    """Write key/value tensors back into one cache layer (quantized or plain)."""
    cache_layer = _cache_layer(cache, layer_idx)

    if _is_quantized_cache_layer(cache_layer) and hasattr(cache_layer, "_quantize"):
        axis_key = getattr(cache_layer, "axis_key", -1)
        axis_value = getattr(cache_layer, "axis_value", -1)
        cache_layer._quantized_keys = cache_layer._quantize(keys, axis=axis_key)
        cache_layer._quantized_values = cache_layer._quantize(values, axis=axis_value)
        cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
        cache_layer.values = torch.zeros(0, dtype=values.dtype, device=values.device)
        if hasattr(cache_layer, "cumulative_length"):
            cache_layer.cumulative_length = keys.shape[2]
        return

    cache_layer.keys = keys
    cache_layer.values = values


@dataclass
class BasePress:
    """Base class for KV-cache compression methods."""

    def post_init_from_model(self, model: PreTrainedModel) -> None:
        """Optional hook to initialize method parameters from model."""

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("compress method must be implemented in subclass")

    def forward_hook(self, module: nn.Module, _input: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> Any:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and _input:
            hidden_states = _input[0]
        if hidden_states is None:
            return output

        cache = kwargs.get("past_key_values")
        if cache is None:
            return output

        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            return output

        q_len = hidden_states.shape[1]
        cache_position = kwargs.get("cache_position")
        # Keep the autokv behavior: only compress during prefill.
        if cache_position is not None and cache_position[-1] > q_len:
            return output

        keys, values = extract_keys_and_values(cache, layer_idx)
        attentions = output[1] if isinstance(output, (list, tuple)) and len(output) > 1 else None
        keys, values = self.compress(module, hidden_states, keys, values, attentions, kwargs)
        set_keys_and_values(cache, layer_idx, keys, values)
        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator[None, None, None]:
        self.post_init_from_model(model)

        language_model = model
        if hasattr(model, "model"):
            language_model = model.model  # type: ignore[assignment]
        if hasattr(language_model, "language_model"):
            language_model = language_model.language_model  # type: ignore[assignment]

        layers = getattr(language_model, "layers", None)
        if layers is None:
            raise ValueError("Unsupported model: cannot find transformer `layers`.")

        hooks: list[Any] = []
        try:
            for layer_idx, layer in enumerate(layers):
                attn = getattr(layer, "self_attn", None)
                if attn is None:
                    continue
                if getattr(attn, "is_sliding", False):
                    continue

                if not hasattr(attn, "layer_idx"):
                    setattr(attn, "layer_idx", layer_idx)
                if hasattr(language_model, "rotary_emb"):
                    try:
                        setattr(attn, "rotary_emb", language_model.rotary_emb)
                    except Exception:  # pragma: no cover - best effort compatibility
                        logger.debug("Failed to set rotary_emb on layer %s", layer_idx)

                hooks.append(attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()

