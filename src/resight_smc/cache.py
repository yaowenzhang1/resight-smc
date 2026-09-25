from __future__ import annotations

import copy
from typing import Any

import torch


def _device(cache: Any) -> torch.device:
    """Find the KV tensor device in either Hugging Face cache representation."""

    if hasattr(cache, "layers"):
        for layer in cache.layers:
            tensor = getattr(layer, "keys", None)
            if torch.is_tensor(tensor):
                return tensor.device
    if isinstance(cache, (tuple, list)):
        for layer in cache:
            if isinstance(layer, (tuple, list)) and layer and torch.is_tensor(layer[0]):
                return layer[0].device
    return torch.device("cpu")


def _select(value: Any, indices: torch.Tensor) -> Any:
    """Recursively select batch rows from a legacy cache, keeping tensors contiguous."""

    if torch.is_tensor(value):
        return value.index_select(0, indices.to(value.device)).contiguous()
    if isinstance(value, tuple):
        return tuple(_select(item, indices) for item in value)
    if isinstance(value, list):
        return [_select(item, indices) for item in value]
    return value


def reorder_cache(model: Any, cache: Any, indices: torch.Tensor) -> Any:
    """Reorder a Hugging Face KV cache by ancestor indices."""

    if hasattr(cache, "reorder_cache"):
        cache.reorder_cache(indices.to(_device(cache)))
        return cache
    if hasattr(cache, "batch_select_indices"):
        cache.batch_select_indices(indices.to(_device(cache)))
        return cache
    if hasattr(model, "_reorder_cache"):
        return model._reorder_cache(cache, indices.to(_device(cache)))
    return _select(cache, indices)


def repeat_cache(model: Any, cache: Any, count: int) -> Any:
    """Replicate one full-image prefill cache for ``K*M`` initial particles."""

    if count == 1:
        return cache
    if hasattr(cache, "batch_repeat_interleave"):
        cache.batch_repeat_interleave(count)
        return cache
    indices = torch.zeros(count, dtype=torch.long, device=_device(cache))
    return reorder_cache(model, cache, indices)


def copy_cache_rows(cache: Any, indices: torch.Tensor) -> Any:
    """Copy selected batch rows so temporary scout branches can extend KV independently."""

    indices = indices.to(_device(cache)).long()
    if hasattr(cache, "layers"):
        selected = copy.copy(cache)
        selected.layers = []
        for layer in cache.layers:
            copied_layer = copy.copy(layer)
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if torch.is_tensor(keys):
                copied_layer.keys = keys.index_select(0, indices).contiguous()
            if torch.is_tensor(values):
                copied_layer.values = values.index_select(0, indices).contiguous()
            selected.layers.append(copied_layer)
        return selected
    return _select(cache, indices)


def crop_cache(cache: Any, max_length: int) -> Any:
    """Crop an independent branch to the specified prefix length."""

    if hasattr(cache, "crop"):
        cache.crop(max_length)
        return cache
    if isinstance(cache, tuple):
        return tuple(crop_cache(layer, max_length) for layer in cache)
    if isinstance(cache, list):
        return [crop_cache(layer, max_length) for layer in cache]
    if torch.is_tensor(cache) and cache.ndim >= 3:
        return cache[..., :max_length, :]
    return cache
