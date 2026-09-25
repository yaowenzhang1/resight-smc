import torch
from transformers.cache_utils import DynamicCache

from resight_smc.cache import copy_cache_rows, crop_cache


def test_scout_cache_copy_selects_rows_without_mutating_full_cache():
    cache = DynamicCache()
    keys = torch.arange(3 * 2 * 4 * 2).reshape(3, 2, 4, 2).float()
    values = keys + 100
    cache.update(keys, values, layer_idx=0)

    selected = copy_cache_rows(cache, torch.tensor([2, 0]))
    crop_cache(selected, 3)

    assert cache.layers[0].keys.shape == (3, 2, 4, 2)
    assert selected.layers[0].keys.shape == (2, 2, 3, 2)
    assert torch.equal(selected.layers[0].keys[0], keys[2, :, :3])
    assert torch.equal(selected.layers[0].values[1], values[0, :, :3])
