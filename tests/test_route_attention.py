from types import SimpleNamespace

import torch

from resight_smc.model import TransformersVLM


def test_route_attention_matches_scaled_raw_qk_attention():
    runtime = TransformersVLM.__new__(TransformersVLM)
    runtime._attention = SimpleNamespace(head_dim=2, scaling=0.5)
    runtime._response_query = torch.tensor([[[2.0, 1.0]]])
    runtime._visual_keys = torch.tensor([[[1.0, 0.0], [0.0, 3.0]]])

    relevance = runtime.route_attention(count=1)
    expected = torch.softmax(torch.tensor([[[1.0, 1.5]]]), dim=-1)

    assert torch.allclose(relevance, expected)
