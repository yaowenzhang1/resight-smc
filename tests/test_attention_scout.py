import torch

from resight_smc.model import TransformersVLM


class _CaptureAttention(torch.nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        self.received_mask = attention_mask
        return hidden_states


def test_visual_reactivation_adds_bias_only_to_visual_keys():
    runtime = TransformersVLM.__new__(TransformersVLM)
    attention = _CaptureAttention()
    runtime._attention_layers = [attention]
    runtime._visual_positions = torch.tensor([1, 3])
    hidden = torch.zeros((1, 1, 4))
    causal = torch.zeros((1, 1, 1, 5))
    visual_bias = torch.tensor([[0.5, 1.5]])

    with runtime._visual_attention_reactivation(visual_bias, key_length=5):
        attention(hidden_states=hidden, attention_mask=causal)

    expected = torch.tensor([[[[0.0, 0.5, 0.0, 1.5, 0.0]]]])
    assert torch.equal(attention.received_mask, expected)
    assert torch.equal(causal, torch.zeros_like(causal))
