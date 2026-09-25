from dataclasses import dataclass
from types import SimpleNamespace

import torch
from PIL import Image

from resight_smc.config import Config
from resight_smc.data import Example
from resight_smc.smc import (
    PowerSMC,
    _stratified,
)


@dataclass
class ToyDecoder:
    generated_ids: torch.Tensor
    next_logits: torch.Tensor


class ToyRuntime:
    device = torch.device("cpu")
    terminal_ids = (3,)
    pad_token_id = 4

    @staticmethod
    def _logits(prefix: torch.Tensor) -> torch.Tensor:
        batch = prefix.shape[0]
        logits = torch.tensor([1.4, 0.8, 0.2, -0.5, -8.0]).repeat(batch, 1)
        if prefix.shape[1]:
            last = prefix[:, -1]
            logits[:, 3] = -0.4 + 0.7 * prefix.shape[1]
            logits[torch.arange(batch), last.clamp_max(2)] += 0.5
        return logits

    def prepare_example(self, image, question):
        return SimpleNamespace(
            images=[image], prompt_texts=[question], input_ids=torch.zeros((1, 3), dtype=torch.long)
        )

    def init_full(self, prepared, count, capture_prompt=False):
        empty = torch.empty((count, 0), dtype=torch.long)
        return ToyDecoder(empty, self._logits(empty))

    def advance(self, state, tokens, capture_query=False, visual_attention_bias=None):
        generated = torch.cat([state.generated_ids, tokens.view(-1, 1)], dim=1)
        return ToyDecoder(generated, self._logits(generated))

    def reorder(self, state, indices):
        return ToyDecoder(
            state.generated_ids.index_select(0, indices),
            state.next_logits.index_select(0, indices),
        )

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens if token not in {3, 4})

    def visual_token_count(self, prepared):
        return 0

class ToyVisualRuntime(ToyRuntime):
    """Exercise one visual episode with a deterministic attention proposal."""

    @staticmethod
    def _logits(prefix: torch.Tensor) -> torch.Tensor:
        batch = prefix.shape[0]
        logits = torch.tensor([1.4, 0.8, 0.2, -20.0, -20.0]).repeat(batch, 1)
        if prefix.shape[1] >= 5:
            logits[:, 3] = 5.0
        return logits

    def route_attention(self, count):
        return torch.ones((count, 1, 4), dtype=torch.float32) / 4

    def visual_grid(self, prepared):
        return (1, 2, 2)

    def init_attention_scout(self, state, particle_indices, visual_bias):
        generated = state.generated_ids.index_select(0, particle_indices)
        logits = self._logits(generated)
        logits[:, 1] += 0.5
        return ToyDecoder(generated, logits)


def test_stratified_resampling_draws_once_per_stratum():
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
    actual = _stratified(weights, torch.Generator().manual_seed(11))

    generator = torch.Generator().manual_seed(11)
    positions = (
        torch.arange(4, dtype=torch.float32) + torch.rand(4, generator=generator)
    ) / 4
    expected = torch.searchsorted(torch.cumsum(weights, dim=0), positions).long()

    assert torch.equal(actual, expected)


def test_toy_island_smc_closes_target_and_ancestry():
    cfg = Config(
        method="island_smc",
        num_particles=8,
        num_islands=2,
        max_new_tokens=6,
        alpha=2.0,
        alpha_ramp_tokens=4,
        checkpoint_interval=2,
        ess_threshold=0.9,
    )
    cfg.finalize()
    runtime = ToyRuntime()
    sampler = PowerSMC(runtime, cfg, torch.Generator().manual_seed(7))
    example = Example("toy", "question", Image.new("RGB", (8, 8)), ["0"])
    result = sampler.run(example, "question")
    assert len(result.particle_tokens) == 8
    assert result.diagnostics["num_resampling_events"] > 0
    assert abs(sum(result.terminal_weights) - 1.0) < 1e-5
    assert result.diagnostics["final_effective_roots"] > 0
    assert 0 < result.diagnostics["largest_root_mass"] <= 1


def test_direct_attention_proposal_corrects_to_full_target():
    full = torch.log(torch.tensor([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1]]))
    attention = torch.log(torch.tensor([[0.1, 0.1, 0.8], [0.6, 0.3, 0.1]]))
    # Sampling from q_attention and correcting by q_full/q_attention recovers the target mass.
    corrected_mass = torch.sum(attention.exp() * (full - attention).exp(), dim=-1)
    assert torch.allclose(corrected_mass, torch.ones(2), atol=1e-6)


def test_single_visual_episode_uses_attention_proposal():
    cfg = Config(
        method="resight",
        num_particles=8,
        num_islands=2,
        max_new_tokens=6,
        alpha=2.0,
        alpha_ramp_tokens=6,
        checkpoint_interval=2,
        ess_threshold=0.0,
        scout_fraction=0.5,
        visual_checkpoint=1,
        visual_length=1,
    )
    cfg.finalize()
    sampler = PowerSMC(ToyVisualRuntime(), cfg, torch.Generator().manual_seed(7))
    example = Example("toy", "question", Image.new("RGB", (8, 8)), ["0"])
    result = sampler.run(example, "question")

    assert result.diagnostics["num_scout_episodes"] == 1
    assert result.diagnostics["num_scouts"] == 4
    assert abs(sum(result.terminal_weights) - 1.0) < 1e-5
