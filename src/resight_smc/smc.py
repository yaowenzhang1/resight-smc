from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .config import Config
from .data import Example
from .model import DecoderState, PreparedBatch, TransformersVLM
from .vision import (
    Route,
    region_attention_bias,
    region_bank,
    region_relevance,
    select_routes,
)


@dataclass
class Population:
    """Hold the minimal state representing joint particle history ``F_t``.

    ``tokens`` stores trajectories and ``log_base_probability`` their cumulative
    full-image log likelihoods. Weights, island masses, and ancestry complete the
    SMC population state.
    """

    tokens: torch.Tensor
    log_weights: torch.Tensor
    log_base_probability: torch.Tensor
    island_log_z: torch.Tensor
    root_ids: torch.Tensor
    done: torch.Tensor


@dataclass
class SMCResult:
    """Return the weighted first-stage trajectories and run diagnostics."""

    answer_text: str
    answer_tokens: list[int]
    particle_texts: list[str]
    particle_tokens: list[list[int]]
    terminal_weights: list[float]
    diagnostics: dict[str, Any]


def _sync_time() -> float:
    """Synchronize CUDA before reading comparable wall-clock timings."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _beta(
    step: int, horizon: int, alpha: float | torch.Tensor, ramp: int
) -> float | torch.Tensor:
    """Compute the linear bridge exponent for ``varphi_t = p_F,t ** beta_t``."""

    if step == horizon:
        return alpha
    return 1.0 + (alpha - 1.0) * min(step / ramp, 1.0)


def _sample(log_probs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Sample by Gumbel-max in log space without low-precision tail probabilities."""

    noise = torch.empty_like(log_probs, dtype=torch.float32).exponential_(1.0, generator=generator)
    return torch.argmax(log_probs.float() - noise.log(), dim=-1)


def _stratified(weights: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Sample independently within equal strata, then select ancestors by CDF."""

    count = weights.numel()
    positions = (
        torch.arange(count, device=weights.device, dtype=torch.float32)
        + torch.rand(count, device=weights.device, generator=generator)
    ) / count
    cdf = torch.cumsum(weights, dim=0)
    cdf[-1] = 1.0
    return torch.searchsorted(cdf, positions).long()


def _ess(log_weights: torch.Tensor) -> float:
    """Compute island ESS as ``1 / sum_m(w_bar_m**2)``."""

    probabilities = torch.softmax(log_weights.float(), dim=0)
    return float(1.0 / probabilities.square().sum().item())


class PowerSMC:
    """Implement first-stage sequence-power SMC with islands and visual scouts."""

    def __init__(self, runtime: TransformersVLM, cfg: Config, generator: torch.Generator):
        """Bind the frozen LVLM, configuration, and sampling generator."""

        self.runtime = runtime
        self.cfg = cfg
        self.generator = generator
        self.regions = region_bank()

    def _shape(self) -> tuple[int, int, bool]:
        """Return population shape ``(K, M, visual_enabled)`` for SMC methods."""

        if self.cfg.method == "power_smc":
            return 1, self.cfg.num_particles, False
        if self.cfg.method == "island_smc":
            return self.cfg.num_islands, int(self.cfg.particles_per_island), False
        return self.cfg.num_islands, int(self.cfg.particles_per_island), True

    def _update_weights(
        self,
        state: Population,
        increment: torch.Tensor,
        islands: int,
        particles_per_island: int,
    ) -> None:
        """Apply ``w_t = w_{t-1} G_t`` and update ``Z_hat_{k,t}`` per island.

        ``increment`` is ``log G_t``. Subtracting island ``log_c`` normalizes
        weights while adding it to ``island_log_z`` updates island mass.
        """

        for island in range(islands):
            start = island * particles_per_island
            end = start + particles_per_island
            unnormalized = state.log_weights[start:end] + increment[start:end]
            log_c = torch.logsumexp(unnormalized, dim=0)
            state.log_weights[start:end] = unnormalized - log_c
            state.island_log_z[island] += log_c

    def _island_probabilities(self, state: Population) -> torch.Tensor:
        """Compute island probabilities ``Omega_k`` from ``Z_hat_k``."""

        return torch.softmax(state.island_log_z.float(), dim=0)

    def _global_masses(
        self, state: Population, islands: int, particles_per_island: int
    ) -> torch.Tensor:
        """Compute pooled particle mass ``W_tilde_{k,m} = Omega_k * w_bar_{k,m}``."""

        island_probabilities = self._island_probabilities(state)
        masses = torch.empty_like(state.log_weights, dtype=torch.float32)
        for island in range(islands):
            start = island * particles_per_island
            end = start + particles_per_island
            masses[start:end] = island_probabilities[island] * torch.softmax(
                state.log_weights[start:end].float(), dim=0
            )
        return masses

    def _genealogy(
        self, state: Population, islands: int, particles_per_island: int
    ) -> dict[str, Any]:
        """Aggregate pooled mass by initial ancestor to measure genealogy diversity."""

        masses = self._global_masses(state, islands, particles_per_island)
        root_mass: dict[int, float] = {}
        for root, mass in zip(state.root_ids.tolist(), masses.tolist()):
            root_mass[int(root)] = root_mass.get(int(root), 0.0) + float(mass)
        values = list(root_mass.values())
        effective_roots = math.exp(-sum(value * math.log(value) for value in values if value > 0))
        return {"effective_roots": effective_roots, "largest_root_mass": max(values)}

    def _resample(
        self,
        state: Population,
        decoder: DecoderState,
        island: int,
        particles_per_island: int,
        reorder_decoder: bool,
    ) -> tuple[Population, DecoderState]:
        """Stratified-resample one island using its normalized weights.

        Ancestors remain within the island. Reorder trajectories and the
        full-image KV cache with the same indices, then reset weights to ``1/M``.
        """

        start = island * particles_per_island
        end = start + particles_per_island
        local = _stratified(
            torch.softmax(state.log_weights[start:end].float(), dim=0), self.generator
        )
        indices = torch.arange(state.tokens.shape[0], device=state.tokens.device)
        indices[start:end] = local + start
        state.tokens = state.tokens.index_select(0, indices)
        state.log_weights = state.log_weights.index_select(0, indices)
        state.log_base_probability = state.log_base_probability.index_select(0, indices)
        state.root_ids = state.root_ids.index_select(0, indices)
        state.done = state.done.index_select(0, indices)
        state.log_weights[start:end] = -math.log(particles_per_island)
        if reorder_decoder:
            decoder = self.runtime.reorder(decoder, indices)
        return state, decoder

    def _start_visual_phase(
        self,
        prepared: PreparedBatch,
        state: Population,
        decoder: DecoderState,
        islands: int,
        particles_per_island: int,
    ) -> tuple[
        list[Route],
        DecoderState | None,
        torch.Tensor | None,
        dict[str, float],
    ]:
        """Build attention-reactivated scouts from the population at ``tau_vis``.

        Compute prefix-conditioned relevance and pooled particle masses, route
        particle--region pairs with island quotas and the configured IoU penalty,
        then fork a temporary decoder from the persistent full-image state.
        Proposal decisions are complete before the next token draw.
        """

        unfinished_capacity = 0
        for island in range(islands):
            start = island * particles_per_island
            end = start + particles_per_island
            unfinished_capacity += max(int((~state.done[start:end]).sum().item()) - 1, 0)
        if unfinished_capacity == 0:
            return [], None, None, {}
        route_start = _sync_time()
        visual_attention = self.runtime.route_attention(state.tokens.shape[0])
        relevance = region_relevance(
            visual_attention,
            self.runtime.visual_grid(prepared),
            self.regions,
            self.cfg.area_exponent,
        )
        routes = select_routes(
            relevance,
            state.log_weights,
            state.island_log_z,
            state.done,
            num_islands=islands,
            particles_per_island=particles_per_island,
            scout_fraction=self.cfg.scout_fraction,
            overlap_penalty=self.cfg.overlap_penalty,
            overlap_penalty_scope=self.cfg.overlap_penalty_scope,
            regions=self.regions,
        )
        route_seconds = _sync_time() - route_start
        if not routes:
            return [], None, None, {"relevance_routing": route_seconds}

        attention_bias = region_attention_bias(
            self.runtime.visual_grid(prepared),
            [self.regions[route.region_index] for route in routes],
            self.cfg.image_attention_logit_bias,
            self.cfg.region_attention_logit_bias,
            device=self.runtime.device,
        )
        branch_start = _sync_time()
        particle_indices = torch.tensor(
            [route.particle for route in routes], dtype=torch.long, device=self.runtime.device
        )
        attention_decoder = self.runtime.init_attention_scout(
            decoder, particle_indices, attention_bias
        )
        branch_seconds = _sync_time() - branch_start
        return (
            routes,
            attention_decoder,
            attention_bias,
            {
                "relevance_routing": route_seconds,
                "attention_scout_initialization": branch_seconds,
            },
        )

    def run(self, example: Example, question: str) -> SMCResult:
        """Run first-stage Island Power-SMC for one example.

        Each step samples from ``q_t^F`` or scout ``q_t^A`` and uses
        ``log G_t = (beta_t-beta_{t-1}) log p_F,<t + beta_t log p_F,t - log q_t``
        to correct to the same full-image sequence-power target. ESS checkpoints
        trigger stratified resampling only within islands. The returned terminal
        trajectories feed the answer-power readout.
        """

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        total_start = _sync_time()
        islands, particles_per_island, visual_enabled = self._shape()
        count = islands * particles_per_island
        visual_checkpoint = self.cfg.visual_checkpoint
        visual_end = visual_checkpoint + self.cfg.visual_length
        prefill_start = _sync_time()
        prepared = self.runtime.prepare_example(example.image, question)
        decoder = self.runtime.init_full(prepared, count, capture_prompt=visual_enabled)
        full_prefill_seconds = _sync_time() - prefill_start
        prompt_qk_seconds = float(getattr(self.runtime, "qk_projection_seconds", 0.0))
        device = self.runtime.device
        particle_alphas = torch.full((count,), self.cfg.alpha, dtype=torch.float32, device=device)
        state = Population(
            tokens=torch.empty((count, 0), dtype=torch.long, device=device),
            log_weights=torch.full((count,), -math.log(particles_per_island), device=device),
            log_base_probability=torch.zeros(count, dtype=torch.float32, device=device),
            island_log_z=torch.zeros(islands, dtype=torch.float32, device=device),
            root_ids=torch.arange(count, device=device),
            done=torch.zeros(count, dtype=torch.bool, device=device),
        )

        routes: list[Route] = []
        attention_decoder: DecoderState | None = None
        scout_attention_bias: torch.Tensor | None = None
        resampling_count = 0
        num_scouts = 0
        timings = {
            "full_image_prefill": max(0.0, full_prefill_seconds - prompt_qk_seconds),
            "full_propagation": 0.0,
            "local_resampling": 0.0,
            "relevance_routing": prompt_qk_seconds,
            "attention_scout_initialization": 0.0,
            "attention_scout_decode": 0.0,
        }
        full_decoder_positions = int(prepared.input_ids.numel())
        attention_scout_decoder_positions = 0
        previous_beta = torch.ones(count, dtype=torch.float32, device=device)
        for step in range(1, self.cfg.max_new_tokens + 1):
            beta = _beta(
                step,
                self.cfg.max_new_tokens,
                particle_alphas,
                self.cfg.alpha_ramp_tokens,
            )
            done_before = state.done.clone()
            active = ~done_before
            old_log_base = state.log_base_probability.clone()
            sampled = torch.full(
                (count,), self.runtime.pad_token_id, dtype=torch.long, device=device
            )
            selected_base = torch.zeros(count, dtype=torch.float32, device=device)
            selected_proposal = torch.zeros(count, dtype=torch.float32, device=device)

            full_start = _sync_time()
            if bool(active.any()):
                full_log_p = F.log_softmax(decoder.next_logits.float(), dim=-1)
                full_log_q = F.log_softmax(beta[:, None] * full_log_p, dim=-1)
                normal_rows = active.clone()
                scout_rows: list[int] = []
                if attention_decoder is not None and routes:
                    scout_rows = [
                        route.particle for route in routes if bool(active[route.particle])
                    ]
                    if scout_rows:
                        normal_rows[torch.tensor(scout_rows, device=device)] = False

                if bool(normal_rows.any()):
                    row_ids = torch.nonzero(normal_rows, as_tuple=False).flatten()
                    tokens = _sample(full_log_q[row_ids], self.generator)
                    sampled[row_ids] = tokens
                    selected_base[row_ids] = (
                        full_log_p[row_ids].gather(1, tokens[:, None]).squeeze(1)
                    )
                    selected_proposal[row_ids] = (
                        full_log_q[row_ids].gather(1, tokens[:, None]).squeeze(1)
                    )

                if scout_rows:
                    route_positions = [
                        index for index, route in enumerate(routes) if bool(active[route.particle])
                    ]
                    particle_rows = torch.tensor(scout_rows, dtype=torch.long, device=device)
                    attention_rows = torch.tensor(
                        route_positions, dtype=torch.long, device=device
                    )
                    attention_log_p = F.log_softmax(
                        attention_decoder.next_logits[attention_rows].float(), dim=-1
                    )
                    attention_log_q = F.log_softmax(
                        beta[particle_rows, None] * attention_log_p, dim=-1
                    )
                    tokens = _sample(attention_log_q, self.generator)
                    sampled[particle_rows] = tokens
                    selected_base[particle_rows] = (
                        full_log_p[particle_rows].gather(1, tokens[:, None]).squeeze(1)
                    )
                    sampled_attention = attention_log_q.gather(
                        1, tokens[:, None]
                    ).squeeze(1)
                    selected_proposal[particle_rows] = sampled_attention

                increment = (beta - previous_beta) * old_log_base
                increment[active] += (
                    beta[active] * selected_base[active] - selected_proposal[active]
                )
                state.log_base_probability[active] += selected_base[active]
                self._update_weights(state, increment, islands, particles_per_island)
                state.tokens = torch.cat([state.tokens, sampled[:, None]], dim=1)
                for token_id in self.runtime.terminal_ids:
                    state.done |= sampled.eq(token_id)

                capture_query = visual_enabled and step == visual_checkpoint
                qk_before = float(getattr(self.runtime, "qk_projection_seconds", 0.0))
                decoder = self.runtime.advance(decoder, sampled, capture_query=capture_query)
                full_decoder_positions += count
                full_elapsed = _sync_time() - full_start
                qk_after = float(getattr(self.runtime, "qk_projection_seconds", 0.0))
                qk_elapsed = qk_after - qk_before
                timings["full_propagation"] += max(0.0, full_elapsed - qk_elapsed)
                timings["relevance_routing"] += qk_elapsed
                if attention_decoder is not None and step < visual_end:
                    assert scout_attention_bias is not None
                    attention_start = _sync_time()
                    attention_tokens = sampled[[route.particle for route in routes]]
                    attention_decoder = self.runtime.advance(
                        attention_decoder,
                        attention_tokens,
                        visual_attention_bias=scout_attention_bias,
                    )
                    attention_scout_decoder_positions += len(routes)
                    timings["attention_scout_decode"] += _sync_time() - attention_start
            else:
                increment = (beta - previous_beta) * old_log_base
                self._update_weights(state, increment, islands, particles_per_island)
                state.tokens = torch.cat([state.tokens, sampled[:, None]], dim=1)

            if attention_decoder is not None and step == visual_end:
                attention_decoder = None
                scout_attention_bias = None

            if step % self.cfg.checkpoint_interval == 0 or step == self.cfg.max_new_tokens:
                for island in range(islands):
                    start = island * particles_per_island
                    end = start + particles_per_island
                    value = _ess(state.log_weights[start:end])
                    if value < self.cfg.ess_threshold * particles_per_island:
                        resample_start = _sync_time()
                        state, decoder = self._resample(
                            state,
                            decoder,
                            island,
                            particles_per_island,
                            reorder_decoder=step < self.cfg.max_new_tokens
                            and not bool(state.done.all()),
                        )
                        timings["local_resampling"] += _sync_time() - resample_start
                        resampling_count += 1

            if visual_enabled and step == visual_checkpoint:
                (
                    routes,
                    attention_decoder,
                    scout_attention_bias,
                    visual_timing,
                ) = self._start_visual_phase(
                    prepared,
                    state,
                    decoder,
                    islands,
                    particles_per_island,
                )
                for key, value in visual_timing.items():
                    timings[key] += value
                if attention_decoder is not None:
                    attention_scout_decoder_positions += len(routes)
                num_scouts = len(routes)
            previous_beta = beta

        masses = self._global_masses(state, islands, particles_per_island)
        # Draw an island in proportion to its normalizing-mass estimate.
        island_probabilities = self._island_probabilities(state)
        chosen_island = int(
            _sample(island_probabilities.log()[None, :], self.generator)[0]
        )
        island_start = chosen_island * particles_per_island
        chosen_local = int(
            _sample(
                state.log_weights[island_start : island_start + particles_per_island][None, :],
                self.generator,
            )[0]
        )
        chosen = island_start + chosen_local
        particle_tokens = [row.tolist() for row in state.tokens]
        particle_texts = [self.runtime.decode(tokens) for tokens in particle_tokens]
        total_seconds = _sync_time() - total_start
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        full_tokens = self.runtime.visual_token_count(prepared)
        genealogy = self._genealogy(state, islands, particles_per_island)
        diagnostics = {
            "latency_seconds": total_seconds,
            "peak_vram_gb": peak_gb,
            "num_islands": islands,
            "particles_per_island": particles_per_island,
            "num_resampling_events": resampling_count,
            "num_scout_episodes": int(num_scouts > 0),
            "num_scouts": num_scouts,
            "chosen_island": chosen_island,
            "chosen_particle": chosen_local,
            "full_visual_tokens": full_tokens,
            "additional_visual_tokens": 0,
            "total_encoded_visual_tokens": full_tokens,
            "full_decoder_positions": full_decoder_positions,
            "attention_scout_decoder_positions": attention_scout_decoder_positions,
            "island_probabilities": island_probabilities.tolist(),
            "final_effective_roots": genealogy["effective_roots"],
            "largest_root_mass": genealogy["largest_root_mass"],
            "timing": timings,
        }
        return SMCResult(
            answer_text=particle_texts[chosen],
            answer_tokens=particle_tokens[chosen],
            particle_texts=particle_texts,
            particle_tokens=particle_tokens,
            terminal_weights=masses.tolist(),
            diagnostics=diagnostics,
        )
