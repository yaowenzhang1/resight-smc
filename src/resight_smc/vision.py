from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Region:
    """Represent a candidate visual region ``R_g`` in normalized image coordinates."""

    name: str
    x0: float
    y0: float
    x1: float
    y1: float

@dataclass
class Route:
    """Record a particle--region assignment selected by greedy routing."""

    particle: int
    island: int
    region_index: int


def region_bank() -> list[Region]:
    """Build the candidate region bank ``R={R_1,...,R_G}``."""

    regions: list[Region] = []
    for size in (2, 3):
        for row in range(size):
            for col in range(size):
                regions.append(
                    Region(
                        f"grid{size}_{row}_{col}",
                        col / size,
                        row / size,
                        (col + 1) / size,
                        (row + 1) / size,
                    )
                )
    for index in range(3):
        regions.append(Region(f"horizontal_{index}", 0.0, index / 3, 1.0, (index + 1) / 3))
        regions.append(Region(f"vertical_{index}", index / 3, 0.0, (index + 1) / 3, 1.0))
    return regions


def iou(left: Region, right: Region) -> float:
    """Compute IoU between two normalized regions for the overlap penalty."""

    width = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    height = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    intersection = width * height
    left_area = (left.x1 - left.x0) * (left.y1 - left.y0)
    right_area = (right.x1 - right.x0) * (right.y1 - right.y0)
    return intersection / (left_area + right_area - intersection) if intersection else 0.0


def region_relevance(
    visual_attention: torch.Tensor,
    grid: tuple[int, int, int],
    regions: Sequence[Region],
    area_exponent: float,
) -> torch.Tensor:
    """Aggregate relevance as ``A_g = sum_{a,i in R_g} xi_{i,a} / (N_h |R_g|^zeta)``."""

    temporal, height, width = grid
    y = (torch.arange(height, device=visual_attention.device) + 0.5) / height
    x = (torch.arange(width, device=visual_attention.device) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.flatten().repeat(temporal)
    yy = yy.flatten().repeat(temporal)
    scores: list[torch.Tensor] = []
    for region in regions:
        mask = (xx >= region.x0) & (xx < region.x1) & (yy >= region.y0) & (yy < region.y1)
        token_count = max(1, int(mask.sum().item()))
        score = visual_attention[:, :, mask].sum(dim=(1, 2))
        score = score / visual_attention.shape[1] / (token_count**area_exponent)
        scores.append(score)
    return torch.stack(scores, dim=1)


def region_attention_bias(
    grid: tuple[int, int, int],
    selected_regions: Sequence[Region],
    image_logit_bias: float,
    region_logit_bias: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build scout bias ``b_j(R_g)=lambda_I 1[V_I]+lambda_R 1[V_I(R_g)]``."""

    temporal, height, width = grid
    y = (torch.arange(height, device=device) + 0.5) / height
    x = (torch.arange(width, device=device) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.flatten().repeat(temporal)
    yy = yy.flatten().repeat(temporal)
    rows: list[torch.Tensor] = []
    for region in selected_regions:
        inside = (xx >= region.x0) & (xx < region.x1) & (yy >= region.y0) & (yy < region.y1)
        rows.append(
            torch.full_like(xx, image_logit_bias, dtype=torch.float32)
            + inside.float() * region_logit_bias
        )
    return torch.stack(rows, dim=0)


def select_routes(
    relevance: torch.Tensor,
    log_weights: torch.Tensor,
    island_log_z: torch.Tensor,
    done: torch.Tensor,
    *,
    num_islands: int,
    particles_per_island: int,
    scout_fraction: float,
    overlap_penalty: float,
    overlap_penalty_scope: str,
    regions: Sequence[Region],
) -> list[Route]:
    """Select particle--region pairs using utility, island quotas, and an IoU penalty.

    Each eligible pair's utility combines pooled SMC mass and normalized region
    relevance. All initially eligible pairs share one stabilized min--max scale.
    """

    active_by_island: list[list[int]] = []
    for island in range(num_islands):
        start = island * particles_per_island
        active_by_island.append(
            [index for index in range(start, start + particles_per_island) if not bool(done[index])]
        )
    scouts_per_island = math.ceil(scout_fraction * particles_per_island)
    island_budgets = [
        min(scouts_per_island, max(len(active) - 1, 0)) for active in active_by_island
    ]
    budget = sum(island_budgets)
    if budget == 0:
        return []

    island_prob = torch.softmax(island_log_z.float(), dim=0)
    particle_mass = torch.empty_like(log_weights, dtype=torch.float32)
    for island in range(num_islands):
        start = island * particles_per_island
        end = start + particles_per_island
        particle_mass[start:end] = island_prob[island] * torch.softmax(
            log_weights[start:end].float(), dim=0
        )
    normalized_relevance = relevance / relevance.sum(dim=1, keepdim=True).clamp_min(1e-12)
    utility = particle_mass[:, None] * normalized_relevance

    eligible_values: list[torch.Tensor] = []
    for active in active_by_island:
        if len(active) >= 2:
            eligible_values.append(utility[active].flatten())
    joined = torch.cat(eligible_values)
    utility = (utility - joined.min()) / (joined.max() - joined.min() + 1e-12)

    chosen: list[Route] = []
    chosen_particles: set[int] = set()
    chosen_per_island = [0] * num_islands
    for _ in range(budget):
        best: tuple[float, int, int] | None = None
        for island, active in enumerate(active_by_island):
            if chosen_per_island[island] >= island_budgets[island]:
                continue
            for particle in active:
                if particle in chosen_particles:
                    continue
                for region_index, region in enumerate(regions):
                    penalty = max(
                        (
                            iou(region, regions[item.region_index])
                            for item in chosen
                            if overlap_penalty_scope == "global" or item.island == island
                        ),
                        default=0.0,
                    )
                    value = (
                        float(utility[particle, region_index].item()) - overlap_penalty * penalty
                    )
                    candidate = (value, particle, region_index)
                    if best is None or candidate > best:
                        best = candidate
        assert best is not None
        value, particle, region_index = best
        island = particle // particles_per_island
        chosen.append(
            Route(
                particle=particle,
                island=island,
                region_index=region_index,
            )
        )
        chosen_particles.add(particle)
        chosen_per_island[island] += 1
    return chosen
