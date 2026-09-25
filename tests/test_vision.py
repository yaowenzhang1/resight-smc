import torch

from resight_smc.vision import (
    region_attention_bias,
    region_bank,
    region_relevance,
    select_routes,
)


def test_region_bank_and_attention_bias():
    regions = region_bank()
    assert len(regions) == 19
    bias = region_attention_bias(
        (1, 3, 3),
        [regions[0], regions[13]],
        image_logit_bias=0.5,
        region_logit_bias=1.0,
        device=torch.device("cpu"),
    )
    assert bias.shape == (2, 9)
    assert torch.equal(bias[0], torch.tensor([1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    assert torch.equal(bias[1], torch.tensor([1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))


def test_router_fills_each_island_quota_and_keeps_one_anchor():
    regions = region_bank()
    attention = torch.full((16, 2, 9), 1 / 9)
    relevance = region_relevance(attention, (1, 3, 3), regions, 0.5)
    routes = select_routes(
        relevance,
        torch.full((16,), -torch.log(torch.tensor(4.0))),
        torch.zeros(4),
        torch.zeros(16, dtype=torch.bool),
        num_islands=4,
        particles_per_island=4,
        scout_fraction=0.75,
        overlap_penalty=0.25,
        overlap_penalty_scope="global",
        regions=regions,
    )
    assert len(routes) == 12
    assert all(sum(route.island == island for route in routes) == 3 for island in range(4))


def test_overlap_penalty_is_local_to_each_island():
    regions = region_bank()[:2]
    relevance = torch.tensor([[0.9, 0.1]] * 4)
    routes = select_routes(
        relevance,
        torch.zeros(4),
        torch.zeros(2),
        torch.zeros(4, dtype=torch.bool),
        num_islands=2,
        particles_per_island=2,
        scout_fraction=0.5,
        overlap_penalty=2.0,
        overlap_penalty_scope="island",
        regions=regions,
    )
    assert len(routes) == 2
    assert {route.island for route in routes} == {0, 1}
    assert all(route.region_index == 0 for route in routes)


def test_global_overlap_penalty_discourages_cross_island_repetition():
    regions = region_bank()[:2]
    relevance = torch.tensor([[0.9, 0.1]] * 4)
    routes = select_routes(
        relevance,
        torch.zeros(4),
        torch.zeros(2),
        torch.zeros(4, dtype=torch.bool),
        num_islands=2,
        particles_per_island=2,
        scout_fraction=0.5,
        overlap_penalty=2.0,
        overlap_penalty_scope="global",
        regions=regions,
    )
    assert len(routes) == 2
    assert {route.island for route in routes} == {0, 1}
    assert {route.region_index for route in routes} == {0, 1}
