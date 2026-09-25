from pathlib import Path

from resight_smc.config import load_config


def test_flat_cli_overrides_paper_defaults():
    config = Path(__file__).parents[1] / "configs" / "default.yaml"
    defaults = load_config(["--config", str(config)])
    assert defaults.dataset == "logicvista"
    assert defaults.seed == 0
    assert defaults.num_particles == 32
    assert defaults.num_islands == 4
    assert defaults.particles_per_island == 8
    assert defaults.scout_fraction == 0.25
    assert defaults.visual_checkpoint == 40
    assert defaults.alpha == 2.0
    assert defaults.gamma == 2.0
    assert defaults.image_attention_logit_bias == 0.6931471805599453
    assert defaults.region_attention_logit_bias == 1.3862943611198906
    assert defaults.attention_layer == -1
    assert defaults.area_exponent == 0.75
    assert defaults.overlap_penalty == 1.0
    assert defaults.overlap_penalty_scope == "global"
    assert defaults.top_p == 1.0
    assert defaults.top_k == 0

    cfg = load_config(
        [
            "--config",
            str(config),
            "--num-particles",
            "32",
            "--num-islands",
            "8",
            "--image-attention-logit-bias",
            "0.5",
            "--region-attention-logit-bias",
            "1.0",
            "--attention-layer",
            "-1",
            "--overlap-penalty-scope",
            "island",
        ]
    )
    assert cfg.particles_per_island == 4
    assert cfg.visual_checkpoint == 40
    assert cfg.image_attention_logit_bias == 0.5
    assert cfg.region_attention_logit_bias == 1.0
    assert cfg.attention_layer == -1.0
    assert cfg.overlap_penalty_scope == "island"


def test_global_power_smc_forces_one_island():
    config = Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = load_config(["--config", str(config), "--method", "power_smc", "--num-particles", "10"])
    assert cfg.num_islands == 1
    assert cfg.particles_per_island == 10
    assert cfg.gamma == 1.0


def test_answer_power_gamma_can_be_overridden_directly():
    config = Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = load_config(["--config", str(config), "--method", "island_smc", "--gamma", "2"])
    assert cfg.gamma == 2.0
