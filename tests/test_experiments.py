import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpo.config import (
    CausalAblationsConfig,
    TerminalRewardAblationsConfig,
    TransformerRlvrConfig,
    TransformerVariationsConfig,
)
from tpo.experiments import (
    causal_ablations,
    rlvr_sweep,
    terminal_reward_ablations,
    transformer_rlvr,
    transformer_variations,
)


def test_terminal_reward_ablations_use_tpo_eta_for_tpo_targets(tmp_path, monkeypatch):
    cfg = TerminalRewardAblationsConfig(
        num_seeds=1,
        sequence_length=1,
        vocab_size=2,
        num_episodes=1,
        batch_size=1,
        eta=7.0,
        ppo_epochs=1,
        k_candidates=2,
        tpo_eta=0.25,
        save_dir=str(tmp_path),
    )
    seen_etas: list[float] = []

    def fake_tpo_target(log_scores_old, episode_scores, eta):
        seen_etas.append(eta)
        return jnp.ones_like(log_scores_old) / log_scores_old.shape[1]

    monkeypatch.setattr(terminal_reward_ablations, "tpo_target", fake_tpo_target)
    run_fns = terminal_reward_ablations._build_ablation_fns(cfg, ("tpo",))

    with jax.disable_jit():
        results = run_fns["tpo"](jnp.arange(cfg.num_seeds))

    assert results.shape == (
        cfg.num_seeds,
        cfg.num_episodes,
        terminal_reward_ablations.NUM_DIAGS,
    )
    assert seen_etas
    assert set(seen_etas) == {cfg.tpo_eta}


@pytest.mark.parametrize(
    ("run_fn", "expected_calls"),
    [
        (causal_ablations.run_epoch_sweep, len(causal_ablations.EPOCH_VALUES) * 2),
        (causal_ablations.run_k_sweep, len(causal_ablations.K_VALUES) * 2),
        (causal_ablations.run_grpo_fix, 2),
    ],
)
def test_causal_ablations_forward_tpo_eta(
    tmp_path, monkeypatch, run_fn, expected_calls
):
    cfg = CausalAblationsConfig(
        num_seeds=2,
        sequence_length=2,
        vocab_size=2,
        num_episodes=3,
        batch_size=2,
        ppo_epochs=1,
        k_candidates=4,
        tpo_eta=0.125,
        save_dir=str(tmp_path),
    )
    calls: list[dict] = []

    def fake_run_trial(**kwargs):
        calls.append(kwargs)
        return {
            algo: np.zeros((cfg.num_seeds, cfg.num_episodes), dtype=np.float32)
            for algo in kwargs["algorithms"]
        }

    monkeypatch.setattr(causal_ablations, "run_trial", fake_run_trial)

    if run_fn is causal_ablations.run_grpo_fix:
        monkeypatch.setattr(
            causal_ablations,
            "_build_grpo_masked_fns",
            lambda cfg: (
                lambda seeds: np.zeros(
                    (len(seeds), cfg.num_episodes),
                    dtype=np.float32,
                )
            ),
        )

    _, path = run_fn(cfg)

    assert path.exists()
    assert len(calls) == expected_calls
    assert {call["tpo_eta"] for call in calls} == {cfg.tpo_eta}


def test_transformer_rlvr_forwards_tpo_eta(tmp_path, monkeypatch):
    cfg = TransformerRlvrConfig(
        num_seeds=2,
        sequence_length=2,
        vocab_size=2,
        num_episodes=3,
        batch_size=2,
        tpo_eta=0.125,
        save_dir=str(tmp_path),
    )
    calls: list[dict] = []

    def fake_run_trial(**kwargs):
        calls.append(kwargs)
        return {
            algo: np.zeros((cfg.num_seeds, cfg.num_episodes), dtype=np.float32)
            for algo in kwargs["algorithms"]
        }

    monkeypatch.setattr(transformer_rlvr, "run_trial", fake_run_trial)

    report = transformer_rlvr.run_transformer_rlvr(
        cfg, algorithms=("tpo",), match="prompts", verbose=False
    )

    assert report.artifact_paths[0].exists()
    assert len(calls) == 1
    assert calls[0]["tpo_eta"] == cfg.tpo_eta


def test_rlvr_sweep_interactions_only_skips_prompt_runs(tmp_path, monkeypatch):
    cfg = TransformerRlvrConfig(
        num_seeds=2,
        num_episodes=3,
        save_dir=str(tmp_path),
    )
    matches: list[str] = []

    def fake_run_match(config, algorithms, sequence_lengths, match):
        matches.append(match)
        return {
            sequence_length: {
                algo: np.zeros((config.num_seeds, config.num_episodes), dtype=np.float32)
                for algo in algorithms
            }
            for sequence_length in sequence_lengths
        }

    def fake_plot_match_grid(all_results, sequence_lengths, figure_path):
        figure_path.write_bytes(b"")

    monkeypatch.setattr(rlvr_sweep, "_run_match", fake_run_match)
    monkeypatch.setattr(rlvr_sweep, "_plot_match_grid", fake_plot_match_grid)

    report = rlvr_sweep.run_rlvr_sweep(
        cfg,
        algorithms=("ppo",),
        sequence_lengths=(3,),
        match="interactions",
        verbose=False,
    )

    assert matches == ["interactions"]
    assert report.artifact_paths[0].exists()
    assert report.artifact_paths[1].exists()


def test_transformer_variations_interactions_only_skips_prompt_collection(
    tmp_path, monkeypatch
):
    cfg = TransformerVariationsConfig(
        num_seeds=2,
        num_episodes=3,
        save_dir=str(tmp_path),
    )
    collected_matches: list[str] = []
    logged_matches: list[str] = []

    def fake_collect_results(config, algorithms, *, match):
        collected_matches.append(match)
        return {
            (target_type, reward_type): {
                algo: np.zeros((config.num_seeds, config.num_episodes), dtype=np.float32)
                for algo in algorithms
            }
            for target_type in transformer_variations.TARGET_TYPES
            for reward_type in transformer_variations.REWARD_TYPES
        }

    def fake_log_match_runs(config, algorithms, match, all_results):
        logged_matches.append(match)

    def fake_plot_single_match(config, algorithms, match, all_results, figure_path):
        figure_path.write_bytes(b"")

    monkeypatch.setattr(
        transformer_variations, "_collect_results", fake_collect_results
    )
    monkeypatch.setattr(transformer_variations, "_log_match_runs", fake_log_match_runs)
    monkeypatch.setattr(
        transformer_variations, "_plot_single_match", fake_plot_single_match
    )

    report = transformer_variations.run_transformer_variations(
        cfg,
        algorithms=("ppo",),
        match="interactions",
    )

    assert collected_matches == ["interactions"]
    assert logged_matches == ["interactions"]
    assert report.artifact_paths[0].exists()
    assert report.artifact_paths[1].exists()
