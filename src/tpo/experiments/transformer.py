"""Dense-reward reverse-copy transformer experiment."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..config import TransformerConfig, coerce_config
from ..tracking import CurveReport, ExperimentReport, log_per_algorithm_runs
from ._trial import run_trial

ALGORITHMS = ("ppo", "dg", "tpo_token", "grpo_token")
ALGO_COLORS = {
    "ppo": "tab:blue",
    "dg": "tab:red",
    "tpo_token": "sienna",
    "grpo_token": "tab:olive",
}


def run_transformer(
    config: TransformerConfig | None = None,
    /,
    algorithms: tuple[str, ...] | None = None,
    verbose: bool = True,
    **overrides: object,
) -> ExperimentReport:
    """Run the dense-reward reverse-copy transformer experiment."""

    config = coerce_config(config, TransformerConfig, overrides)
    algo_list = ALGORITHMS if algorithms is None else tuple(algorithms)
    num_seeds = config.num_seeds
    num_episodes = config.num_episodes

    print(
        "Running dense-reward transformer: "
        f"{num_seeds} seeds, {num_episodes} episodes, "
        f"H={config.sequence_length}, V={config.vocab_size}, "
        f"B={config.batch_size}, K={config.k_candidates}, "
        f"algorithms={','.join(algo_list)}..."
    )

    results = run_trial(
        num_seeds=config.num_seeds,
        sequence_length=config.sequence_length,
        vocab_size=config.vocab_size,
        num_episodes=config.num_episodes,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        eta=config.eta,
        ppo_epsilon=config.ppo_epsilon,
        ppo_epochs=config.ppo_epochs,
        k_candidates=config.k_candidates,
        target_type="reverse_copy",
        reward_type="bag_of_tokens",
        algorithms=algo_list,
        tpo_eta=config.tpo_eta,
    )

    episodes = np.arange(num_episodes)
    log_per_algorithm_runs(
        "transformer",
        config,
        [(algo, {"sequence_error": np.asarray(results[algo])}) for algo in algo_list],
        x_name="episode",
        x_values=episodes,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    summary: dict[str, float] = {}
    series: dict[str, np.ndarray] = {}
    for algo_name in algo_list:
        errors = np.asarray(results[algo_name])
        mean = errors.mean(axis=0)
        se = errors.std(axis=0) / np.sqrt(num_seeds)
        summary[f"{algo_name}/final_error"] = float(mean[-1])
        series[f"{algo_name}_mean"] = mean
        series[f"{algo_name}_se"] = se
        ax.semilogy(
            episodes, mean, color=ALGO_COLORS.get(algo_name, "gray"), label=algo_name
        )
        ax.fill_between(
            episodes,
            np.maximum(mean - se, 1e-5),
            mean + se,
            color=ALGO_COLORS.get(algo_name, "gray"),
            alpha=0.15,
        )

    if verbose:
        print("\n  Dense transformer metrics:")
        print(f"  {'algo':12s} {'final':>8s} {'best':>8s} {'steps→1%':>10s}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 10}")
        for algo_name in algo_list:
            mean = series[f"{algo_name}_mean"]
            best = float(np.min(mean))
            below = np.where(mean < 0.01)[0]
            step = str(int(below[0])) if len(below) > 0 else "never"
            print(f"  {algo_name:12s} {mean[-1]:8.4f} {best:8.4f} {step:>10s}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Sequence error")
    ax.set_title(f"Reverse copy (V={config.vocab_size})")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0.005)
    fig.tight_layout()

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    figure_path = save_dir / f"transformer_v{config.vocab_size}.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {figure_path}")

    return ExperimentReport(
        name="transformer",
        config=asdict(config),
        summary=summary,
        curves=(
            CurveReport(
                name="sequence_error",
                title=f"Dense transformer (V={config.vocab_size})",
                x_name="episode",
                x_values=episodes,
                series=series,
                plot_series=tuple(f"{algo}_mean" for algo in algo_list),
            ),
        ),
        artifact_paths=(figure_path,),
        raw_errors={algo: np.asarray(results[algo]) for algo in algo_list},
    )
