"""Vocabulary sweep for the dense-reward transformer experiment."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..config import TransformerConfig, coerce_config
from ..tracking import CurveReport, ExperimentReport
from ._trial import run_trial

VOCAB_SIZES = (2, 4, 8, 16)
ALGORITHMS = ("ppo", "dg", "tpo_token", "grpo_token")
ALGO_STYLES = {
    "tpo_token": {"label": "TPO", "color": "#2c7bb6", "ls": "-", "lw": 2.4},
    "grpo_token": {"label": "GRPO", "color": "#fdae61", "ls": "-", "lw": 2.0},
    "dg": {"label": "DG", "color": "#d7191c", "ls": "--", "lw": 2.0},
    "ppo": {"label": "PPO", "color": "#abd9e9", "ls": "--", "lw": 2.0},
}


def run_vocab_sweep(
    config: TransformerConfig | None = None,
    /,
    algorithms: tuple[str, ...] | None = None,
    vocab_sizes: tuple[int, ...] = VOCAB_SIZES,
    verbose: bool = True,
    **overrides: object,
) -> ExperimentReport:
    """Run the dense-reward transformer sweep over vocabulary size."""

    config = coerce_config(config, TransformerConfig, overrides)
    algo_list = ALGORITHMS if algorithms is None else tuple(algorithms)
    all_results: dict[int, dict[str, np.ndarray]] = {}

    print(
        "Running dense transformer vocab sweep: "
        f"V={','.join(str(v) for v in vocab_sizes)}, "
        f"{config.num_seeds} seeds, {config.num_episodes} episodes, "
        f"algorithms={','.join(algo_list)}..."
    )

    for vocab_size in vocab_sizes:
        print(f"  V={vocab_size}...", flush=True)
        trial = run_trial(
            num_seeds=config.num_seeds,
            sequence_length=config.sequence_length,
            vocab_size=vocab_size,
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
            label=f"V={vocab_size}: ",
            tpo_eta=config.tpo_eta,
        )
        all_results[vocab_size] = {
            algo: np.asarray(values) for algo, values in trial.items()
        }

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    raw_path = save_dir / "transformer_vocab_sweep.npz"
    np.savez(
        raw_path,
        **{
            f"v{vocab_size}_{algo}": values
            for vocab_size, results in all_results.items()
            for algo, values in results.items()
        },
    )

    episodes = np.arange(config.num_episodes)
    fig, axes = plt.subplots(1, len(vocab_sizes), figsize=(14, 3.8), sharey=True)
    if len(vocab_sizes) == 1:
        axes = [axes]

    summary: dict[str, float] = {}
    curves: list[CurveReport] = []

    for index, vocab_size in enumerate(vocab_sizes):
        ax = axes[index]
        results = all_results[vocab_size]
        series: dict[str, np.ndarray] = {}
        for algo, style in ALGO_STYLES.items():
            if algo not in results:
                continue
            errors = results[algo]
            mean = errors.mean(axis=0)
            se = errors.std(axis=0) / np.sqrt(errors.shape[0])
            summary[f"v{vocab_size}/{algo}/final_error"] = float(mean[-1])
            series[f"{algo}_mean"] = mean
            series[f"{algo}_se"] = se
            ax.plot(
                episodes,
                mean,
                color=style["color"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                label=style["label"],
            )
            ax.fill_between(
                episodes,
                np.maximum(mean - se, 0.0),
                mean + se,
                color=style["color"],
                alpha=0.12,
            )
        ax.set_title(f"V = {vocab_size}")
        ax.set_xlabel("Episode")
        ax.set_ylim(bottom=0.0)
        if index == 0:
            ax.set_ylabel("Sequence error")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        curves.append(
            CurveReport(
                name=f"transformer_v{vocab_size}",
                title=f"Dense transformer vocabulary sweep (V={vocab_size})",
                x_name="episode",
                x_values=episodes,
                series=series,
                plot_series=tuple(
                    f"{algo}_mean" for algo in algo_list if f"{algo}_mean" in series
                ),
            )
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 1.04),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=1.0)

    figure_path = save_dir / "transformer_vocab_sweep.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {figure_path}")
    print(f"  Saved {raw_path}")

    if verbose:
        for vocab_size in vocab_sizes:
            print(f"\n  V={vocab_size} metrics:")
            print(f"  {'algo':12s} {'final':>8s} {'best':>8s} {'steps→1%':>10s}")
            print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 10}")
            for algo in algo_list:
                if algo not in all_results[vocab_size]:
                    continue
                mean = all_results[vocab_size][algo].mean(axis=0)
                below = np.where(mean < 0.01)[0]
                step = str(int(below[0])) if len(below) > 0 else "never"
                print(f"  {algo:12s} {mean[-1]:8.4f} {mean.min():8.4f} {step:>10s}")

    return ExperimentReport(
        name="vocab_sweep",
        config={**asdict(config), "vocab_sizes": list(vocab_sizes)},
        summary=summary,
        curves=tuple(curves),
        artifact_paths=(figure_path, raw_path),
        raw_errors=all_results,
    )
