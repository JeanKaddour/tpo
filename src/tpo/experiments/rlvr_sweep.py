"""Sequence-length sweep for the terminal-reward transformer experiment."""

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .._typing import ResultsByLength, ScopedResultsByLength, savez_dict
from ..config import TransformerRlvrConfig, coerce_config
from ..tracking import CurveReport, ExperimentReport
from ._trial import run_trial

SEQUENCE_LENGTHS = (7, 8, 9, 10)
ALGORITHMS = ("ppo", "dg", "tpo", "grpo")
GROUPED_ALGOS = {"tpo", "grpo", "grpo_no_kl", "group_pg", "tpo_no_anchor"}
ALGO_STYLES = {
    "tpo": {"label": "TPO", "color": "#2c7bb6", "ls": "-", "lw": 2.0},
    "grpo": {"label": "GRPO", "color": "#fdae61", "ls": "-", "lw": 1.6},
    "grpo_no_kl": {"label": "GRPO (no KL)", "color": "#018571", "ls": ":", "lw": 1.3},
    "dg": {"label": "DG", "color": "#d7191c", "ls": "--", "lw": 1.6},
    "ppo": {"label": "PPO", "color": "#abd9e9", "ls": "--", "lw": 1.6},
}


def _effective_batch_size(config: TransformerRlvrConfig, algo: str, match: str) -> int:
    if match == "interactions" and algo not in GROUPED_ALGOS:
        return config.batch_size * config.k_candidates
    return config.batch_size


def _effective_learning_rate(
    config: TransformerRlvrConfig, algo: str, match: str
) -> float:
    if match == "interactions" and algo not in GROUPED_ALGOS:
        return config.learning_rate * (config.k_candidates**0.5)
    return config.learning_rate


def _run_match(
    config: TransformerRlvrConfig,
    algorithms: tuple[str, ...],
    sequence_lengths: tuple[int, ...],
    match: str,
) -> dict[int, dict[str, np.ndarray]]:
    all_results: dict[int, dict[str, np.ndarray]] = {}
    for sequence_length in sequence_lengths:
        grouped = tuple(algo for algo in algorithms if algo in GROUPED_ALGOS)
        other = tuple(algo for algo in algorithms if algo not in GROUPED_ALGOS)
        results: dict[str, np.ndarray] = {}

        for algo_subset in (other, grouped):
            if not algo_subset:
                continue
            batch_size = _effective_batch_size(config, algo_subset[0], match)
            learning_rate = _effective_learning_rate(config, algo_subset[0], match)
            trial = run_trial(
                num_seeds=config.num_seeds,
                sequence_length=sequence_length,
                vocab_size=config.vocab_size,
                num_episodes=config.num_episodes,
                batch_size=batch_size,
                learning_rate=learning_rate,
                eta=config.eta,
                ppo_epsilon=config.ppo_epsilon,
                ppo_epochs=config.ppo_epochs,
                k_candidates=config.k_candidates,
                target_type="reverse_copy",
                reward_type="terminal",
                algorithms=algo_subset,
                label=f"{match}/H={sequence_length}: ",
                tpo_eta=config.tpo_eta,
            )
            results.update({algo: np.asarray(values) for algo, values in trial.items()})

        all_results[sequence_length] = results
    return all_results


def _plot_match_grid(
    all_results: dict[int, dict[str, np.ndarray]],
    sequence_lengths: tuple[int, ...],
    figure_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(sequence_lengths), figsize=(14, 3.8), sharey=True)
    if len(sequence_lengths) == 1:
        axes = [axes]

    for index, sequence_length in enumerate(sequence_lengths):
        ax = axes[index]
        results = all_results[sequence_length]
        episodes = np.arange(next(iter(results.values())).shape[1])
        for algo, style in ALGO_STYLES.items():
            if algo not in results:
                continue
            errors = results[algo]
            mean = errors.mean(axis=0)
            se = errors.std(axis=0) / np.sqrt(errors.shape[0])
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
        ax.set_title(f"H = {sequence_length}")
        ax.set_xlabel("Episode")
        ax.set_ylim(bottom=0.0)
        if index == 0:
            ax.set_ylabel("Exact-match error")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 1.04),
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=1.0)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_grid(
    prompt_results: dict[int, dict[str, np.ndarray]],
    interaction_results: dict[int, dict[str, np.ndarray]],
    sequence_lengths: tuple[int, ...],
    figure_path: Path,
) -> None:
    fig, axes = plt.subplots(
        2, len(sequence_lengths), figsize=(14, 6.0), sharey="row", sharex=True
    )
    rows = (
        ("Prompt-matched", prompt_results),
        ("Interaction-matched", interaction_results),
    )

    for row_index, (row_label, all_results) in enumerate(rows):
        for col_index, sequence_length in enumerate(sequence_lengths):
            ax = axes[row_index, col_index]
            results = all_results[sequence_length]
            episodes = np.arange(next(iter(results.values())).shape[1])
            for algo, style in ALGO_STYLES.items():
                if algo not in results:
                    continue
                errors = results[algo]
                mean = errors.mean(axis=0)
                se = errors.std(axis=0) / np.sqrt(errors.shape[0])
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
            if row_index == 0:
                ax.set_title(f"H = {sequence_length}")
            if row_index == 1:
                ax.set_xlabel("Episode")
            if col_index == 0:
                ax.set_ylabel(f"{row_label}\nExact-match error")
            ax.set_ylim(bottom=0.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=1.0, h_pad=1.0)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_rlvr_sweep(
    config: TransformerRlvrConfig | None = None,
    /,
    algorithms: tuple[str, ...] | None = None,
    sequence_lengths: tuple[int, ...] = SEQUENCE_LENGTHS,
    match: str = "both",
    verbose: bool = True,
    **overrides: object,
) -> ExperimentReport:
    """Run the terminal-reward sweep across sequence lengths."""

    config = coerce_config(config, TransformerRlvrConfig, overrides)
    algo_list = ALGORITHMS if algorithms is None else tuple(algorithms)
    if match not in {"prompts", "interactions", "both"}:
        raise ValueError(f"Unknown match mode: {match}")

    print(
        "Running terminal-reward RLVR sweep: "
        f"H={','.join(str(h) for h in sequence_lengths)}, "
        f"match={match}, {config.num_seeds} seeds, "
        f"{config.num_episodes} episodes, "
        f"algorithms={','.join(algo_list)}..."
    )

    prompt_results: ResultsByLength = (
        _run_match(config, algo_list, sequence_lengths, "prompts")
        if match in {"prompts", "both"}
        else {}
    )
    interaction_results: ResultsByLength = (
        _run_match(config, algo_list, sequence_lengths, "interactions")
        if match in {"interactions", "both"}
        else {}
    )

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: list[Path] = []
    if match == "both":
        raw_path = save_dir / "rlvr_sweep_combined.npz"
        savez_dict(
            raw_path,
            {
                f"prompts_h{h}_{algo}": values
                for h, results in prompt_results.items()
                for algo, values in results.items()
            }
            | {
                f"interactions_h{h}_{algo}": values
                for h, results in interaction_results.items()
                for algo, values in results.items()
            },
        )
        figure_path = save_dir / "rlvr_sweep_combined.png"
        _plot_combined_grid(
            prompt_results, interaction_results, sequence_lengths, figure_path
        )
        scoped_results: ScopedResultsByLength = {
            "prompts": prompt_results,
            "interactions": interaction_results,
        }
        raw_errors: ScopedResultsByLength | ResultsByLength = scoped_results
    else:
        label = match
        raw_path = save_dir / f"rlvr_sweep_{label}.npz"
        selected_results = prompt_results if match == "prompts" else interaction_results
        savez_dict(
            raw_path,
            {
                f"h{h}_{algo}": values
                for h, results in selected_results.items()
                for algo, values in results.items()
            },
        )
        figure_path = save_dir / f"rlvr_sweep_{label}.png"
        _plot_match_grid(selected_results, sequence_lengths, figure_path)
        raw_errors = selected_results
    artifact_paths.extend([figure_path, raw_path])

    print(f"  Saved {figure_path}")
    print(f"  Saved {raw_path}")

    summary: dict[str, float] = {}
    curves: list[CurveReport] = []
    if match == "both":
        for scope, results_by_h in scoped_results.items():
            for sequence_length in sequence_lengths:
                results = results_by_h[sequence_length]
                episodes = np.arange(next(iter(results.values())).shape[1])
                series: dict[str, np.ndarray] = {}
                for algo in algo_list:
                    if algo not in results:
                        continue
                    errors = results[algo]
                    mean = errors.mean(axis=0)
                    se = errors.std(axis=0) / np.sqrt(errors.shape[0])
                    summary[f"{scope}/h{sequence_length}/{algo}/final_error"] = float(
                        mean[-1]
                    )
                    series[f"{algo}_mean"] = mean
                    series[f"{algo}_se"] = se
                curves.append(
                    CurveReport(
                        name=f"{scope}_h{sequence_length}",
                        title=f"{scope} RLVR sweep (H={sequence_length})",
                        x_name="episode",
                        x_values=episodes,
                        series=series,
                        plot_series=tuple(
                            f"{algo}_mean"
                            for algo in algo_list
                            if f"{algo}_mean" in series
                        ),
                    )
                )
    else:
        for sequence_length in sequence_lengths:
            results = selected_results[sequence_length]
            episodes = np.arange(next(iter(results.values())).shape[1])
            series = {}
            for algo in algo_list:
                if algo not in results:
                    continue
                errors = results[algo]
                mean = errors.mean(axis=0)
                se = errors.std(axis=0) / np.sqrt(errors.shape[0])
                summary[f"h{sequence_length}/{algo}/final_error"] = float(mean[-1])
                series[f"{algo}_mean"] = mean
                series[f"{algo}_se"] = se
            curves.append(
                CurveReport(
                    name=f"rlvr_h{sequence_length}",
                    title=f"RLVR sweep (H={sequence_length})",
                    x_name="episode",
                    x_values=episodes,
                    series=series,
                    plot_series=tuple(
                        f"{algo}_mean" for algo in algo_list if f"{algo}_mean" in series
                    ),
                )
            )

    if verbose:
        display_results = prompt_results if match == "both" else selected_results
        for sequence_length in sequence_lengths:
            print(f"\n  RLVR H={sequence_length} metrics:")
            print(f"  {'algo':12s} {'final':>8s} {'best':>8s} {'steps→5%':>10s}")
            print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 10}")
            for algo in algo_list:
                if algo not in display_results[sequence_length]:
                    continue
                mean = display_results[sequence_length][algo].mean(axis=0)
                below = np.where(mean < 0.05)[0]
                step = str(int(below[0])) if len(below) > 0 else "never"
                print(f"  {algo:12s} {mean[-1]:8.4f} {mean.min():8.4f} {step:>10s}")

    return ExperimentReport(
        name="rlvr_sweep",
        config={
            **asdict(config),
            "sequence_lengths": list(sequence_lengths),
            "match": match,
        },
        summary=summary,
        curves=tuple(curves),
        artifact_paths=tuple(artifact_paths),
        raw_errors=raw_errors,
    )
