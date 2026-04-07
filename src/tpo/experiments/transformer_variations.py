"""Task-variation sweep for dense token-level transformer grouping."""

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..config import TransformerVariationsConfig, coerce_config
from ..tracking import CurveReport, ExperimentReport, log_per_algorithm_runs
from ._trial import run_trial

TARGET_TYPES = ("copy", "flip", "reverse_copy", "reverse_flip")
REWARD_TYPES = ("bag_of_tokens", "sequential")
ALGORITHMS = ("ppo", "dg", "tpo_token", "grpo_token")
SEQUENCE_GROUPED_ALGOS = {"tpo", "grpo", "grpo_no_kl"}

ALGO_COLORS = {
    "ppo": "tab:blue",
    "dg": "tab:red",
    "tpo": "tab:brown",
    "grpo": "tab:cyan",
    "grpo_no_kl": "teal",
    "tpo_token": "sienna",
    "grpo_token": "tab:olive",
    "grpo_token_no_kl": "darkolivegreen",
}
TARGET_LABELS = {
    "copy": "copy",
    "flip": "flip",
    "reverse_copy": "reverse copy",
    "reverse_flip": "reverse flip",
}
REWARD_LABELS = {
    "bag_of_tokens": "Bag of tokens",
    "sequential": "Sequential",
}


def _warn_prompt_matched_group_budget(
    algorithms: tuple[str, ...], *, match: str, k_candidates: int
) -> None:
    grouped = tuple(algo for algo in algorithms if algo in SEQUENCE_GROUPED_ALGOS)
    other = tuple(algo for algo in algorithms if algo not in SEQUENCE_GROUPED_ALGOS)
    if match != "prompts" or not grouped or not other:
        return
    print(
        "  Note: prompt-matched mode gives "
        f"{', '.join(grouped)} {k_candidates} full-sequence rollouts per prompt while "
        f"{', '.join(other)} follow a single behavior rollout."
    )


def _note_match_noop(algorithms: tuple[str, ...], *, match: str) -> None:
    if match != "interactions":
        return
    if any(algo in SEQUENCE_GROUPED_ALGOS for algo in algorithms):
        return
    print(
        "  Note: dense token-candidate grouped methods already match PPO/DG in "
        "environment interactions, so match='interactions' is a no-op here."
    )


def _run_variant(
    config: TransformerVariationsConfig,
    target_type: str,
    reward_type: str,
    algorithms: tuple[str, ...],
    *,
    match: str,
) -> dict[str, np.ndarray]:
    grouped_algos = tuple(algo for algo in algorithms if algo in SEQUENCE_GROUPED_ALGOS)
    other_algos = tuple(
        algo for algo in algorithms if algo not in SEQUENCE_GROUPED_ALGOS
    )

    if match == "interactions" and grouped_algos:
        other_bs = config.batch_size * config.k_candidates
        other_lr = config.learning_rate * (config.k_candidates**0.5)
    else:
        other_bs = config.batch_size
        other_lr = config.learning_rate

    def _run(
        algo_list: tuple[str, ...], batch_size: int, learning_rate: float
    ) -> dict[str, np.ndarray]:
        trial = run_trial(
            num_seeds=config.num_seeds,
            sequence_length=config.sequence_length,
            vocab_size=config.vocab_size,
            num_episodes=config.num_episodes,
            batch_size=batch_size,
            learning_rate=learning_rate,
            eta=config.eta,
            ppo_epsilon=config.ppo_epsilon,
            ppo_epochs=config.ppo_epochs,
            k_candidates=config.k_candidates,
            target_type=target_type,
            reward_type=reward_type,
            algorithms=algo_list,
            label=f"{match}/{target_type}/{reward_type}: ",
            tpo_eta=config.tpo_eta,
        )
        return {algo: np.asarray(values) for algo, values in trial.items()}

    results: dict[str, np.ndarray] = {}
    for algo_list, batch_size, learning_rate in (
        (other_algos, other_bs, other_lr),
        (grouped_algos, config.batch_size, config.learning_rate),
    ):
        if not algo_list:
            continue
        results.update(_run(algo_list, batch_size, learning_rate))
    return results


def _collect_results(
    config: TransformerVariationsConfig,
    algorithms: tuple[str, ...],
    *,
    match: str,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    all_results: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for target_type in TARGET_TYPES:
        for reward_type in REWARD_TYPES:
            all_results[(target_type, reward_type)] = _run_variant(
                config,
                target_type,
                reward_type,
                algorithms,
                match=match,
            )
    return all_results


def _log_match_runs(
    config: TransformerVariationsConfig,
    algorithms: tuple[str, ...],
    match: str,
    all_results: dict[tuple[str, str], dict[str, np.ndarray]],
) -> None:
    episodes = np.arange(config.num_episodes)
    for target_type in TARGET_TYPES:
        for reward_type in REWARD_TYPES:
            variant_results = all_results[(target_type, reward_type)]
            algo_entries = [
                (algo_name, {"token_error": np.asarray(variant_results[algo_name])})
                for algo_name in algorithms
                if algo_name in variant_results
            ]
            log_per_algorithm_runs(
                f"transformer_variations/{match}/{target_type}/{reward_type}",
                config,
                algo_entries,
                x_name="episode",
                x_values=episodes,
            )


def _plot_single_match(
    config: TransformerVariationsConfig,
    algorithms: tuple[str, ...],
    match: str,
    all_results: dict[tuple[str, str], dict[str, np.ndarray]],
    figure_path: Path,
) -> None:
    episodes = np.arange(config.num_episodes)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey="row")

    for col, target_type in enumerate(TARGET_TYPES):
        for row, reward_type in enumerate(REWARD_TYPES):
            ax = axes[row, col]
            variant_results = all_results[(target_type, reward_type)]
            for algo_name in algorithms:
                if algo_name not in variant_results:
                    continue
                errors = variant_results[algo_name]
                mean = errors.mean(axis=0)
                se = errors.std(axis=0) / np.sqrt(errors.shape[0])
                ax.semilogy(
                    episodes,
                    mean,
                    color=ALGO_COLORS.get(algo_name, "gray"),
                    label=algo_name,
                )
                ax.fill_between(
                    episodes,
                    np.maximum(mean - se, 1e-5),
                    mean + se,
                    color=ALGO_COLORS.get(algo_name, "gray"),
                    alpha=0.15,
                )
            if row == 0:
                ax.set_title(TARGET_LABELS[target_type])
            if col == 0:
                ax.set_ylabel(REWARD_LABELS[reward_type])
            if row == 1:
                ax.set_xlabel("Episode")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_matches(
    config: TransformerVariationsConfig,
    algorithms: tuple[str, ...],
    prompt_results: dict[tuple[str, str], dict[str, np.ndarray]],
    interaction_results: dict[tuple[str, str], dict[str, np.ndarray]],
    figure_path: Path,
) -> None:
    episodes = np.arange(config.num_episodes)
    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharey="row", sharex=True)
    row_configs = (
        ("Prompt-matched", "bag_of_tokens", prompt_results),
        ("Prompt-matched", "sequential", prompt_results),
        ("Interaction-matched", "bag_of_tokens", interaction_results),
        ("Interaction-matched", "sequential", interaction_results),
    )

    for row_index, (row_label, reward_type, result_set) in enumerate(row_configs):
        for col_index, target_type in enumerate(TARGET_TYPES):
            ax = axes[row_index, col_index]
            variant_results = result_set[(target_type, reward_type)]
            for algo_name in algorithms:
                if algo_name not in variant_results:
                    continue
                errors = variant_results[algo_name]
                mean = errors.mean(axis=0)
                se = errors.std(axis=0) / np.sqrt(errors.shape[0])
                ax.semilogy(
                    episodes,
                    mean,
                    color=ALGO_COLORS.get(algo_name, "gray"),
                    label=algo_name,
                )
                ax.fill_between(
                    episodes,
                    np.maximum(mean - se, 1e-5),
                    mean + se,
                    color=ALGO_COLORS.get(algo_name, "gray"),
                    alpha=0.15,
                )
            if row_index == 0:
                ax.set_title(TARGET_LABELS[target_type])
            if row_index == 3:
                ax.set_xlabel("Episode")
            if col_index == 0:
                ax.set_ylabel(f"{row_label}\n{REWARD_LABELS[reward_type]}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_transformer_variations(
    config: TransformerVariationsConfig | None = None,
    /,
    algorithms: tuple[str, ...] | None = None,
    match: str = "both",
    **overrides: object,
) -> ExperimentReport:
    """Run the transformer variation sweep."""

    config = coerce_config(config, TransformerVariationsConfig, overrides)
    algo_list = ALGORITHMS if algorithms is None else tuple(algorithms)
    if match not in {"prompts", "interactions", "both"}:
        raise ValueError(f"Unknown match mode: {match}")

    print(
        f"Running Transformer Task Variations: "
        f"{len(TARGET_TYPES)}×{len(REWARD_TYPES)} variants, "
        f"{len(algo_list)} algorithms, "
        f"{config.num_seeds} seeds, {config.num_episodes} episodes, "
        f"match={match}..."
    )

    prompt_results = {}
    if match in {"prompts", "both"}:
        prompt_results = _collect_results(config, algo_list, match="prompts")
        _log_match_runs(config, algo_list, "prompts", prompt_results)
        _warn_prompt_matched_group_budget(
            algo_list, match="prompts", k_candidates=config.k_candidates
        )

    interaction_results = {}
    if match in {"interactions", "both"}:
        interaction_results = _collect_results(config, algo_list, match="interactions")
        _log_match_runs(config, algo_list, "interactions", interaction_results)
        _note_match_noop(algo_list, match="interactions")

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if match == "both":
        figure_path = save_dir / "transformer_variations_combined.png"
        raw_path = save_dir / "transformer_variations_combined.npz"
        _plot_combined_matches(
            config, algo_list, prompt_results, interaction_results, figure_path
        )
        np.savez(
            raw_path,
            **{
                f"prompts_{target}_{reward}_{algo}": values
                for (target, reward), payload in prompt_results.items()
                for algo, values in payload.items()
            },
            **{
                f"interactions_{target}_{reward}_{algo}": values
                for (target, reward), payload in interaction_results.items()
                for algo, values in payload.items()
            },
        )
        result_sets = {"prompts": prompt_results, "interactions": interaction_results}
    else:
        result_set = prompt_results if match == "prompts" else interaction_results
        figure_path = save_dir / f"transformer_variations_{match}.png"
        raw_path = save_dir / f"transformer_variations_{match}.npz"
        _plot_single_match(config, algo_list, match, result_set, figure_path)
        np.savez(
            raw_path,
            **{
                f"{target}_{reward}_{algo}": values
                for (target, reward), payload in result_set.items()
                for algo, values in payload.items()
            },
        )
        result_sets = {match: result_set}

    print(f"  Saved {figure_path}")
    print(f"  Saved {raw_path}")

    summary: dict[str, float] = {}
    curves: list[CurveReport] = []
    episodes = np.arange(config.num_episodes)
    for scope, all_results in result_sets.items():
        for (target_type, reward_type), variant_results in all_results.items():
            prefix = f"{scope}/{target_type}/{reward_type}"
            series: dict[str, np.ndarray] = {}
            for algo_name in algo_list:
                if algo_name not in variant_results:
                    continue
                errors = variant_results[algo_name]
                mean = errors.mean(axis=0)
                se = errors.std(axis=0) / np.sqrt(errors.shape[0])
                summary[f"{prefix}/{algo_name}/final_error"] = float(mean[-1])
                series[f"{algo_name}_mean"] = mean
                series[f"{algo_name}_se"] = se
            curves.append(
                CurveReport(
                    name=f"{scope}_{target_type}_{reward_type}",
                    title=f"{scope}: {TARGET_LABELS[target_type]} / {REWARD_LABELS[reward_type]}",
                    x_name="episode",
                    x_values=episodes,
                    series=series,
                    plot_series=tuple(
                        f"{algo}_mean" for algo in algo_list if f"{algo}_mean" in series
                    ),
                )
            )

    return ExperimentReport(
        name="transformer_variations",
        config={**asdict(config), "match": match},
        summary=summary,
        curves=tuple(curves),
        artifact_paths=(figure_path, raw_path),
        raw_errors=result_sets,
    )
