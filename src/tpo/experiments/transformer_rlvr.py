"""Terminal-reward (RLVR-style) transformer experiment."""

import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..config import TransformerRlvrConfig, coerce_config
from ..tracking import CurveReport, ExperimentReport
from ._trial import run_trial

ALGORITHMS = ("ppo", "dg", "tpo", "grpo")
_GROUPED_ALGOS = {"tpo", "grpo", "grpo_no_kl"}

ALGO_COLORS = {
    "ppo": "tab:blue",
    "dg": "tab:red",
    "tpo": "tab:brown",
    "grpo": "tab:cyan",
    "grpo_no_kl": "teal",
    "tpo_token": "sienna",
    "grpo_token": "tab:olive",
}


def _warn_prompt_matched_group_budget(
    algorithms: tuple[str, ...],
    *,
    match: str,
    k_candidates: int,
) -> None:
    grouped = tuple(algo for algo in algorithms if algo in _GROUPED_ALGOS)
    other = tuple(algo for algo in algorithms if algo not in _GROUPED_ALGOS)
    if match != "prompts" or not grouped or not other:
        return
    print(
        "  Note: prompt-matched mode gives "
        f"{', '.join(grouped)} {k_candidates} rollouts per prompt while "
        f"{', '.join(other)} get 1. Use match='interactions' to equalize total rollout budget."
    )


def _cache_key(config, match: str):
    """Hash config (excluding save_dir) to key the cache file."""
    cfg = {k: v for k, v in asdict(config).items() if k != "save_dir"}
    cfg["match"] = match
    raw = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()[:12]


def _cache_path(save_dir, config, match: str):
    return Path(save_dir) / f".rlvr_cache_{_cache_key(config, match)}.pkl"


def _effective_batch_size(config: TransformerRlvrConfig, algo: str, match: str) -> int:
    if match == "interactions" and algo not in _GROUPED_ALGOS:
        return config.batch_size * config.k_candidates
    return config.batch_size


def _effective_learning_rate(
    config: TransformerRlvrConfig, algo: str, match: str
) -> float:
    if match == "interactions" and algo not in _GROUPED_ALGOS:
        return config.learning_rate * (config.k_candidates**0.5)
    return config.learning_rate


def _figure_filename(match: str, algorithms: tuple[str, ...]) -> str:
    if match == "prompts" and algorithms == ALGORITHMS:
        return "figure_transformer_rlvr.png"
    algo_slug = "-".join(algorithms)
    return f"figure_transformer_rlvr_{match}_{algo_slug}.png"


def _load_cache(path):
    if path.exists():
        try:
            with open(path, "rb") as f:
                cache = pickle.load(f)
            if not isinstance(cache, dict):
                return {}
            return {algo: np.asarray(values) for algo, values in cache.items()}
        except Exception:
            return {}
    return {}


def _save_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({algo: np.asarray(values) for algo, values in cache.items()}, f)


def _print_metrics_table(results, num_episodes, algos, interval=None):
    """Print a table of mean error at regular episode intervals."""
    if interval is None:
        interval = max(1, num_episodes // 10)
    checkpoints = list(range(interval, num_episodes, interval))
    if num_episodes - 1 not in checkpoints:
        checkpoints.append(num_episodes - 1)

    header = f"{'episode':>8}"
    for algo in algos:
        header += f"  {algo:>16}"
    print(f"\n  {header}")
    print(f"  {'─' * len(header)}")

    mean_errors = {algo: results[algo].mean(axis=0) for algo in algos}
    for ep in checkpoints:
        row = f"{ep + 1:>8}"
        for algo in algos:
            mean_err = float(mean_errors[algo][ep])
            row += f"  {mean_err:>16.4f}"
        print(f"  {row}")
    print()


def run_transformer_rlvr(
    config: TransformerRlvrConfig | None = None,
    /,
    algorithms: tuple[str, ...] | None = None,
    match: str = "prompts",
    verbose: bool = True,
    **overrides: object,
):
    """Run terminal-reward transformer experiment."""
    config = coerce_config(config, TransformerRlvrConfig, overrides)
    algo_list = ALGORITHMS if algorithms is None else tuple(algorithms)
    if not algo_list:
        raise ValueError("At least one algorithm must be provided.")
    if match not in {"prompts", "interactions"}:
        raise ValueError(f"Unknown match mode: {match}")

    num_seeds = config.num_seeds
    num_episodes = config.num_episodes
    save_dir = config.save_dir

    print(
        f"Running Transformer RLVR (terminal reward): "
        f"{num_seeds} seeds, {num_episodes} episodes, "
        f"H={config.sequence_length}, V={config.vocab_size}, "
        f"B={config.batch_size}, match={match}, "
        f"algorithms={','.join(algo_list)}..."
    )
    _warn_prompt_matched_group_budget(
        algo_list,
        match=match,
        k_candidates=config.k_candidates,
    )

    cache_path = _cache_path(save_dir, config, match)
    cache = _load_cache(cache_path)
    algos_cached = [algo for algo in algo_list if algo in cache]
    algos_to_run = [algo for algo in algo_list if algo not in cache]

    if algos_cached:
        print(f"  Cached ({len(algos_cached)}): {', '.join(algos_cached)}")
    if algos_to_run:
        grouped_algos = tuple(algo for algo in algos_to_run if algo in _GROUPED_ALGOS)
        other_algos = tuple(algo for algo in algos_to_run if algo not in _GROUPED_ALGOS)

        def _run(algo_subset: tuple[str, ...], batch_size: int, learning_rate: float):
            return run_trial(
                num_seeds=num_seeds,
                sequence_length=config.sequence_length,
                vocab_size=config.vocab_size,
                num_episodes=num_episodes,
                batch_size=batch_size,
                learning_rate=learning_rate,
                eta=config.eta,
                ppo_epsilon=config.ppo_epsilon,
                ppo_epochs=config.ppo_epochs,
                k_candidates=config.k_candidates,
                target_type="reverse_copy",
                reward_type="terminal",
                algorithms=algo_subset,
                tpo_eta=config.tpo_eta,
            )

        for algo_subset, batch_size, lr in (
            (
                other_algos,
                (
                    _effective_batch_size(config, other_algos[0], match)
                    if other_algos
                    else config.batch_size
                ),
                (
                    _effective_learning_rate(config, other_algos[0], match)
                    if other_algos
                    else config.learning_rate
                ),
            ),
            (grouped_algos, config.batch_size, config.learning_rate),
        ):
            if not algo_subset:
                continue
            print(
                f"  Computing ({len(algo_subset)}): {', '.join(algo_subset)} "
                f"(batch_size={batch_size}, lr={lr:.1e})"
            )
            new_results = _run(algo_subset, batch_size, lr)
            cache.update(
                {algo: np.asarray(values) for algo, values in new_results.items()}
            )
            _save_cache(cache_path, cache)

    results = {algo: np.asarray(cache[algo]) for algo in algo_list}

    if verbose:
        _print_metrics_table(results, num_episodes, algo_list)

    episodes = np.arange(num_episodes)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo_name in algo_list:
        color = ALGO_COLORS.get(algo_name, "gray")
        errors = results[algo_name]
        mean = errors.mean(axis=0)
        se = errors.std(axis=0) / np.sqrt(num_seeds)
        ax.semilogy(episodes, mean, color=color, label=algo_name)
        ax.fill_between(
            episodes,
            np.maximum(mean - se, 1e-5),
            mean + se,
            color=color,
            alpha=0.15,
        )

    ax.set_ylim(bottom=0.05)
    ax.set_xlabel("episode K")
    ax.set_ylabel("exact-match error")
    ax.set_title(f"Terminal reward (RLVR-style, {match})")
    ax.legend(fontsize=8)
    fig.tight_layout()

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    figure_path = save_dir_path / _figure_filename(match, algo_list)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {figure_path}")

    summary = {}
    series = {}
    for algo_name in algo_list:
        errors = results[algo_name]
        mean = errors.mean(axis=0)
        se = errors.std(axis=0) / np.sqrt(num_seeds)
        summary[f"{algo_name}/final_error"] = float(mean[-1])
        series[f"{algo_name}_mean"] = mean
        series[f"{algo_name}_se"] = se

    report_config = {
        **asdict(config),
        "algorithms": list(algo_list),
        "match": match,
        "effective_batch_sizes": {
            algo_name: _effective_batch_size(config, algo_name, match)
            for algo_name in algo_list
        },
    }

    return ExperimentReport(
        name="transformer_rlvr",
        config=report_config,
        summary=summary,
        curves=(
            CurveReport(
                name="exact_match_error",
                title="Terminal reward (RLVR-style) — exact-match error",
                x_name="episode",
                x_values=episodes,
                series=series,
                plot_series=tuple(f"{algo}_mean" for algo in algo_list),
            ),
        ),
        artifact_paths=(figure_path,),
        raw_errors={algo: np.asarray(values) for algo, values in results.items()},
    )
