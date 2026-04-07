"""Regenerate the README sparse-reward token-reversal asset from the TPO package.

Usage:
    uv run python scripts/plot_token_reversal_sparse_reward.py
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from tpo.config import TransformerRlvrConfig
from tpo.experiments.transformer_rlvr import run_transformer_rlvr
from tpo.runtime import bootstrap_runtime

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
RESULTS_PATH = ASSET_DIR / "token_reversal_sparse_reward_results.npz"
PNG_PATH = ASSET_DIR / "token_reversal_sparse_reward.png"
PDF_PATH = ASSET_DIR / "token_reversal_sparse_reward.pdf"

ALGORITHMS = ("tpo", "grpo", "dg")
STYLES = {
    "tpo": {"label": "TPO", "color": "#2C7BB6", "linestyle": "-", "linewidth": 3.6},
    "grpo": {"label": "GRPO", "color": "#F4A259", "linestyle": "-", "linewidth": 2.8},
    "dg": {"label": "DG", "color": "#D62728", "linestyle": "--", "linewidth": 2.8},
}
SE_ALPHA = 0.12
SMOOTH_WINDOW = 40


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Apply left-padded moving-average smoothing."""

    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    bootstrap_runtime()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tpo-rlvr-readme-") as tmpdir:
        report = run_transformer_rlvr(
            TransformerRlvrConfig(
                num_seeds=20,
                sequence_length=10,
                vocab_size=2,
                num_episodes=2_000,
                batch_size=100,
                k_candidates=8,
                save_dir=tmpdir,
            ),
            algorithms=ALGORITHMS,
            match="prompts",
            verbose=False,
        )

    errors = {name: np.asarray(report.raw_errors[name]) for name in ALGORITHMS}
    episodes = np.arange(errors["tpo"].shape[1], dtype=np.int32)
    np.savez(
        RESULTS_PATH,
        episodes=episodes,
        num_seeds=np.int32(errors["tpo"].shape[0]),
        num_episodes=np.int32(2_000),
        sequence_length=np.int32(10),
        match="prompts",
        **{f"{name}_errors": values for name, values in errors.items()},
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 17,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 18,
            "axes.linewidth": 1.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for name in ALGORITHMS:
        arr = errors[name]
        mean = smooth(arr.mean(axis=0), SMOOTH_WINDOW)
        se = smooth(arr.std(axis=0, ddof=0) / np.sqrt(arr.shape[0]), SMOOTH_WINDOW)
        style = STYLES[name]
        ax.plot(
            episodes,
            mean,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
        ax.fill_between(
            episodes,
            np.maximum(mean - se, 0.0),
            mean + se,
            color=style["color"],
            alpha=SE_ALPHA,
        )

    ax.set_title("Token reversal (sparse reward)", pad=16)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Exact-match error")
    ax.set_xlim(0, int(episodes[-1]))
    ax.set_ylim(0.0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=8, width=1.8, pad=8)

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#B5B5B5",
    )
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_facecolor("white")

    fig.subplots_adjust(bottom=0.30, top=0.88, left=0.15, right=0.98)
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    print(f"Saved {RESULTS_PATH}")
    print(f"Saved {PNG_PATH}")
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    main()
