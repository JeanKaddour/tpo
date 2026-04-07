"""Regenerate the README MNIST contextual-bandit asset from the TPO package.

Usage:
    uv run python scripts/plot_mnist_contextual_bandit.py
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from tpo._typing import savez_dict
from tpo.config import MnistConfig
from tpo.experiments.mnist import run_mnist
from tpo.runtime import bootstrap_runtime

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
RESULTS_PATH = ASSET_DIR / "mnist_contextual_bandit_results.npz"
PNG_PATH = ASSET_DIR / "mnist_contextual_bandit.png"
PDF_PATH = ASSET_DIR / "mnist_contextual_bandit.pdf"

ALGORITHMS = ("tpo", "grpo", "dg")
STYLES = {
    "tpo": {"label": "TPO", "color": "#2C7BB6", "linestyle": "-", "linewidth": 4.0},
    "grpo": {"label": "GRPO", "color": "#F4A259", "linestyle": "-", "linewidth": 3.0},
    "dg": {"label": "DG", "color": "#D62728", "linestyle": "--", "linewidth": 3.0},
}
SE_ALPHA = 0.14


def main() -> None:
    bootstrap_runtime()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tpo-mnist-readme-") as tmpdir:
        report = run_mnist(
            MnistConfig(
                num_seeds=20,
                num_steps=10_000,
                eval_every=200,
                save_dir=tmpdir,
            ),
            algorithms=ALGORITHMS,
        )

    errors = {name: np.asarray(report.raw_errors[name]) for name in ALGORITHMS}
    steps = np.arange(errors["tpo"].shape[1], dtype=np.int32) * 200
    savez_dict(
        RESULTS_PATH,
        {
            "steps": steps,
            "num_seeds": np.int32(errors["tpo"].shape[0]),
            "num_steps": np.int32(10_000),
            "eval_every": np.int32(200),
            **{f"{name}_errors": values for name, values in errors.items()},
        },
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 18,
            "axes.titlesize": 28,
            "axes.labelsize": 26,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
            "axes.linewidth": 1.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for name in ALGORITHMS:
        arr = errors[name]
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=0) / np.sqrt(arr.shape[0])
        style = STYLES[name]
        ax.plot(
            steps,
            mean,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
        ax.fill_between(
            steps,
            mean - se,
            mean + se,
            color=style["color"],
            alpha=SE_ALPHA,
        )

    ax.set_title("MNIST contextual bandit", pad=20)
    ax.set_xlabel("Step")
    ax.set_ylabel("Classification error")
    ax.set_xlim(0, int(steps[-1]))
    ax.set_ylim(0.02, 0.145)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=8, width=1.8, pad=8)

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#B5B5B5",
    )
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    print(f"Saved {RESULTS_PATH}")
    print(f"Saved {PNG_PATH}")
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    main()
