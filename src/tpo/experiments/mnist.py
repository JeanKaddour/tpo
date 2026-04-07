"""MNIST contextual bandit experiment — Figure 2.

MNIST as one-step bandit: given image X, sample A ∈ {0..9}, get R = 𝟙{A=Y}.
Compare PG (REINFORCE), DG, and CE (supervised cross-entropy).

Matches the official implementation: two-layer MLP (50,50), expected baseline
b = Σ_a π(a)², test-set evaluation every eval_every steps, independent
batch sampling per seed.
"""

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax

from ..algorithms import (
    classification_dg_loss,
    classification_grpo_loss,
    classification_group_pg_loss,
    classification_pg_loss,
    classification_tpo_loss,
)
from ..config import MnistConfig, coerce_config
from ..models import PolicyMLP
from ..tracking import CurveReport, ExperimentReport, log_per_algorithm_runs


@lru_cache(maxsize=1)
def _load_mnist():
    """Load MNIST train and test splits, return flat float32 arrays."""
    import gzip
    import struct
    import urllib.request
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "mnist"
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    def download(name):
        path = cache_dir / name
        if not path.exists():
            print(f"    Downloading {name}...")
            urllib.request.urlretrieve(base_url + name, path)
        return path

    def load_images(path):
        with gzip.open(path, "rb") as f:
            _, num, rows, cols = struct.unpack(">4I", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        return data.astype(np.float32) / 255.0

    def load_labels(path):
        with gzip.open(path, "rb") as f:
            _, num = struct.unpack(">2I", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.astype(np.int32)

    train_images = jnp.array(load_images(download(files["train_images"])))
    train_labels = jnp.array(load_labels(download(files["train_labels"])))
    test_images = jnp.array(load_images(download(files["test_images"])))
    test_labels = jnp.array(load_labels(download(files["test_labels"])))
    return train_images, train_labels, test_images, test_labels


def _validate_eval_schedule(num_steps: int, eval_every: int) -> int:
    """Return the number of eval blocks or raise on an invalid schedule."""
    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {eval_every}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if num_steps % eval_every != 0:
        raise ValueError(
            "MNIST expects num_steps to be divisible by eval_every, "
            f"got num_steps={num_steps}, eval_every={eval_every}"
        )
    return num_steps // eval_every


# Keep the main paper comparison as the default public surface; the other
# algorithms remain available for appendix diagnostics and ad hoc runs.
DEFAULT_ALGORITHMS = (
    "ce",
    "pg",
    "dg",
    "tpo",
)
AVAILABLE_ALGORITHMS = DEFAULT_ALGORITHMS + (
    "grpo",
    "group_pg",
)


def _run_mnist_algos(
    algo_names: list[str],
    run_fns: dict[str, Callable],
    keys,
) -> dict[str, jax.Array]:
    """Run MNIST algorithms and return {algo_name: errors (num_seeds, num_blocks)}."""
    all_errors = {}
    for name in algo_names:
        print(f"  Training {name}...")
        all_errors[name] = run_fns[name](keys)
    return all_errors


@lru_cache(maxsize=None)
def _build_mnist_run_fns(
    algo_names: tuple[str, ...],
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    eta: float,
    eval_every: int,
) -> dict[str, Callable]:
    """Build cached, compiled MNIST runners for a fixed training setup."""
    num_blocks = _validate_eval_schedule(num_steps, eval_every)
    train_images, train_labels, test_images, test_labels = _load_mnist()
    num_train = train_images.shape[0]

    model = PolicyMLP()
    optimizer = optax.adam(learning_rate)
    init_inputs = jnp.ones((1, 784), dtype=train_images.dtype)

    def pg_loss_fn(logits, batch_labels, key):
        return classification_pg_loss(logits, batch_labels, key)

    def dg_loss_fn(logits, batch_labels, key):
        return classification_dg_loss(logits, batch_labels, key, eta=eta)

    def ce_loss_fn(logits, batch_labels, key):
        del key
        log_probs = jax.nn.log_softmax(logits)
        return -jnp.mean(jnp.take_along_axis(log_probs, batch_labels[:, None], axis=1))

    def tpo_loss_fn(logits, batch_labels, key):
        return classification_tpo_loss(logits, batch_labels, key, eta=eta)

    def group_pg_loss_fn(logits, batch_labels, key):
        return classification_group_pg_loss(logits, batch_labels, key)

    def grpo_loss_fn(logits, batch_labels, key):
        return classification_grpo_loss(logits, batch_labels, key)

    _all_loss_fns = {
        "pg": pg_loss_fn,
        "dg": dg_loss_fn,
        "ce": ce_loss_fn,
        "tpo": tpo_loss_fn,
        "group_pg": group_pg_loss_fn,
        "grpo": grpo_loss_fn,
    }

    loss_fns = {}
    for name in algo_names:
        if name not in _all_loss_fns:
            raise ValueError(
                f"Unknown MNIST algorithm: {name}. Available: {list(_all_loss_fns)}"
            )
        loss_fns[name] = _all_loss_fns[name]

    def _make_train_fn(loss_fn):
        def train_one(key):
            key, init_key = jr.split(key)
            params = model.init(init_key, init_inputs)
            opt_state = optimizer.init(params)

            def eval_block(carry, block_keys):
                params, opt_state = carry

                def step_fn(carry, step_key):
                    params, opt_state = carry
                    bk, lk = jr.split(step_key)
                    idx = jr.randint(bk, (batch_size,), 0, num_train)
                    batch_imgs = train_images[idx]
                    batch_lbls = train_labels[idx]

                    def full_loss(p):
                        logits = model.apply(p, batch_imgs)
                        return loss_fn(logits, batch_lbls, lk)

                    grads = jax.grad(full_loss)(params)
                    updates, new_opt = optimizer.update(grads, opt_state, params)
                    return (optax.apply_updates(params, updates), new_opt), None

                (params, opt_state), _ = jax.lax.scan(
                    step_fn, (params, opt_state), block_keys
                )
                test_logits = model.apply(params, test_images)
                test_err = 1.0 - jnp.mean(jnp.argmax(test_logits, -1) == test_labels)
                return (params, opt_state), test_err

            all_keys = jr.split(key, num_steps).reshape(num_blocks, eval_every, 2)
            _, test_errors = jax.lax.scan(eval_block, (params, opt_state), all_keys)
            return test_errors

        return train_one

    return {
        name: jax.jit(jax.vmap(_make_train_fn(loss_fns[name]))) for name in algo_names
    }


def build_mnist_algos(config: MnistConfig, algo_names: list[str]):
    """Build compiled MNIST runners for the requested algorithms.

    Returns (run_fns, keys) — everything needed to call _run_mnist_algos
    without any plotting or reporting side effects.
    """
    requested_algos = tuple(dict.fromkeys(algo_names))
    run_fns = _build_mnist_run_fns(
        requested_algos,
        config.num_steps,
        config.batch_size,
        config.learning_rate,
        config.eta,
        config.eval_every,
    )
    keys = jr.split(jr.PRNGKey(config.seed), config.num_seeds)
    return run_fns, keys


def run_mnist(
    config: MnistConfig | None = None,
    /,
    algorithms: tuple[str, ...] | None = None,
    **overrides: object,
):
    """Run MNIST contextual bandit (Figure 2)."""
    config = coerce_config(config, MnistConfig, overrides)
    num_seeds = config.num_seeds
    num_steps = config.num_steps
    eval_every = config.eval_every
    save_dir = config.save_dir
    num_blocks = _validate_eval_schedule(num_steps, eval_every)

    print(f"Running MNIST bandit: {num_seeds} seeds, {num_steps} steps...")

    if algorithms is not None:
        algo_names = list(algorithms)
    else:
        algo_names = list(DEFAULT_ALGORITHMS)

    run_fns, keys = build_mnist_algos(config, algo_names)
    all_errors = _run_mnist_algos(algo_names, run_fns, keys)

    _BUILTIN_COLORS = {
        "pg": "tab:blue",
        "dg": "tab:red",
        "ce": "tab:gray",
        "tpo": "tab:brown",
        "grpo": "tab:green",
        "group_pg": "tab:orange",
    }
    _EXTRA_COLORS = ["tab:green", "tab:pink", "darkblue", "tab:olive"]

    # ---- Log per-algorithm wandb runs ----
    eval_steps = jnp.arange(num_blocks) * eval_every
    log_per_algorithm_runs(
        "mnist",
        config,
        [
            (name, {"classification_error": np.asarray(errs)})
            for name, errs in all_errors.items()
        ],
        x_name="episode",
        x_values=np.asarray(eval_steps),
    )

    # ---- Plot Figure 2 ----
    fig, ax = plt.subplots(figsize=(6, 4))
    summary = {}
    series = {}
    plot_series = []
    extra_idx = 0

    for name, errors in all_errors.items():
        mean = errors.mean(axis=0)
        se = errors.std(axis=0) / jnp.sqrt(num_seeds)
        if name in _BUILTIN_COLORS:
            color = _BUILTIN_COLORS[name]
        else:
            color = _EXTRA_COLORS[extra_idx % len(_EXTRA_COLORS)]
            extra_idx += 1
        ax.semilogy(eval_steps, mean, color=color, label=name.upper())
        ax.fill_between(
            eval_steps, jnp.maximum(mean - se, 1e-4), mean + se, color=color, alpha=0.2
        )
        summary[f"{name}/final_error"] = float(mean[-1])
        series[f"{name}_mean"] = mean
        series[f"{name}_se"] = se
        plot_series.append(f"{name}_mean")

    ax.set_xlabel("episode T")
    ax.set_ylabel("classification error")
    ax.legend()
    ax.set_title("Figure 2: MNIST classification error")
    fig.tight_layout()
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    figure_path = save_dir_path / "figure2_mnist.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {figure_path}")

    return ExperimentReport(
        name="mnist",
        config=asdict(config),
        summary=summary,
        curves=(
            CurveReport(
                name="classification_error",
                title="MNIST classification error",
                x_name="episode",
                x_values=eval_steps,
                series=series,
                plot_series=tuple(plot_series),
            ),
        ),
        artifact_paths=(figure_path,),
        raw_errors={name: np.asarray(errs) for name, errs in all_errors.items()},
    )
