"""MNIST mechanism diagnostics for concentration-binned logit updates."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
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
from ..tracking import CurveReport, ExperimentReport
from .mnist import _load_mnist, _validate_eval_schedule

MECHANISM_ALGORITHMS = ("pg", "dg", "tpo", "grpo", "group_pg")
CONCENTRATION_BINS = jnp.array([0.0, 0.25, 0.5, 0.75, 1.01], dtype=jnp.float32)
LAMBDA_TPO = jnp.exp(jnp.array(10.0 / 3.0, dtype=jnp.float32))
EPS = 1e-8


def _first_order_gain(
    probs: jax.Array,
    labels: jax.Array,
    update: jax.Array,
) -> jax.Array:
    one_hot = jax.nn.one_hot(labels, probs.shape[1], dtype=probs.dtype)
    correct_prob = jnp.take_along_axis(probs, labels[:, None], axis=1).squeeze(1)
    direction = one_hot - probs
    return correct_prob * jnp.sum(direction * update, axis=1)


def _scalar_gain(
    probs: jax.Array,
    labels: jax.Array,
    beta: jax.Array,
) -> jax.Array:
    one_hot = jax.nn.one_hot(labels, probs.shape[1], dtype=probs.dtype)
    correct_prob = jnp.take_along_axis(probs, labels[:, None], axis=1).squeeze(1)
    direction = one_hot - probs
    return correct_prob * beta * jnp.sum(direction * direction, axis=1)


def _wrong_class_concentration(probs: jax.Array, labels: jax.Array) -> jax.Array:
    one_hot = jax.nn.one_hot(labels, probs.shape[1], dtype=probs.dtype)
    correct_prob = jnp.take_along_axis(probs, labels[:, None], axis=1).squeeze(1)
    wrong_probs = jnp.where(one_hot > 0, -jnp.inf, probs)
    max_wrong = jnp.max(wrong_probs, axis=1)
    return max_wrong / jnp.maximum(1.0 - correct_prob, EPS)


def _bin_means(
    values: jax.Array,
    concentration: jax.Array,
    include_mask: jax.Array,
    bin_edges: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    num_bins = bin_edges.shape[0] - 1
    bin_index = jnp.clip(
        jnp.digitize(concentration, bin_edges[1:-1], right=False),
        0,
        num_bins - 1,
    )
    valid = include_mask & jnp.isfinite(values)
    one_hot = jax.nn.one_hot(bin_index, num_bins, dtype=values.dtype) * valid[:, None]
    counts = one_hot.sum(axis=0)
    totals = (one_hot * values[:, None]).sum(axis=0)
    means = jnp.where(counts > 0, totals / counts, jnp.nan)
    return means, counts


def _pg_diagnostic(probs: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    correct_prob = jnp.take_along_axis(probs, labels[:, None], axis=1).squeeze(1)
    one_hot = jax.nn.one_hot(labels, probs.shape[1], dtype=probs.dtype)
    direction = one_hot - probs
    update = correct_prob[:, None] * direction
    gain = _first_order_gain(probs, labels, update)
    return gain, gain


def _group_pg_diagnostic(
    probs: jax.Array, labels: jax.Array
) -> tuple[jax.Array, jax.Array]:
    correct_prob = jnp.take_along_axis(probs, labels[:, None], axis=1).squeeze(1)
    beta = 6.0 * correct_prob
    gain = _scalar_gain(probs, labels, beta)
    return gain, gain


def _dg_diagnostic(
    probs: jax.Array,
    labels: jax.Array,
    *,
    eta: float,
) -> tuple[jax.Array, jax.Array]:
    num_classes = probs.shape[1]
    one_hot = jax.nn.one_hot(labels, num_classes, dtype=probs.dtype)
    correct_prob = jnp.take_along_axis(probs, labels[:, None], axis=1).squeeze(1)
    direction = one_hot - probs
    baseline = jnp.sum(probs**2, axis=1)
    surprisal = -jnp.log(jnp.clip(probs, EPS))

    success_coeff = (
        correct_prob
        * (1.0 - baseline)
        * jax.nn.sigmoid(
            (1.0 - baseline) * (-jnp.log(jnp.clip(correct_prob, EPS))) / eta
        )
    )
    success = success_coeff[:, None] * direction

    wrong_alpha = (
        probs
        * baseline[:, None]
        * jax.nn.sigmoid((-baseline[:, None] * surprisal) / eta)
        * (1.0 - one_hot)
    )
    wrong = probs * wrong_alpha.sum(axis=1, keepdims=True) - wrong_alpha
    exact_update = success + wrong
    exact_gain = _first_order_gain(probs, labels, exact_update)

    q = (1.0 - correct_prob) / (num_classes - 1)
    baseline_sym = correct_prob**2 + (num_classes - 1) * (q**2)
    beta_sym = (
        correct_prob
        * (1.0 - baseline_sym)
        * jax.nn.sigmoid(
            (1.0 - baseline_sym) * (-jnp.log(jnp.clip(correct_prob, EPS))) / eta
        )
    )
    beta_sym += (
        correct_prob
        * baseline_sym
        * jax.nn.sigmoid((-baseline_sym * (-jnp.log(jnp.clip(q, EPS)))) / eta)
    )
    scalar_gain = _scalar_gain(probs, labels, beta_sym)
    return exact_gain, scalar_gain


def _tpo_diagnostic(probs: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    num_classes = probs.shape[1]
    one_hot = jax.nn.one_hot(labels, num_classes, dtype=probs.dtype)
    correct_prob = jnp.take_along_axis(probs, labels[:, None], axis=1).squeeze(1)
    direction = one_hot - probs

    beta_plus = (
        correct_prob
        * (LAMBDA_TPO - 1.0)
        / (1.0 - correct_prob + LAMBDA_TPO * correct_prob)
    )
    success = (correct_prob * beta_plus)[:, None] * direction

    gamma = probs * (LAMBDA_TPO - 1.0) / (LAMBDA_TPO * (1.0 - probs) + probs)
    wrong_coeff = probs * gamma * (1.0 - one_hot)
    wrong = probs * wrong_coeff.sum(axis=1, keepdims=True) - wrong_coeff
    exact_update = success + wrong
    exact_gain = _first_order_gain(probs, labels, exact_update)

    q = (1.0 - correct_prob) / (num_classes - 1)
    gamma_q = q * (LAMBDA_TPO - 1.0) / (LAMBDA_TPO * (1.0 - q) + q)
    beta_sym = correct_prob * beta_plus + correct_prob * gamma_q
    scalar_gain = _scalar_gain(probs, labels, beta_sym)
    return exact_gain, scalar_gain


def _noop_diagnostic(
    probs: jax.Array, labels: jax.Array
) -> tuple[jax.Array, jax.Array]:
    del labels
    nan = jnp.full((probs.shape[0],), jnp.nan, dtype=probs.dtype)
    return nan, nan


def _diagnostic_fn(
    algo_name: str, eta: float
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    if algo_name == "pg":
        return _pg_diagnostic
    if algo_name == "group_pg":
        return _group_pg_diagnostic
    if algo_name == "dg":
        return lambda probs, labels: _dg_diagnostic(probs, labels, eta=eta)
    if algo_name == "tpo":
        return _tpo_diagnostic
    if algo_name == "grpo":
        return _noop_diagnostic
    raise ValueError(f"Unsupported MNIST mechanism algorithm: {algo_name}")


@lru_cache(maxsize=None)
def _build_mnist_mechanism_run_fn(
    algo_name: str,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    eta: float,
    eval_every: int,
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    num_blocks = _validate_eval_schedule(num_steps, eval_every)
    train_images, train_labels, test_images, test_labels = _load_mnist()
    num_train = train_images.shape[0]

    model = PolicyMLP()
    optimizer = optax.adam(learning_rate)
    init_inputs = jnp.ones((1, 784), dtype=train_images.dtype)
    diagnostic_fn = _diagnostic_fn(algo_name, eta)

    def pg_loss_fn(logits, batch_labels, key):
        return classification_pg_loss(logits, batch_labels, key)

    def dg_loss_fn(logits, batch_labels, key):
        return classification_dg_loss(logits, batch_labels, key, eta=eta)

    def tpo_loss_fn(logits, batch_labels, key):
        return classification_tpo_loss(logits, batch_labels, key, eta=eta)

    def group_pg_loss_fn(logits, batch_labels, key):
        return classification_group_pg_loss(logits, batch_labels, key)

    def grpo_loss_fn(logits, batch_labels, key):
        return classification_grpo_loss(logits, batch_labels, key, eps=EPS)

    loss_fns = {
        "pg": pg_loss_fn,
        "dg": dg_loss_fn,
        "tpo": tpo_loss_fn,
        "grpo": grpo_loss_fn,
        "group_pg": group_pg_loss_fn,
    }
    loss_fn = loss_fns[algo_name]

    def train_one(key):
        key, init_key = jr.split(key)
        params = model.init(init_key, init_inputs)
        opt_state = optimizer.init(params)

        def eval_block(carry, block_keys):
            params, opt_state = carry

            def step_fn(carry, step_key):
                params, opt_state = carry
                batch_key, loss_key = jr.split(step_key)
                idx = jr.randint(batch_key, (batch_size,), 0, num_train)
                batch_imgs = train_images[idx]
                batch_lbls = train_labels[idx]

                def full_loss(p):
                    logits = model.apply(p, batch_imgs)
                    return loss_fn(logits, batch_lbls, loss_key)

                grads = jax.grad(full_loss)(params)
                updates, new_opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return (new_params, new_opt_state), None

            (params, opt_state), _ = jax.lax.scan(
                step_fn, (params, opt_state), block_keys
            )

            test_logits = model.apply(params, test_images)
            test_probs = jax.nn.softmax(test_logits, axis=-1)
            test_pred = jnp.argmax(test_logits, axis=-1)
            test_err = 1.0 - jnp.mean(test_pred == test_labels)

            exact_gain, scalar_gain = diagnostic_fn(test_probs, test_labels)
            concentration = _wrong_class_concentration(test_probs, test_labels)
            mistake_mask = test_pred != test_labels
            gain_by_bin, counts = _bin_means(
                exact_gain, concentration, mistake_mask, CONCENTRATION_BINS
            )
            scalar_by_bin, _ = _bin_means(
                scalar_gain, concentration, mistake_mask, CONCENTRATION_BINS
            )
            return (params, opt_state), (test_err, gain_by_bin, scalar_by_bin, counts)

        all_keys = jr.split(key, num_steps).reshape(num_blocks, eval_every, 2)
        _, outputs = jax.lax.scan(eval_block, (params, opt_state), all_keys)
        return outputs

    return jax.jit(jax.vmap(train_one))


def run_mnist_mechanism(
    config: MnistConfig | None = None,
    /,
    algorithms: tuple[str, ...] | None = None,
    **overrides: object,
) -> ExperimentReport:
    config = coerce_config(config, MnistConfig, overrides)
    algo_names = algorithms if algorithms is not None else MECHANISM_ALGORITHMS
    keys = jr.split(jr.PRNGKey(config.seed), config.num_seeds)
    eval_steps = (
        jnp.arange(_validate_eval_schedule(config.num_steps, config.eval_every))
        * config.eval_every
    )

    results: dict[str, dict[str, np.ndarray]] = {}
    for algo_name in algo_names:
        if algo_name not in MECHANISM_ALGORITHMS:
            raise ValueError(
                f"Unsupported MNIST mechanism algorithm: {algo_name}. "
                f"Available: {MECHANISM_ALGORITHMS}"
            )
        run_fn = _build_mnist_mechanism_run_fn(
            algo_name,
            config.num_steps,
            config.batch_size,
            config.learning_rate,
            config.eta,
            config.eval_every,
        )
        print(f"  Training {algo_name} with mechanism diagnostics...")
        errors, gain_by_bin, scalar_by_bin, counts_by_bin = run_fn(keys)
        results[algo_name] = {
            "errors": np.asarray(errors),
            "gain_by_bin": np.asarray(gain_by_bin),
            "scalar_gain_by_bin": np.asarray(scalar_by_bin),
            "surplus_gain_by_bin": np.asarray(gain_by_bin - scalar_by_bin),
            "counts_by_bin": np.asarray(counts_by_bin),
            "eval_steps": np.asarray(eval_steps),
            "bin_edges": np.asarray(CONCENTRATION_BINS),
        }

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    raw_path = save_dir / "mnist_mechanism.npz"
    np.savez(
        raw_path,
        **{
            f"{algo}_{key}": value
            for algo, payload in results.items()
            for key, value in payload.items()
        },
    )
    print(f"  Saved {raw_path}")

    summary: dict[str, float] = {}
    series: dict[str, np.ndarray] = {}
    for algo, payload in results.items():
        errors = payload["errors"]
        mean = errors.mean(axis=0)
        se = errors.std(axis=0) / np.sqrt(errors.shape[0])
        summary[f"{algo}/final_error"] = float(mean[-1])
        series[f"{algo}_mean"] = mean
        series[f"{algo}_se"] = se
        step_idx = int(np.argmin(np.abs(payload["eval_steps"] - 2000)))
        high_bin = -1
        valid = payload["counts_by_bin"][:, step_idx, high_bin] > 0
        if np.any(valid):
            summary[f"{algo}/step2000_high_concentration_surplus"] = float(
                payload["surplus_gain_by_bin"][valid, step_idx, high_bin].mean()
            )

    return ExperimentReport(
        name="mnist_mechanism",
        config=asdict(config),
        summary=summary,
        curves=(
            CurveReport(
                name="mnist_mechanism_error",
                title="MNIST mechanism classification error",
                x_name="step",
                x_values=np.asarray(eval_steps),
                series=series,
                plot_series=tuple(f"{algo}_mean" for algo in algo_names),
            ),
        ),
        artifact_paths=(raw_path,),
        raw_errors=results,
    )
