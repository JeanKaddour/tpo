"""Tabular bandit experiments — Figures 5 and 6.

Single-context (Section 4.1): K=100 actions, sampled gradients, normalized steps.
Multi-context (Section 4.2): N=100 contexts, K=10 actions, exact population gradients.
"""

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

import numpy as np

from ..algorithms import tpo_skill
from ..config import TabularMultiConfig, TabularSingleConfig, coerce_config
from ..core import cosine_misalignment
from ..tracking import CurveReport, ExperimentReport, log_per_algorithm_runs


def _build_algo_series(algos, num_seeds):
    """Build summary, error_series, misalign_series from algo results."""
    summary = {}
    error_series = {}
    misalign_series = {}
    for label, errors, misalign, _ in algos:
        key = label.lower()
        err_mean = errors.mean(axis=0)
        mis_mean = misalign.mean(axis=0)
        summary[f"{key}/final_error"] = float(err_mean[-1])
        summary[f"{key}/final_misalignment"] = float(mis_mean[-1])
        error_series[f"{key}_mean"] = err_mean
        error_series[f"{key}_se"] = errors.std(axis=0) / jnp.sqrt(num_seeds)
        misalign_series[f"{key}_mean"] = mis_mean
        misalign_series[f"{key}_se"] = misalign.std(axis=0) / jnp.sqrt(num_seeds)
    return summary, error_series, misalign_series


# ---------------------------------------------------------------------------
# Single-context (Figure 5)
# ---------------------------------------------------------------------------


def _run_single_seed(key, K, B, alpha, eta, num_steps):
    """Run single-context experiment for one seed, tracking PG and DG in parallel."""
    logits_init = jnp.zeros(K)
    keys = jr.split(key, num_steps)

    def _sample_actions(key_a, logits):
        return jr.categorical(key_a, logits, shape=(B,))

    def step_pg(logits, key):
        pi = jax.nn.softmax(logits)
        y_star = 0
        key_a, _ = jr.split(key)
        actions = _sample_actions(key_a, logits)
        rewards = (actions == y_star).astype(jnp.float32)
        advantages = rewards - 0.5  # constant baseline b=0.5
        one_hot_a = jax.nn.one_hot(actions, K)
        scores = one_hot_a - pi[None, :]
        pg_grad = (advantages[:, None] * scores).mean(axis=0)
        oracle_pg = pi[y_star] * (jax.nn.one_hot(y_star, K) - pi)
        pg_norm = jnp.sqrt(jnp.sum(pg_grad**2) + 1e-30)
        logits_new = logits + alpha * pg_grad / pg_norm
        error = 1.0 - pi[y_star]  # current policy error (before step)
        misalign = cosine_misalignment(pg_grad, oracle_pg)
        return logits_new, (error, misalign)

    def step_dg(logits, key):
        pi = jax.nn.softmax(logits)
        y_star = 0
        key_a, _ = jr.split(key)
        actions = _sample_actions(key_a, logits)
        rewards = (actions == y_star).astype(jnp.float32)
        advantages = rewards - 0.5  # constant baseline b=0.5
        one_hot_a = jax.nn.one_hot(actions, K)
        scores = one_hot_a - pi[None, :]
        surprisal = -jnp.log(pi[actions] + 1e-30)
        chi = advantages * surprisal
        w = jax.nn.sigmoid(chi / eta)
        dg_grad = (w[:, None] * advantages[:, None] * scores).mean(axis=0)
        oracle_pg = pi[y_star] * (jax.nn.one_hot(y_star, K) - pi)
        dg_norm = jnp.sqrt(jnp.sum(dg_grad**2) + 1e-30)
        logits_new = logits + alpha * dg_grad / dg_norm
        error = 1.0 - pi[y_star]  # current policy error (before step)
        misalign = cosine_misalignment(dg_grad, oracle_pg)
        return logits_new, (error, misalign)

    def step_tpo(logits, key):
        pi = jax.nn.softmax(logits)
        y_star = 0
        key_a, _ = jr.split(key)
        actions = _sample_actions(key_a, logits)
        rewards = (actions == y_star).astype(jnp.float32)
        advantages = rewards - 0.5  # constant baseline b=0.5
        # Build per-sample score vectors and average
        scores = advantages[:, None] * jax.nn.one_hot(actions, K)  # (B, K)
        avg_scores = scores.mean(axis=0)  # (K,)
        u = tpo_skill(avg_scores)
        # Mirror target: q ∝ π * exp(u)
        q = jax.nn.softmax(jnp.log(pi + 1e-8) + u)
        # Gradient of CE loss w.r.t. logits: update direction is q - π
        tsm_grad = q - pi
        oracle_pg = pi[y_star] * (jax.nn.one_hot(y_star, K) - pi)
        tsm_norm = jnp.sqrt(jnp.sum(tsm_grad**2) + 1e-30)
        logits_new = logits + alpha * tsm_grad / tsm_norm
        error = 1.0 - pi[y_star]
        misalign = cosine_misalignment(tsm_grad, oracle_pg)
        return logits_new, (error, misalign)

    def step_grpo(logits, key):
        pi = jax.nn.softmax(logits)
        y_star = 0
        key_a, _ = jr.split(key)
        actions = _sample_actions(key_a, logits)
        rewards = (actions == y_star).astype(jnp.float32)
        # Group-relative advantage normalization
        adv_mean = rewards.mean()
        adv_std = jnp.sqrt(jnp.mean((rewards - adv_mean) ** 2) + 1e-8)
        advantages = (rewards - adv_mean) / adv_std
        one_hot_a = jax.nn.one_hot(actions, K)
        scores = one_hot_a - pi[None, :]
        grpo_grad = (advantages[:, None] * scores).mean(axis=0)
        oracle_pg = pi[y_star] * (jax.nn.one_hot(y_star, K) - pi)
        grpo_norm = jnp.sqrt(jnp.sum(grpo_grad**2) + 1e-30)
        logits_new = logits + alpha * grpo_grad / grpo_norm
        error = 1.0 - pi[y_star]
        misalign = cosine_misalignment(grpo_grad, oracle_pg)
        return logits_new, (error, misalign)

    _, (pg_errors, pg_misalign) = jax.lax.scan(step_pg, logits_init, keys)
    _, (dg_errors, dg_misalign) = jax.lax.scan(step_dg, logits_init, keys)
    _, (tpo_errors, tpo_misalign) = jax.lax.scan(step_tpo, logits_init, keys)
    _, (grpo_errors, grpo_misalign) = jax.lax.scan(step_grpo, logits_init, keys)

    return (
        pg_errors,
        dg_errors,
        tpo_errors,
        grpo_errors,
        pg_misalign,
        dg_misalign,
        tpo_misalign,
        grpo_misalign,
    )


@lru_cache(maxsize=None)
def _build_single_context_runner(
    num_actions: int,
    batch_size: int,
    alpha: float,
    eta: float,
    num_steps: int,
):
    """Build a cached compiled runner for the single-context tabular experiment."""
    return jax.jit(
        jax.vmap(
            lambda key: _run_single_seed(
                key, num_actions, batch_size, alpha, eta, num_steps
            )
        )
    )


def run_single_context(
    config: TabularSingleConfig | None = None,
    /,
    **overrides: object,
):
    """Run single-context tabular bandit (Figure 5)."""
    config = coerce_config(config, TabularSingleConfig, overrides)
    num_seeds = config.num_seeds
    num_actions = config.num_actions
    batch_size = config.batch_size
    alpha = config.alpha
    eta = config.eta
    num_steps = config.num_steps
    save_dir = config.save_dir

    print(
        f"Running single-context tabular bandit: {num_seeds} seeds, {num_steps} steps..."
    )

    keys = jr.split(jr.PRNGKey(0), num_seeds)
    run_fn = _build_single_context_runner(
        num_actions, batch_size, alpha, eta, num_steps
    )
    (
        pg_errors,
        dg_errors,
        tpo_errors,
        grpo_errors,
        pg_misalign,
        dg_misalign,
        tpo_misalign,
        grpo_misalign,
    ) = run_fn(keys)
    # Each shape: (num_seeds, num_steps)

    steps = jnp.arange(num_steps)
    algos = [
        ("PG", pg_errors, pg_misalign, "tab:blue"),
        ("DG", dg_errors, dg_misalign, "tab:red"),
        ("TPO", tpo_errors, tpo_misalign, "tab:purple"),
        ("GRPO", grpo_errors, grpo_misalign, "tab:green"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for label, errors, misalign, color in algos:
        err_mean = errors.mean(axis=0)
        err_se = errors.std(axis=0) / jnp.sqrt(num_seeds)
        mis_mean = misalign.mean(axis=0)
        mis_se = misalign.std(axis=0) / jnp.sqrt(num_seeds)
        ax1.semilogy(steps, err_mean, color=color, label=label)
        ax1.fill_between(
            steps,
            jnp.maximum(err_mean - err_se, 1e-12),
            err_mean + err_se,
            color=color,
            alpha=0.2,
        )
        ax2.semilogy(steps, mis_mean, color=color, label=label)
        ax2.fill_between(
            steps,
            jnp.maximum(mis_mean - mis_se, 1e-12),
            mis_mean + mis_se,
            color=color,
            alpha=0.2,
        )

    ax1.set_xlabel("step T")
    ax1.set_ylabel(r"$\varepsilon = 1 - \pi(y^*)$")
    ax1.legend()
    ax1.set_title("(a) Error")
    ax2.set_xlabel("step T")
    ax2.set_ylabel(r"$1 - \cos(g, g^*_{\mathrm{PG}})$")
    ax2.legend()
    ax2.set_title("(b) Misalignment")

    fig.suptitle(
        f"Figure 5: Symmetric bandit (K={num_actions}, B={batch_size}, α={alpha})"
    )
    fig.tight_layout()
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    figure_path = save_dir_path / "figure5_tabular_single.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {figure_path}")

    summary, error_series, misalign_series = _build_algo_series(algos, num_seeds)

    return ExperimentReport(
        name="tabular_single",
        config=asdict(config),
        summary=summary,
        curves=(
            CurveReport(
                name="error",
                title="Single-context error",
                x_name="step",
                x_values=steps,
                series=error_series,
                plot_series=tuple(k for k in error_series if k.endswith("_mean")),
            ),
            CurveReport(
                name="misalignment",
                title="Single-context misalignment",
                x_name="step",
                x_values=steps,
                series=misalign_series,
                plot_series=tuple(k for k in misalign_series if k.endswith("_mean")),
            ),
        ),
        artifact_paths=(figure_path,),
    )


# ---------------------------------------------------------------------------
# Multi-context (Figure 6)
# ---------------------------------------------------------------------------


def _run_multi_seed(key, N, K, alpha, eta, num_steps):
    """Run multi-context experiment for one seed with exact population gradients."""
    # Initialize logits ~ N(0, 1)
    logits_init = jr.normal(key, (N, K))
    # Correct action for each context: index 0 (WLOG since logits are random)
    y_star = jnp.zeros(N, dtype=jnp.int32)

    def run_alg(logits_init, alg):
        """Run one algorithm for all steps."""

        def step_alg(logits, _):
            pi = jax.nn.softmax(logits, axis=-1)
            p_n = pi[:, 0]
            e_y = jax.nn.one_hot(y_star, K)
            v_n = e_y - pi

            if alg == "pg":
                grad_n = p_n[:, None] * v_n
            elif alg == "dg":
                surprisal = -jnp.log(p_n + 1e-30)
                dg_weight = p_n * jax.nn.sigmoid(surprisal / eta)
                grad_n = dg_weight[:, None] * v_n
            elif alg == "tpo":
                # TPO target per context, using z-scored whitening
                u = tpo_skill(e_y)  # (N, K)
                q = jax.nn.softmax(jnp.log(pi + 1e-8) + u, axis=-1)  # (N, K)
                grad_n = q - pi  # (N, K)
            elif alg == "grpo":
                # Exact population GRPO: group-normalized advantages
                # σ_n = sqrt(p_n(1-p_n)), weight = sqrt(p_n/(1-p_n))
                grpo_weight = jnp.sqrt(p_n / (1 - p_n + 1e-8))
                grad_n = grpo_weight[:, None] * v_n
            else:  # ce
                grad_n = v_n

            ce_grad = v_n.reshape(-1)
            alg_grad = grad_n.reshape(-1)

            grad_norm = jnp.sqrt(jnp.sum(grad_n**2) + 1e-30)
            logits_new = logits + alpha * grad_n / grad_norm

            error = 1.0 - jax.nn.softmax(logits_new, axis=-1)[:, 0].mean()
            misalign = cosine_misalignment(alg_grad, ce_grad)

            return logits_new, (error, misalign)

        return jax.lax.scan(step_alg, logits_init, None, length=num_steps)

    _, (pg_errors, pg_misalign) = run_alg(logits_init, "pg")
    _, (dg_errors, dg_misalign) = run_alg(logits_init, "dg")
    _, (tpo_errors, tpo_misalign) = run_alg(logits_init, "tpo")
    _, (grpo_errors, grpo_misalign) = run_alg(logits_init, "grpo")
    _, (ce_errors, ce_misalign) = run_alg(logits_init, "ce")

    return (
        pg_errors,
        dg_errors,
        tpo_errors,
        grpo_errors,
        ce_errors,
        pg_misalign,
        dg_misalign,
        tpo_misalign,
        grpo_misalign,
        ce_misalign,
    )


@lru_cache(maxsize=None)
def _build_multi_context_runner(
    num_contexts: int,
    num_actions: int,
    alpha: float,
    eta: float,
    num_steps: int,
):
    """Build a cached compiled runner for the multi-context tabular experiment."""
    return jax.jit(
        jax.vmap(
            lambda key: _run_multi_seed(
                key, num_contexts, num_actions, alpha, eta, num_steps
            )
        )
    )


def run_multi_context(
    config: TabularMultiConfig | None = None,
    /,
    **overrides: object,
):
    """Run multi-context tabular bandit (Figure 6)."""
    config = coerce_config(config, TabularMultiConfig, overrides)
    num_seeds = config.num_seeds
    num_contexts = config.num_contexts
    num_actions = config.num_actions
    alpha = config.alpha
    eta = config.eta
    num_steps = config.num_steps
    save_dir = config.save_dir

    print(
        f"Running multi-context tabular bandit: {num_seeds} seeds, {num_steps} steps..."
    )

    keys = jr.split(jr.PRNGKey(42), num_seeds)
    run_fn = _build_multi_context_runner(
        num_contexts, num_actions, alpha, eta, num_steps
    )
    (
        pg_errors,
        dg_errors,
        tpo_errors,
        grpo_errors,
        ce_errors,
        pg_misalign,
        dg_misalign,
        tpo_misalign,
        grpo_misalign,
        ce_misalign,
    ) = run_fn(keys)

    steps = jnp.arange(num_steps)

    algos = [
        ("PG", pg_errors, pg_misalign, "tab:blue"),
        ("DG", dg_errors, dg_misalign, "tab:red"),
        ("TPO", tpo_errors, tpo_misalign, "tab:purple"),
        ("GRPO", grpo_errors, grpo_misalign, "tab:green"),
        ("CE", ce_errors, ce_misalign, "tab:gray"),
    ]

    # ---- Log per-algorithm wandb runs ----
    log_per_algorithm_runs(
        "tabular_multi",
        config,
        [
            (
                label.lower(),
                {
                    "average_error": np.asarray(errors),
                    "misalignment": np.asarray(misalign),
                },
            )
            for label, errors, misalign, _ in algos
        ],
        x_name="step",
        x_values=np.asarray(steps),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for label, errors, misalign, color in algos:
        err_mean = errors.mean(axis=0)
        err_se = errors.std(axis=0) / jnp.sqrt(num_seeds)
        mis_mean = misalign.mean(axis=0)
        mis_se = misalign.std(axis=0) / jnp.sqrt(num_seeds)
        ax1.semilogy(steps, err_mean, color=color, label=label)
        ax1.fill_between(
            steps,
            jnp.maximum(err_mean - err_se, 1e-12),
            err_mean + err_se,
            color=color,
            alpha=0.2,
        )
        ax2.semilogy(steps, mis_mean, color=color, label=label)
        ax2.fill_between(
            steps,
            jnp.maximum(mis_mean - mis_se, 1e-12),
            mis_mean + mis_se,
            color=color,
            alpha=0.2,
        )

    ax1.set_xlabel("step")
    ax1.set_ylabel(r"average error $1 - \bar{p}$")
    ax1.legend()
    ax1.set_title(r"(a) Average error $1 - \bar{p}$")
    ax2.set_xlabel("step")
    ax2.set_ylabel(r"$1 - \cos(g, g^*_{\mathrm{CE}})$")
    ax2.legend()
    ax2.set_title(r"(b) Misalignment to $g^*_{\mathrm{CE}}$")

    fig.suptitle(
        f"Figure 6: Multi-context (N={num_contexts}, K={num_actions}, exact gradients)"
    )
    fig.tight_layout()
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    figure_path = save_dir_path / "figure6_tabular_multi.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {figure_path}")

    summary, error_series, misalign_series = _build_algo_series(algos, num_seeds)

    return ExperimentReport(
        name="tabular_multi",
        config=asdict(config),
        summary=summary,
        curves=(
            CurveReport(
                name="average_error",
                title="Multi-context average error",
                x_name="step",
                x_values=steps,
                series=error_series,
                plot_series=tuple(k for k in error_series if k.endswith("_mean")),
            ),
            CurveReport(
                name="misalignment",
                title="Multi-context misalignment",
                x_name="step",
                x_values=steps,
                series=misalign_series,
                plot_series=tuple(k for k in misalign_series if k.endswith("_mean")),
            ),
        ),
        artifact_paths=(figure_path,),
    )
