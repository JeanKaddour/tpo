"""Terminal reward ablation diagnostics for the TPO paper.

Tests three hypotheses about why TPO handles sparse reward better:
A. Self-extinguishing gradient: TPO's gradient vanishes once policy matches target
B. Zero-variance group robustness: TPO ignores uninformative (all-fail) groups
C. Multi-epoch extraction: TPO safely reuses data across epochs

Run locally:
    uv run python -m tpo.cli terminal_reward_ablations --smoke
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..algorithms import dg_loss, grpo_loss, ppo_loss, tpo_target
from ..config import TRANSFORMER_MODEL, TerminalRewardAblationsConfig, coerce_config
from ..core import dg_gate, flatten_pytree
from ..models import CausalTransformer
from ..tracking import CurveReport, ExperimentReport

# ── Diagnostic indices ──────────────────────────────────────────────
IDX_ERROR = 0
IDX_GRAD_NORM = 1
IDX_FRAC_SUCCESS = 2
IDX_FRAC_ZERO_VAR = 3
IDX_WEIGHT_WINNERS = 4
IDX_WEIGHT_LOSERS = 5
NUM_DIAGS = 6

DIAG_NAMES = (
    "error",
    "grad_norm",
    "frac_success",
    "frac_zero_var",
    "weight_winners",
    "weight_losers",
)

# ── Algorithms ──────────────────────────────────────────────────────
ALGORITHMS = ("tpo", "tpo_1ep", "grpo", "grpo_no_kl", "ppo", "dg")
_GROUPED = {"tpo", "tpo_1ep", "grpo", "grpo_no_kl"}

ALGO_COLORS = {
    "tpo": "tab:brown",
    "tpo_1ep": "tab:orange",
    "grpo": "tab:cyan",
    "grpo_no_kl": "teal",
    "ppo": "tab:blue",
    "dg": "tab:red",
}
ALGO_LABELS = {
    "tpo": "TPO (4 ep)",
    "tpo_1ep": "TPO (1 ep)",
    "grpo": "GRPO",
    "grpo_no_kl": "GRPO (no KL)",
    "ppo": "PPO",
    "dg": "DG",
}


# ════════════════════════════════════════════════════════════════════
#  Build JIT-compiled runners with per-step diagnostics
# ════════════════════════════════════════════════════════════════════


def _build_ablation_fns(
    cfg: TerminalRewardAblationsConfig,
    algorithms: tuple[str, ...],
) -> dict[str, callable]:
    """Build vmapped, JIT-compiled runners for each requested algorithm."""

    sl = cfg.sequence_length
    vs = cfg.vocab_size
    bs = cfg.batch_size
    K = cfg.k_candidates
    seq_len = 2 * sl

    model = CausalTransformer(
        vocab_size=vs,
        max_seq_len=seq_len,
        d_model=TRANSFORMER_MODEL.d_model,
        num_heads=TRANSFORMER_MODEL.num_heads,
        num_layers=TRANSFORMER_MODEL.num_layers,
        ffn_mult=TRANSFORMER_MODEL.ffn_mult,
    )

    from optax.contrib import muon as optax_muon

    optimizer = optax_muon(learning_rate=cfg.learning_rate * 10)

    def init_params(key):
        return model.init(key, jnp.zeros((1, seq_len), dtype=jnp.int32))

    def compute_targets(prompts):
        return prompts[:, ::-1]  # reverse_copy

    def compute_rewards(actions, targets):
        correct = (actions == targets).astype(jnp.float32)
        all_correct = jnp.all(correct, axis=1, keepdims=True).astype(jnp.float32)
        return jnp.broadcast_to(all_correct, correct.shape)

    # ── Single rollout (PPO, DG) ────────────────────────────────────

    def rollout(params, key):
        key_p, key_s = jr.split(key)
        prompts = jr.randint(key_p, (bs, sl), 0, vs)
        targets = compute_targets(prompts)

        def step(carry, t):
            tokens, rng = carry
            rng_t, rng_next = jr.split(rng)
            all_lp = model.apply(params, tokens)
            tok_lp = all_lp[:, sl + t - 1, :]
            sampled = jr.categorical(rng_t, tok_lp)
            lp = tok_lp[jnp.arange(bs), sampled]
            tokens = tokens.at[:, sl + t].set(sampled)
            return (tokens, rng_next), (sampled, lp)

        buf = jnp.zeros((bs, seq_len), dtype=jnp.int32).at[:, :sl].set(prompts)
        (_, _), (acts, lps) = jax.lax.scan(step, (buf, key_s), jnp.arange(sl))
        actions, log_probs = acts.T, lps.T
        return prompts, actions, log_probs, compute_rewards(actions, targets)

    # ── Grouped rollout (TPO, GRPO) ─────────────────────────────────

    def grouped_rollout(params, key):
        key_p, key_s = jr.split(key)
        prompts = jr.randint(key_p, (bs, sl), 0, vs)
        targets = compute_targets(prompts)

        def single_rollout(key_k):
            def step(carry, t):
                tokens, rng = carry
                rng_t, rng_next = jr.split(rng)
                all_lp = model.apply(params, tokens)
                tok_lp = all_lp[:, sl + t - 1, :]
                sampled = jr.categorical(rng_t, tok_lp)
                lp = tok_lp[jnp.arange(bs), sampled]
                tokens = tokens.at[:, sl + t].set(sampled)
                return (tokens, rng_next), (sampled, lp)

            buf = jnp.zeros((bs, seq_len), dtype=jnp.int32).at[:, :sl].set(prompts)
            (_, _), (acts, lps) = jax.lax.scan(step, (buf, key_k), jnp.arange(sl))
            return acts.T, lps.T, compute_rewards(acts.T, targets)

        keys = jr.split(key_s, K)
        all_a, all_lp, all_r = jax.vmap(single_rollout)(keys)
        return (
            prompts,
            jnp.transpose(all_a, (1, 0, 2)),  # (B, K, H)
            jnp.transpose(all_lp, (1, 0, 2)),  # (B, K, H)
            jnp.transpose(all_r, (1, 0, 2)),  # (B, K, H)
        )

    def compute_log_probs(params, prompts, actions):
        full = jnp.concatenate([prompts, actions], axis=1)
        all_lp = model.apply(params, full)
        pred_lp = all_lp[:, sl - 1 : 2 * sl - 1, :]
        return pred_lp[
            jnp.arange(pred_lp.shape[0])[:, None],
            jnp.arange(sl)[None, :],
            actions,
        ]

    def _grad_norm(grads):
        return jnp.sqrt(jnp.sum(flatten_pytree(grads) ** 2))

    # ── TPO step (parametrized by epoch count) ──────────────────────

    def make_tpo_step(epochs):
        def step(params, opt_state, key):
            prompts, all_actions, all_tok_lp, all_tok_reward = grouped_rollout(
                params, key
            )
            episode_scores = all_tok_reward.mean(axis=2)  # (B, K), binary
            log_scores_old = jax.lax.stop_gradient(all_tok_lp.sum(axis=2))
            q_target = jax.lax.stop_gradient(
                tpo_target(log_scores_old, episode_scores, eta=cfg.tpo_eta)
            )

            flat_prompts = jnp.repeat(prompts, K, axis=0)
            flat_actions = all_actions.reshape(bs * K, sl)

            def loss_fn(p):
                full = jnp.concatenate([flat_prompts, flat_actions], axis=1)
                all_lp = model.apply(p, full)
                pred_lp = all_lp[:, sl - 1 : 2 * sl - 1, :]
                flat_tok_lp = pred_lp[
                    jnp.arange(bs * K)[:, None],
                    jnp.arange(sl)[None, :],
                    flat_actions,
                ]
                new_log_p = jax.nn.log_softmax(
                    flat_tok_lp.reshape(bs, K, sl).sum(axis=2), axis=1
                )
                return -(q_target * new_log_p).sum(axis=1).mean()

            def epoch_step(carry, _):
                p, os_ = carry
                _, g = jax.value_and_grad(loss_fn)(p)
                gn = _grad_norm(g)
                updates, new_os = optimizer.update(g, os_, p)
                new_p = optax.apply_updates(p, updates)
                return (new_p, new_os), gn

            (new_params, new_opt_state), epoch_gns = jax.lax.scan(
                epoch_step, (params, opt_state), None, length=epochs
            )
            grad_norm = epoch_gns[0]

            # ── Diagnostics ──
            error = 1.0 - all_tok_reward.mean()
            success = episode_scores > 0.5  # (B, K)
            frac_success = success.any(axis=1).mean()
            frac_zero_var = (episode_scores.std(axis=1) < 1e-6).mean()

            # Weight proxy: mean target mass q_i on successful vs failed candidates
            n_win = success.sum().clip(1)
            n_lose = (~success).sum().clip(1)
            w_win = (q_target * success).sum() / n_win
            w_lose = (q_target * ~success).sum() / n_lose

            diags = jnp.array(
                [error, grad_norm, frac_success, frac_zero_var, w_win, w_lose]
            )
            return new_params, new_opt_state, diags

        return step

    # ── GRPO step ───────────────────────────────────────────────────

    def make_grpo_ablation_step(beta):
        def step(params, opt_state, key):
            prompts, all_actions, all_tok_lp, all_tok_reward = grouped_rollout(
                params, key
            )
            episode_scores = all_tok_reward.mean(axis=2)  # (B, K)
            group_mean = episode_scores.mean(axis=1, keepdims=True)
            group_std = episode_scores.std(axis=1, keepdims=True)
            advantages = jax.lax.stop_gradient(
                (episode_scores - group_mean) / (group_std + 1e-8)
            )

            flat_prompts = jnp.repeat(prompts, K, axis=0)
            flat_actions = all_actions.reshape(bs * K, sl)
            old_log_scores = jax.lax.stop_gradient(all_tok_lp.sum(axis=2))

            def loss_fn(p):
                full = jnp.concatenate([flat_prompts, flat_actions], axis=1)
                all_lp = model.apply(p, full)
                pred_lp = all_lp[:, sl - 1 : 2 * sl - 1, :]
                flat_tok_lp = pred_lp[
                    jnp.arange(bs * K)[:, None],
                    jnp.arange(sl)[None, :],
                    flat_actions,
                ]
                new_log_scores = flat_tok_lp.reshape(bs, K, sl).sum(axis=2)
                return grpo_loss(
                    new_log_scores,
                    old_log_scores,
                    advantages,
                    epsilon=cfg.ppo_epsilon,
                    beta=beta,
                )

            def epoch_step(carry, _):
                p, os_ = carry
                _, g = jax.value_and_grad(loss_fn)(p)
                gn = _grad_norm(g)
                updates, new_os = optimizer.update(g, os_, p)
                new_p = optax.apply_updates(p, updates)
                return (new_p, new_os), gn

            (new_params, new_opt_state), epoch_gns = jax.lax.scan(
                epoch_step, (params, opt_state), None, length=cfg.ppo_epochs
            )
            grad_norm = epoch_gns[0]

            error = 1.0 - all_tok_reward.mean()
            success = episode_scores > 0.5
            frac_success = success.any(axis=1).mean()
            frac_zero_var = (episode_scores.std(axis=1) < 1e-6).mean()

            n_win = success.sum().clip(1)
            n_lose = (~success).sum().clip(1)
            w_win = (jnp.abs(advantages) * success).sum() / n_win
            w_lose = (jnp.abs(advantages) * ~success).sum() / n_lose

            diags = jnp.array(
                [error, grad_norm, frac_success, frac_zero_var, w_win, w_lose]
            )
            return new_params, new_opt_state, diags

        return step

    grpo_step_fn = make_grpo_ablation_step(beta=0.04)
    grpo_no_kl_step_fn = make_grpo_ablation_step(beta=0.0)

    # ── PPO step ────────────────────────────────────────────────────

    def ppo_step_fn(params, opt_state, key):
        prompts, actions, old_lp, rewards = rollout(params, key)
        advantages = rewards - rewards.mean(axis=0, keepdims=True)
        old_log_probs = jax.lax.stop_gradient(old_lp)

        def loss_fn(p):
            lp = compute_log_probs(p, prompts, actions)
            return ppo_loss(lp, old_log_probs, advantages, epsilon=cfg.ppo_epsilon)

        def epoch_step(carry, _):
            p, os_ = carry
            _, g = jax.value_and_grad(loss_fn)(p)
            gn = _grad_norm(g)
            updates, new_os = optimizer.update(g, os_, p)
            new_p = optax.apply_updates(p, updates)
            return (new_p, new_os), gn

        (new_params, new_opt_state), epoch_gns = jax.lax.scan(
            epoch_step, (params, opt_state), None, length=cfg.ppo_epochs
        )
        grad_norm = epoch_gns[0]

        error = 1.0 - rewards.mean()
        ep_reward = rewards[:, 0]  # terminal: all cols identical
        frac_success = (ep_reward > 0.5).mean()

        # Weight: mean |advantage| on successful vs failed rollouts
        success = ep_reward > 0.5  # (B,)
        n_win = success.sum().clip(1)
        n_lose = (~success).sum().clip(1)
        abs_adv = jnp.abs(advantages[:, 0])
        w_win = (abs_adv * success).sum() / n_win
        w_lose = (abs_adv * ~success).sum() / n_lose

        diags = jnp.array([error, grad_norm, frac_success, -1.0, w_win, w_lose])
        return new_params, new_opt_state, diags

    # ── DG step ─────────────────────────────────────────────────────

    def dg_step_fn(params, opt_state, key):
        prompts, actions, _, rewards = rollout(params, key)
        advantages = rewards - rewards.mean(axis=0, keepdims=True)

        def loss_fn(p):
            return dg_loss(
                compute_log_probs(p, prompts, actions), advantages, eta=cfg.eta
            )

        # DG: single epoch (no trust region)
        _, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = _grad_norm(grads)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        error = 1.0 - rewards.mean()
        ep_reward = rewards[:, 0]
        frac_success = (ep_reward > 0.5).mean()

        # DG effective weight: |gate * advantage|
        log_probs = compute_log_probs(params, prompts, actions)
        surprisal = -log_probs
        gate = dg_gate(advantages, surprisal, eta=cfg.eta)
        eff = jnp.abs(gate * advantages).mean(axis=1)  # per-episode

        success = ep_reward > 0.5
        n_win = success.sum().clip(1)
        n_lose = (~success).sum().clip(1)
        w_win = (eff * success).sum() / n_win
        w_lose = (eff * ~success).sum() / n_lose

        diags = jnp.array([error, grad_norm, frac_success, -1.0, w_win, w_lose])
        return new_params, new_opt_state, diags

    # ── Seed runner ─────────────────────────────────────────────────

    def run_seed(step_fn, seed):
        key = jr.PRNGKey(seed)
        params = init_params(key)
        opt_state = optimizer.init(params)

        def scan_step(carry, t):
            p, os_ = carry
            key_t = jr.fold_in(key, t)
            new_p, new_os, diags = step_fn(p, os_, key_t)
            return (new_p, new_os), diags

        (_, _), all_diags = jax.lax.scan(
            scan_step, (params, opt_state), jnp.arange(cfg.num_episodes)
        )
        return all_diags  # (num_episodes, NUM_DIAGS)

    # ── Assemble requested algorithms ───────────────────────────────

    available = {}
    if "tpo" in algorithms:
        available["tpo"] = make_tpo_step(epochs=cfg.ppo_epochs)
    if "tpo_1ep" in algorithms:
        available["tpo_1ep"] = make_tpo_step(epochs=1)
    if "grpo" in algorithms:
        available["grpo"] = grpo_step_fn
    if "grpo_no_kl" in algorithms:
        available["grpo_no_kl"] = grpo_no_kl_step_fn
    if "ppo" in algorithms:
        available["ppo"] = ppo_step_fn
    if "dg" in algorithms:
        available["dg"] = dg_step_fn

    unknown = [a for a in algorithms if a not in available]
    if unknown:
        raise ValueError(
            f"Unknown algorithm(s): {unknown}. Available: {list(ALGORITHMS)}"
        )

    return {
        name: jax.jit(jax.vmap(lambda seed, _sf=fn: run_seed(_sf, seed)))
        for name, fn in available.items()
    }


# ════════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════════


def _smooth(x, window=50):
    """Moving average with edge padding."""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    padded = np.pad(x, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def _plot_ablation_results(results, cfg, save_dir):
    """Generate 6-panel diagnostic figure."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(cfg.num_episodes)
    algos = list(results.keys())
    grouped_algos = [a for a in algos if a in _GROUPED]
    smooth_w = max(1, cfg.num_episodes // 40)

    def get(algo, idx):
        return results[algo][:, :, idx].mean(axis=0)

    def get_se(algo, idx):
        d = results[algo][:, :, idx]
        return d.std(axis=0) / np.sqrt(d.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    _panel_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]

    def _label_panel(ax, row, col):
        ax.text(
            0.02,
            0.95,
            _panel_labels[row][col],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    # ── (a) Error curves ───────────────────────────────────────────
    ax = axes[0, 0]
    for algo in algos:
        m, se = get(algo, IDX_ERROR), get_se(algo, IDX_ERROR)
        ax.semilogy(episodes, m, color=ALGO_COLORS[algo], label=ALGO_LABELS[algo])
        ax.fill_between(
            episodes,
            np.maximum(m - se, 1e-5),
            m + se,
            color=ALGO_COLORS[algo],
            alpha=0.15,
        )
    ax.set_xlabel("episode")
    ax.set_ylabel("exact-match error")
    ax.set_title("Error curves")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0.05)
    _label_panel(ax, 0, 0)

    # ── (b) Gradient norms ─────────────────────────────────────────
    ax = axes[0, 1]
    for algo in algos:
        m = _smooth(get(algo, IDX_GRAD_NORM), smooth_w)
        ax.plot(episodes, m, color=ALGO_COLORS[algo], label=ALGO_LABELS[algo])
    ax.set_xlabel("episode")
    ax.set_ylabel("gradient L2 norm (first epoch)")
    ax.set_title("Gradient norms")
    ax.legend(fontsize=7)
    _label_panel(ax, 0, 1)

    # ── (c) Informative batch rate ─────────────────────────────────
    ax = axes[0, 2]
    for algo in algos:
        m = _smooth(get(algo, IDX_FRAC_SUCCESS), smooth_w)
        ax.plot(episodes, m, color=ALGO_COLORS[algo], label=ALGO_LABELS[algo])
    ax.set_xlabel("episode")
    ax.set_ylabel("frac. prompts with ≥1 success")
    ax.set_title("Informative batch rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    _label_panel(ax, 0, 2)

    # ── (d) Zero-variance group fraction ───────────────────────────
    ax = axes[1, 0]
    for algo in grouped_algos:
        m = _smooth(get(algo, IDX_FRAC_ZERO_VAR), smooth_w)
        ax.plot(episodes, m, color=ALGO_COLORS[algo], label=ALGO_LABELS[algo])
    ax.set_xlabel("episode")
    ax.set_ylabel("frac. zero-variance groups")
    ax.set_title("Zero-variance groups")
    ax.set_ylim(-0.05, 1.05)
    if grouped_algos:
        ax.legend(fontsize=7)
    _label_panel(ax, 1, 0)

    # ── (e) Weight on winners vs losers ────────────────────────────
    ax = axes[1, 1]
    for algo in grouped_algos:
        w_win = _smooth(get(algo, IDX_WEIGHT_WINNERS), smooth_w)
        w_lose = _smooth(get(algo, IDX_WEIGHT_LOSERS), smooth_w)
        ax.plot(
            episodes,
            w_win,
            color=ALGO_COLORS[algo],
            label=f"{ALGO_LABELS[algo]} winners",
            linestyle="-",
        )
        ax.plot(
            episodes,
            w_lose,
            color=ALGO_COLORS[algo],
            label=f"{ALGO_LABELS[algo]} losers",
            linestyle="--",
            alpha=0.6,
        )
    ax.set_xlabel("episode")
    ax.set_ylabel("per-candidate weight proxy")
    ax.set_title("Weight proxy")
    if grouped_algos:
        ax.legend(fontsize=6)
    _label_panel(ax, 1, 1)

    # ── (f) Multi-epoch ablation ───────────────────────────────────
    ax = axes[1, 2]
    for algo in ("tpo", "tpo_1ep", "dg"):
        if algo not in results:
            continue
        m, se = get(algo, IDX_ERROR), get_se(algo, IDX_ERROR)
        ax.semilogy(
            episodes,
            m,
            color=ALGO_COLORS[algo],
            label=ALGO_LABELS[algo],
        )
        ax.fill_between(
            episodes,
            np.maximum(m - se, 1e-5),
            m + se,
            color=ALGO_COLORS[algo],
            alpha=0.15,
        )
    ax.set_xlabel("episode")
    ax.set_ylabel("exact-match error")
    ax.set_title("Multi-epoch ablation")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0.05)
    _label_panel(ax, 1, 2)

    fig.suptitle(
        f"Terminal reward ablations  "
        f"(H={cfg.sequence_length}, V={cfg.vocab_size}, "
        f"K={cfg.k_candidates}, B={cfg.batch_size})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = save_path / "terminal_reward_ablations.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    # Also save PDF for paper
    pdf_path = save_path / "terminal_reward_ablations.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    # Copy to paper/figures/ if it exists
    paper_fig_dir = Path("paper/figures")
    if paper_fig_dir.exists():
        import shutil

        shutil.copy2(fig_path, paper_fig_dir / fig_path.name)
        shutil.copy2(pdf_path, paper_fig_dir / pdf_path.name)
    plt.close(fig)
    return fig_path


def _save_ablation_series_npz(results, save_dir):
    """Save mean/se diagnostic series so paper plots can be regenerated selectively."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    series: dict[str, np.ndarray] = {}
    for algo, data in results.items():
        for i, name in enumerate(DIAG_NAMES):
            diag = data[:, :, i]
            series[f"{algo}_{name}_mean"] = diag.mean(axis=0)
            series[f"{algo}_{name}_se"] = diag.std(axis=0) / np.sqrt(diag.shape[0])

    npz_path = save_path / "ablation_data.npz"
    np.savez(npz_path, **series)
    return npz_path


# ════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════


def run_terminal_reward_ablations(
    config: TerminalRewardAblationsConfig | None = None,
    /,
    algorithms: tuple[str, ...] | None = None,
    verbose: bool = True,
    **overrides,
) -> ExperimentReport:
    """Run terminal reward ablation diagnostics."""
    config = coerce_config(config, TerminalRewardAblationsConfig, overrides)
    algo_list = ALGORITHMS if algorithms is None else tuple(algorithms)

    print(
        f"Running terminal reward ablations: "
        f"{config.num_seeds} seeds, {config.num_episodes} episodes, "
        f"H={config.sequence_length}, V={config.vocab_size}, "
        f"B={config.batch_size}, K={config.k_candidates}, "
        f"algorithms={','.join(algo_list)}"
    )

    run_fns = _build_ablation_fns(config, algo_list)
    seeds = jnp.arange(config.num_seeds)
    results: dict[str, np.ndarray] = {}

    for algo in algo_list:
        print(f"  {algo}...", flush=True)
        data = run_fns[algo](seeds)  # (num_seeds, num_episodes, NUM_DIAGS)
        results[algo] = np.asarray(data)

    if verbose:
        _print_diagnostics_table(results, config.num_episodes, algo_list)

    data_path = _save_ablation_series_npz(results, config.save_dir)
    print(f"  Saved {data_path}")
    fig_path = _plot_ablation_results(results, config, config.save_dir)
    print(f"  Saved {fig_path}")

    # Build summary
    summary: dict[str, float] = {}
    for algo in algo_list:
        for i, name in enumerate(DIAG_NAMES):
            final_val = float(results[algo][:, -1, i].mean())
            summary[f"{algo}/{name}"] = final_val

    # Build curves for W&B
    series = dict(np.load(data_path))

    return ExperimentReport(
        name="terminal_reward_ablations",
        config=asdict(config),
        summary=summary,
        curves=(
            CurveReport(
                name="ablation_diagnostics",
                title="Terminal reward ablation diagnostics",
                x_name="episode",
                x_values=np.arange(config.num_episodes),
                series=series,
                plot_series=tuple(f"{a}_error_mean" for a in algo_list),
            ),
        ),
        artifact_paths=(fig_path, data_path),
    )


def _print_diagnostics_table(results, num_episodes, algos, interval=None):
    """Print summary diagnostics at regular intervals."""
    if interval is None:
        interval = max(1, num_episodes // 5)
    checkpoints = list(range(interval, num_episodes, interval))
    if num_episodes - 1 not in checkpoints:
        checkpoints.append(num_episodes - 1)

    # Header
    header = f"{'ep':>6}"
    for algo in algos:
        header += f"  {algo + ' err':>12}  {algo + ' ∇':>10}"
    print(f"\n  {header}")
    print(f"  {'─' * len(header)}")

    for ep in checkpoints:
        row = f"{ep + 1:>6}"
        for algo in algos:
            err = float(results[algo][:, ep, IDX_ERROR].mean())
            gn = float(results[algo][:, ep, IDX_GRAD_NORM].mean())
            row += f"  {err:>12.4f}  {gn:>10.4f}"
        print(f"  {row}")
    print()
