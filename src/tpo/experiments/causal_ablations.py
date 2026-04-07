"""Causal ablation experiments for the TPO analysis section.

Three ablations that provide causal (not just correlational) evidence:
  1. Epoch sweep:  TPO vs GRPO across {1,2,4,8,16} gradient epochs
  2. K sweep:      TPO vs GRPO across {4,8,16,32,64} candidates
  3. GRPO fix:     GRPO with zero-variance group masking

Each ablation saves results as .npz and logs to a separate W&B run.

Run locally:
    uv run python -m tpo.cli causal_ablations --smoke
"""

from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .._typing import NumpyArray, savez_dict
from ..config import TRANSFORMER_MODEL, CausalAblationsConfig, coerce_config
from ..models import CausalTransformer
from ..tracking import CurveReport, ExperimentReport
from ._trial import run_trial

# ── Shared constants ────────────────────────────────────────────────

_ALGO_STYLES = {
    "tpo": "TPO",
    "grpo": "GRPO",
    "grpo_masked": "GRPO (masked)",
}


def _save_npz(save_dir: str, name: str, data: dict[str, NumpyArray]) -> Path:
    """Save results as npz and return the path."""
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    path = p / f"{name}.npz"
    savez_dict(path, data)
    print(f"  Saved {path}")
    return path


def _build_report(
    name: str,
    config: CausalAblationsConfig,
    all_results: dict[str, np.ndarray],
    artifact_paths: tuple[Path, ...],
) -> ExperimentReport:
    """Build an ExperimentReport from collected error curves."""
    summary: dict[str, float] = {}
    series: dict[str, np.ndarray] = {}

    for key, errors in all_results.items():
        # errors: (num_seeds, num_episodes)
        mean = errors.mean(axis=0)
        se = errors.std(axis=0) / np.sqrt(errors.shape[0])
        summary[f"{key}/final_error"] = float(mean[-1])
        series[f"{key}_mean"] = mean
        series[f"{key}_se"] = se

    num_episodes = next(iter(all_results.values())).shape[1]
    return ExperimentReport(
        name=name,
        config=asdict(config),
        summary=summary,
        curves=(
            CurveReport(
                name=name,
                title=name.replace("_", " ").title(),
                x_name="episode",
                x_values=np.arange(num_episodes),
                series=series,
                plot_series=tuple(f"{k}_mean" for k in all_results),
            ),
        ),
        artifact_paths=artifact_paths,
    )


# ════════════════════════════════════════════════════════════════════
#  Ablation 1: Epoch sweep
# ════════════════════════════════════════════════════════════════════

EPOCH_VALUES = (1, 2, 4, 8, 16)


def run_epoch_sweep(cfg: CausalAblationsConfig) -> tuple[dict, Path]:
    """Run TPO and GRPO at each epoch count, return {label: errors} + npz path."""
    print(f"\n  ── Epoch sweep ({'×'.join(str(e) for e in EPOCH_VALUES)}) ──")
    results: dict[str, np.ndarray] = {}

    for epochs in EPOCH_VALUES:
        for algo in ("tpo", "grpo"):
            label = f"{algo}_ep{epochs}"
            print(f"    {label}...", flush=True)
            trial = run_trial(
                num_seeds=cfg.num_seeds,
                sequence_length=cfg.sequence_length,
                vocab_size=cfg.vocab_size,
                num_episodes=cfg.num_episodes,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                eta=cfg.eta,
                ppo_epsilon=cfg.ppo_epsilon,
                ppo_epochs=epochs,
                k_candidates=cfg.k_candidates,
                target_type="reverse_copy",
                reward_type="terminal",
                algorithms=(algo,),
                label=f"epoch_sweep/{label}: ",
                tpo_eta=cfg.tpo_eta,
            )
            results[label] = np.asarray(trial[algo])

    path = _save_npz(cfg.save_dir, "ablation_epoch_sweep", results)
    return results, path


# ════════════════════════════════════════════════════════════════════
#  Ablation 2: K sweep
# ════════════════════════════════════════════════════════════════════

K_VALUES = (4, 8, 16, 32, 64)


def run_k_sweep(cfg: CausalAblationsConfig) -> tuple[dict, Path]:
    """Run TPO and GRPO at each K, return {label: errors} + npz path."""
    print(f"\n  ── K sweep ({'×'.join(str(k) for k in K_VALUES)}) ──")
    results: dict[str, np.ndarray] = {}

    for k in K_VALUES:
        for algo in ("tpo", "grpo"):
            label = f"{algo}_k{k}"
            print(f"    {label}...", flush=True)
            trial = run_trial(
                num_seeds=cfg.num_seeds,
                sequence_length=cfg.sequence_length,
                vocab_size=cfg.vocab_size,
                num_episodes=cfg.num_episodes,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                eta=cfg.eta,
                ppo_epsilon=cfg.ppo_epsilon,
                ppo_epochs=cfg.ppo_epochs,
                k_candidates=k,
                target_type="reverse_copy",
                reward_type="terminal",
                algorithms=(algo,),
                label=f"k_sweep/{label}: ",
                tpo_eta=cfg.tpo_eta,
            )
            results[label] = np.asarray(trial[algo])

    path = _save_npz(cfg.save_dir, "ablation_k_sweep", results)
    return results, path


# ════════════════════════════════════════════════════════════════════
#  Ablation 3: GRPO with zero-variance masking
# ════════════════════════════════════════════════════════════════════


def _build_grpo_masked_fns(cfg: CausalAblationsConfig):
    """Build a GRPO variant that masks zero-variance groups."""
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
        return prompts[:, ::-1]

    def compute_rewards(actions, targets):
        correct = (actions == targets).astype(jnp.float32)
        all_correct = jnp.all(correct, axis=1, keepdims=True).astype(jnp.float32)
        return jnp.broadcast_to(all_correct, correct.shape)

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
            jnp.transpose(all_a, (1, 0, 2)),
            jnp.transpose(all_lp, (1, 0, 2)),
            jnp.transpose(all_r, (1, 0, 2)),
        )

    def grpo_masked_step(params, opt_state, key):
        prompts, all_actions, all_tok_lp, all_tok_reward = grouped_rollout(params, key)

        episode_scores = all_tok_reward.mean(axis=2)  # (B, K)
        group_mean = episode_scores.mean(axis=1, keepdims=True)
        group_std = episode_scores.std(axis=1, keepdims=True)

        # Mask: only include groups with nonzero variance
        has_variance = (group_std.squeeze(-1) > 1e-6).astype(jnp.float32)  # (B,)
        n_informative = has_variance.sum().clip(1)

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

            # Standard GRPO loss components on full-sequence candidates
            adv = jax.lax.stop_gradient(advantages)
            old_lp = old_log_scores
            log_ratio = jnp.clip(new_log_scores - old_lp, -20.0, 20.0)
            ratio = jnp.exp(log_ratio)
            kl = jnp.exp(-log_ratio) - (-log_ratio) - 1
            clipped = jnp.clip(ratio, 1.0 - cfg.ppo_epsilon, 1.0 + cfg.ppo_epsilon)
            surrogate = jnp.minimum(ratio * adv, clipped * adv)
            per_group = (surrogate - 0.04 * kl).sum(axis=1)  # (B,)

            # Apply mask: zero out loss from zero-variance groups
            masked = per_group * has_variance
            return -(masked.sum() / n_informative)

        def epoch_step(carry, _):
            p, os_ = carry
            _, g = jax.value_and_grad(loss_fn)(p)
            updates, new_os = optimizer.update(g, os_, p)
            new_p = optax.apply_updates(p, updates)
            return (new_p, new_os), None

        (new_params, new_opt_state), _ = jax.lax.scan(
            epoch_step, (params, opt_state), None, length=cfg.ppo_epochs
        )
        return new_params, new_opt_state, 1.0 - all_tok_reward.mean()

    def run_seed(seed):
        key = jr.PRNGKey(seed)
        params = init_params(key)
        opt_state = optimizer.init(params)

        def scan_step(carry, t):
            p, os_ = carry
            key_t = jr.fold_in(key, t)
            new_p, new_os, error = grpo_masked_step(p, os_, key_t)
            return (new_p, new_os), error

        (_, _), errors = jax.lax.scan(
            scan_step, (params, opt_state), jnp.arange(cfg.num_episodes)
        )
        return errors

    return jax.jit(jax.vmap(run_seed))


def run_grpo_fix(cfg: CausalAblationsConfig) -> tuple[dict, Path]:
    """Run GRPO, GRPO-masked, and TPO for comparison."""
    print("\n  ── GRPO zero-variance masking ──")
    results: dict[str, np.ndarray] = {}

    # Standard TPO and GRPO baselines
    for algo in ("tpo", "grpo"):
        print(f"    {algo}...", flush=True)
        trial = run_trial(
            num_seeds=cfg.num_seeds,
            sequence_length=cfg.sequence_length,
            vocab_size=cfg.vocab_size,
            num_episodes=cfg.num_episodes,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            eta=cfg.eta,
            ppo_epsilon=cfg.ppo_epsilon,
            ppo_epochs=cfg.ppo_epochs,
            k_candidates=cfg.k_candidates,
            target_type="reverse_copy",
            reward_type="terminal",
            algorithms=(algo,),
            label=f"grpo_fix/{algo}: ",
            tpo_eta=cfg.tpo_eta,
        )
        results[algo] = np.asarray(trial[algo])

    # GRPO with zero-variance masking
    print("    grpo_masked...", flush=True)
    run_fn = _build_grpo_masked_fns(cfg)
    seeds = jnp.arange(cfg.num_seeds)
    results["grpo_masked"] = np.asarray(run_fn(seeds))

    path = _save_npz(cfg.save_dir, "ablation_grpo_fix", results)
    return results, path


# ════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════


ABLATION_NAMES = ("epoch_sweep", "k_sweep", "grpo_fix")


def run_causal_ablations(
    config: CausalAblationsConfig | None = None,
    /,
    algorithms: tuple[str, ...] | None = None,
    verbose: bool = True,
    **overrides,
) -> ExperimentReport:
    """Run causal ablation experiments.

    Parameters
    ----------
    algorithms : which ablations to run.  Accepts any subset of
        ``("epoch_sweep", "k_sweep", "grpo_fix")``.
        Default (None) runs all three.
    """
    config = coerce_config(config, CausalAblationsConfig, overrides)
    ablations = ABLATION_NAMES if algorithms is None else tuple(algorithms)

    print(
        f"Running causal ablations: "
        f"{config.num_seeds} seeds, {config.num_episodes} episodes, "
        f"H={config.sequence_length}, V={config.vocab_size}, "
        f"B={config.batch_size}, K={config.k_candidates}, "
        f"ablations={','.join(ablations)}"
    )

    all_results: dict[str, np.ndarray] = {}
    artifact_paths: list[Path] = []

    if "epoch_sweep" in ablations:
        epoch_results, epoch_path = run_epoch_sweep(config)
        all_results.update(epoch_results)
        artifact_paths.append(epoch_path)

    if "k_sweep" in ablations:
        k_results, k_path = run_k_sweep(config)
        all_results.update(k_results)
        artifact_paths.append(k_path)

    if "grpo_fix" in ablations:
        fix_results, fix_path = run_grpo_fix(config)
        all_results.update({f"fix_{k}": v for k, v in fix_results.items()})
        artifact_paths.append(fix_path)

    if verbose and all_results:
        print("\n  ── Final errors ──")
        for key in sorted(all_results):
            err = float(all_results[key].mean(axis=0)[-1])
            print(f"    {key:>25s}: {err:.4f}")

    return _build_report(
        "causal_ablations",
        config,
        all_results,
        tuple(artifact_paths),
    )
