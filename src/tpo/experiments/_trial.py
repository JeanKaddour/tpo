"""Shared trial runner for transformer-based experiments.

This module keeps the transformer algorithms exposed by the experiment entry
points, including sequence-level grouped variants and dense token-candidate
variants.
"""

from functools import lru_cache
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from ..algorithms import (
    dg_loss,
    grpo_loss,
    ppo_loss,
    reinforce_loss,
    tpo_skill,
    tpo_target,
    tpo_target_no_anchor,
)
from ..config import TRANSFORMER_MODEL
from ..models import CausalTransformer


@lru_cache(maxsize=None)
def _build_trial_run_fns(
    algorithms: tuple[str, ...],
    sequence_length: int,
    vocab_size: int,
    num_episodes: int,
    batch_size: int,
    learning_rate: float,
    eta: float,
    ppo_epsilon: float,
    ppo_epochs: int,
    k_candidates: int | None,
    target_type: str,
    reward_type: str,
    tpo_eta: float = 1.0,
) -> dict[str, Callable]:
    """Build cached compiled runners for one transformer task configuration."""

    seq_len = 2 * sequence_length
    model = CausalTransformer(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=TRANSFORMER_MODEL.d_model,
        num_heads=TRANSFORMER_MODEL.num_heads,
        num_layers=TRANSFORMER_MODEL.num_layers,
        ffn_mult=TRANSFORMER_MODEL.ffn_mult,
    )

    def init_params(key):
        dummy = jnp.zeros((1, seq_len), dtype=jnp.int32)
        return model.init(key, dummy)

    def compute_targets(prompts):
        if target_type == "copy":
            return prompts
        if target_type == "flip":
            return (vocab_size - 1) - prompts
        if target_type == "reverse_copy":
            return prompts[:, ::-1]
        if target_type == "reverse_flip":
            return (vocab_size - 1) - prompts[:, ::-1]
        raise ValueError(f"Unknown target_type: {target_type}")

    def compute_rewards(actions, targets):
        correct = (actions == targets).astype(jnp.float32)
        if reward_type == "bag_of_tokens":
            return correct
        if reward_type == "sequential":
            return jnp.cumprod(correct, axis=1)
        if reward_type == "terminal":
            all_correct = jnp.all(correct, axis=1, keepdims=True).astype(jnp.float32)
            return jnp.broadcast_to(all_correct, correct.shape)
        raise ValueError(f"Unknown reward_type: {reward_type}")

    def rollout(params, key):
        key_prompt, key_sample = jr.split(key)
        prompts = jr.randint(key_prompt, (batch_size, sequence_length), 0, vocab_size)
        targets = compute_targets(prompts)

        def generate_step(carry, t):
            tokens, rng = carry
            rng_t, rng_next = jr.split(rng)
            all_log_probs = model.apply(params, tokens)
            pred_pos = sequence_length + t - 1
            token_log_probs = all_log_probs[:, pred_pos, :]
            sampled = jr.categorical(rng_t, token_log_probs)
            lp = token_log_probs[jnp.arange(batch_size), sampled]
            tokens = tokens.at[:, sequence_length + t].set(sampled)
            return (tokens, rng_next), (sampled, lp)

        token_buffer = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        token_buffer = token_buffer.at[:, :sequence_length].set(prompts)
        (_, _), (actions_stacked, log_probs_stacked) = jax.lax.scan(
            generate_step,
            (token_buffer, key_sample),
            jnp.arange(sequence_length),
        )
        actions = actions_stacked.T
        log_probs = log_probs_stacked.T
        rewards = compute_rewards(actions, targets)
        return prompts, actions, log_probs, rewards

    def compute_log_probs(params, prompts, actions):
        full_seq = jnp.concatenate([prompts, actions], axis=1)
        all_log_probs = model.apply(params, full_seq)
        pred_log_probs = all_log_probs[
            :, sequence_length - 1 : 2 * sequence_length - 1, :
        ]
        return pred_log_probs[
            jnp.arange(batch_size)[:, None],
            jnp.arange(sequence_length)[None, :],
            actions,
        ]

    from optax.contrib import muon as optax_muon

    optimizer = optax_muon(learning_rate=learning_rate * 10)

    def apply_epochs(params, opt_state, loss_fn, epochs):
        """Run a fixed number of optimizer epochs on one sampled batch."""

        def epoch_step(carry, _):
            p, os_ = carry
            _, grads = jax.value_and_grad(loss_fn)(p)
            updates, new_os = optimizer.update(grads, os_, p)
            new_p = optax.apply_updates(p, updates)
            return (new_p, new_os), None

        return jax.lax.scan(epoch_step, (params, opt_state), None, length=epochs)[0]

    def reinforce_step(params, opt_state, key):
        prompts, actions, _, rewards = rollout(params, key)
        advantages = rewards - rewards.mean(axis=0, keepdims=True)

        def loss_fn(p):
            return reinforce_loss(compute_log_probs(p, prompts, actions), advantages)

        new_params, new_opt_state = apply_epochs(
            params,
            opt_state,
            loss_fn,
            ppo_epochs,
        )
        return new_params, new_opt_state, 1.0 - rewards.mean()

    def dg_step(params, opt_state, key):
        prompts, actions, _, rewards = rollout(params, key)
        advantages = rewards - rewards.mean(axis=0, keepdims=True)

        def loss_fn(p):
            return dg_loss(compute_log_probs(p, prompts, actions), advantages, eta=eta)

        # Single epoch: multi-epoch DG diverges (no trust region).
        _, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, 1.0 - rewards.mean()

    def ppo_step(params, opt_state, key):
        prompts, actions, old_lp, rewards = rollout(params, key)
        advantages = rewards - rewards.mean(axis=0, keepdims=True)
        old_log_probs = jax.lax.stop_gradient(old_lp)

        def loss_fn(p):
            lp = compute_log_probs(p, prompts, actions)
            return ppo_loss(lp, old_log_probs, advantages, epsilon=ppo_epsilon)

        new_params, new_opt_state = apply_epochs(
            params,
            opt_state,
            loss_fn,
            ppo_epochs,
        )
        return new_params, new_opt_state, 1.0 - rewards.mean()

    def grouped_rollout(params, key, k_candidates):
        key_prompt, key_sample = jr.split(key)
        prompts = jr.randint(key_prompt, (batch_size, sequence_length), 0, vocab_size)
        targets = compute_targets(prompts)

        def single_rollout(key_k):
            def generate_step(carry, t):
                tokens, rng = carry
                rng_t, rng_next = jr.split(rng)
                all_log_probs = model.apply(params, tokens)
                pred_pos = sequence_length + t - 1
                token_log_probs = all_log_probs[:, pred_pos, :]
                sampled = jr.categorical(rng_t, token_log_probs)
                lp = token_log_probs[jnp.arange(batch_size), sampled]
                tokens = tokens.at[:, sequence_length + t].set(sampled)
                return (tokens, rng_next), (sampled, lp)

            token_buffer = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
            token_buffer = token_buffer.at[:, :sequence_length].set(prompts)
            (_, _), (acts, lps) = jax.lax.scan(
                generate_step,
                (token_buffer, key_k),
                jnp.arange(sequence_length),
            )
            actions_k = acts.T
            token_lp = lps.T
            token_reward = compute_rewards(actions_k, targets)
            return actions_k, token_lp, token_reward

        candidate_keys = jr.split(key_sample, k_candidates)
        all_actions, all_tok_lp, all_tok_reward = jax.vmap(single_rollout)(
            candidate_keys
        )
        return (
            prompts,
            jnp.transpose(all_actions, (1, 0, 2)),
            jnp.transpose(all_tok_lp, (1, 0, 2)),
            jnp.transpose(all_tok_reward, (1, 0, 2)),
        )

    def token_candidate_rollout(params, key, k_candidates):
        """Roll out one behavior trajectory plus K sampled token actions per state.

        This keeps the visited prefix states aligned with PPO/DG while giving
        grouped methods K sampled next-token candidates at each state.
        """
        if reward_type == "terminal":
            raise ValueError(
                "Token-candidate grouped methods are only defined for dense "
                "token rewards; use sequence-level tpo/grpo variants for terminal reward."
            )

        key_prompt, key_sample = jr.split(key)
        prompts = jr.randint(key_prompt, (batch_size, sequence_length), 0, vocab_size)
        targets = compute_targets(prompts)
        token_buffer = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        token_buffer = token_buffer.at[:, :sequence_length].set(prompts)
        alive0 = jnp.ones((batch_size,), dtype=jnp.float32)

        def sample_candidates(key_t, logits):
            sample = jr.categorical(key_t, logits, shape=(k_candidates,))
            return sample

        def generate_step(carry, t):
            tokens, alive, rng = carry
            rng_step, rng_next = jr.split(rng)
            all_log_probs = model.apply(params, tokens)
            pred_pos = sequence_length + t - 1
            token_log_probs = all_log_probs[:, pred_pos, :]
            batch_keys = jr.split(rng_step, batch_size)
            candidate_actions = jax.vmap(sample_candidates)(batch_keys, token_log_probs)
            candidate_log_probs = token_log_probs[
                jnp.arange(batch_size)[:, None],
                candidate_actions,
            ]

            target_t = targets[:, t]
            candidate_correct = (candidate_actions == target_t[:, None]).astype(
                jnp.float32
            )
            if reward_type == "bag_of_tokens":
                candidate_rewards = candidate_correct
            else:
                candidate_rewards = alive[:, None] * candidate_correct

            behavior_actions = candidate_actions[:, 0]
            behavior_rewards = candidate_rewards[:, 0]
            if reward_type == "sequential":
                next_alive = behavior_rewards
            else:
                next_alive = alive

            next_tokens = tokens.at[:, sequence_length + t].set(behavior_actions)
            return (
                next_tokens,
                next_alive,
                rng_next,
            ), (
                tokens,
                candidate_actions,
                candidate_log_probs,
                candidate_rewards,
                behavior_actions,
                behavior_rewards,
            )

        (_, _, _), outputs = jax.lax.scan(
            generate_step,
            (token_buffer, alive0, key_sample),
            jnp.arange(sequence_length),
        )
        (
            state_tokens,
            candidate_actions,
            candidate_log_probs,
            candidate_rewards,
            behavior_actions,
            behavior_rewards,
        ) = outputs
        return (
            prompts,
            jnp.transpose(state_tokens, (1, 0, 2)),
            jnp.transpose(candidate_actions, (1, 0, 2)),
            jnp.transpose(candidate_log_probs, (1, 0, 2)),
            jnp.transpose(candidate_rewards, (1, 0, 2)),
            behavior_actions.T,
            behavior_rewards.T,
        )

    def _flatten_grouped_sequences(prompts, all_actions, group_size):
        flat_prompts = jnp.repeat(prompts, group_size, axis=0)
        flat_actions = all_actions.reshape(batch_size * group_size, sequence_length)
        return flat_prompts, flat_actions

    def _sequence_candidate_log_probs(params, flat_prompts, flat_actions, group_size):
        full_seq = jnp.concatenate([flat_prompts, flat_actions], axis=1)
        all_log_probs = model.apply(params, full_seq)
        pred_lp = all_log_probs[:, sequence_length - 1 : 2 * sequence_length - 1, :]
        flat_tok_lp = pred_lp[
            jnp.arange(batch_size * group_size)[:, None],
            jnp.arange(sequence_length)[None, :],
            flat_actions,
        ]
        return flat_tok_lp.reshape(batch_size, group_size, sequence_length).sum(axis=2)

    def make_tpo_step(*, k_candidates, epochs, anchor_old_policy=True):

        def step(params, opt_state, key):
            prompts, all_actions, all_tok_lp, all_tok_reward = grouped_rollout(
                params,
                key,
                k_candidates,
            )

            episode_scores = all_tok_reward.mean(axis=2)
            if anchor_old_policy:
                log_scores_old = jax.lax.stop_gradient(all_tok_lp.sum(axis=2))
                q_target = jax.lax.stop_gradient(
                    tpo_target(
                        log_scores_old,
                        episode_scores,
                        eta=tpo_eta,
                    )
                )
            else:
                q_target = jax.lax.stop_gradient(
                    tpo_target_no_anchor(
                        episode_scores,
                        eta=tpo_eta,
                    )
                )
            flat_prompts, flat_actions = _flatten_grouped_sequences(
                prompts,
                all_actions,
                k_candidates,
            )

            def loss_fn(p):
                new_log_scores = _sequence_candidate_log_probs(
                    p,
                    flat_prompts,
                    flat_actions,
                    k_candidates,
                )
                new_log_p = jax.nn.log_softmax(
                    new_log_scores,
                    axis=1,
                )
                return -(q_target * new_log_p).sum(axis=1).mean()

            new_params, new_opt_state = apply_epochs(
                params,
                opt_state,
                loss_fn,
                epochs,
            )
            return new_params, new_opt_state, 1.0 - all_tok_reward.mean()

        return step

    def make_tpo_token_step(*, k_candidates, epochs):
        """Per-state token-candidate TPO for dense rewards."""

        def step(params, opt_state, key):
            (
                _prompts,
                state_tokens,
                candidate_actions,
                old_candidate_lp,
                candidate_rewards,
                _behavior_actions,
                behavior_rewards,
            ) = token_candidate_rollout(
                params,
                key,
                k_candidates,
            )
            flat_states = state_tokens.reshape(batch_size * sequence_length, seq_len)
            flat_candidate_actions = candidate_actions.reshape(
                batch_size * sequence_length, k_candidates
            )
            flat_old_candidate_lp = jax.lax.stop_gradient(
                old_candidate_lp.reshape(batch_size * sequence_length, k_candidates)
            )
            flat_candidate_rewards = candidate_rewards.reshape(
                batch_size * sequence_length, k_candidates
            )
            step_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :],
                (batch_size, sequence_length),
            ).reshape(batch_size * sequence_length)
            q_target = jax.lax.stop_gradient(
                tpo_target(
                    flat_old_candidate_lp,
                    flat_candidate_rewards,
                    eta=tpo_eta,
                )
            )

            def loss_fn(p):
                all_log_probs = model.apply(p, flat_states)
                token_log_probs = all_log_probs[
                    jnp.arange(batch_size * sequence_length),
                    sequence_length + step_ids - 1,
                    :,
                ]
                new_candidate_lp = token_log_probs[
                    jnp.arange(batch_size * sequence_length)[:, None],
                    flat_candidate_actions,
                ]
                new_log_p = jax.nn.log_softmax(new_candidate_lp, axis=-1)
                return -(q_target * new_log_p).sum(axis=-1).mean()

            new_params, new_opt_state = apply_epochs(
                params,
                opt_state,
                loss_fn,
                epochs,
            )
            return new_params, new_opt_state, 1.0 - behavior_rewards.mean()

        return step

    _k = (
        k_candidates
        if k_candidates is not None
        else (32 if reward_type == "terminal" else 16)
    )
    tpo_step = make_tpo_step(
        k_candidates=_k,
        epochs=ppo_epochs,
    )
    tpo_no_anchor_step = make_tpo_step(
        k_candidates=_k,
        epochs=ppo_epochs,
        anchor_old_policy=False,
    )
    tpo_token_step = make_tpo_token_step(
        k_candidates=_k,
        epochs=ppo_epochs,
    )

    def make_group_pg_step(*, k_candidates, epochs):
        """Scalar-weighted grouped policy gradient using the same grouped signal as TPO."""

        def step(params, opt_state, key):
            prompts, all_actions, _all_tok_lp, all_tok_reward = grouped_rollout(
                params,
                key,
                k_candidates,
            )
            group_scores = jax.lax.stop_gradient(
                tpo_skill(all_tok_reward.mean(axis=2)) / tpo_eta
            )
            flat_prompts, flat_actions = _flatten_grouped_sequences(
                prompts,
                all_actions,
                k_candidates,
            )

            def loss_fn(p):
                new_log_scores = _sequence_candidate_log_probs(
                    p,
                    flat_prompts,
                    flat_actions,
                    k_candidates,
                )
                return reinforce_loss(new_log_scores, group_scores)

            new_params, new_opt_state = apply_epochs(
                params,
                opt_state,
                loss_fn,
                epochs,
            )
            return new_params, new_opt_state, 1.0 - all_tok_reward.mean()

        return step

    def make_grpo_step(*, k_candidates, epochs, epsilon=0.2, beta=0.04):
        """Factory for GRPO step: K candidates, group-relative advantages,
        clipped surrogate with optional KL penalty, multiple gradient epochs."""

        def step(params, opt_state, key):
            prompts, all_actions, all_tok_lp, all_tok_reward = grouped_rollout(
                params,
                key,
                k_candidates,
            )

            # Group-relative advantages: normalise per-prompt across K candidates
            episode_scores = all_tok_reward.mean(axis=2)  # (B, K)
            group_mean = episode_scores.mean(axis=1, keepdims=True)
            group_std = episode_scores.std(axis=1, keepdims=True)
            advantages = jax.lax.stop_gradient(
                (episode_scores - group_mean) / (group_std + 1e-8)
            )  # (B, K)

            flat_prompts, flat_actions = _flatten_grouped_sequences(
                prompts,
                all_actions,
                k_candidates,
            )
            old_log_scores = jax.lax.stop_gradient(all_tok_lp.sum(axis=2))

            def loss_fn(p):
                new_log_scores = _sequence_candidate_log_probs(
                    p,
                    flat_prompts,
                    flat_actions,
                    k_candidates,
                )
                return grpo_loss(
                    new_log_scores,
                    old_log_scores,
                    advantages,
                    epsilon=epsilon,
                    beta=beta,
                )

            new_params, new_opt_state = apply_epochs(
                params,
                opt_state,
                loss_fn,
                epochs,
            )
            return new_params, new_opt_state, 1.0 - all_tok_reward.mean()

        return step

    def make_grpo_token_step(*, k_candidates, epochs, epsilon=0.2, beta=0.04):
        """Per-state token-candidate GRPO with optional KL penalty."""

        def step(params, opt_state, key):
            (
                _prompts,
                state_tokens,
                candidate_actions,
                old_candidate_lp,
                candidate_rewards,
                _behavior_actions,
                behavior_rewards,
            ) = token_candidate_rollout(
                params,
                key,
                k_candidates,
            )
            group_mean = candidate_rewards.mean(axis=2, keepdims=True)
            group_std = candidate_rewards.std(axis=2, keepdims=True)
            advantages = jax.lax.stop_gradient(
                (candidate_rewards - group_mean) / (group_std + 1e-8)
            )
            flat_states = state_tokens.reshape(batch_size * sequence_length, seq_len)
            flat_candidate_actions = candidate_actions.reshape(
                batch_size * sequence_length, k_candidates
            )
            flat_advantages = advantages.reshape(
                batch_size * sequence_length, k_candidates
            )
            flat_old_candidate_lp = jax.lax.stop_gradient(
                old_candidate_lp.reshape(batch_size * sequence_length, k_candidates)
            )
            step_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :],
                (batch_size, sequence_length),
            ).reshape(batch_size * sequence_length)

            def loss_fn(p):
                all_log_probs = model.apply(p, flat_states)
                token_log_probs = all_log_probs[
                    jnp.arange(batch_size * sequence_length),
                    sequence_length + step_ids - 1,
                    :,
                ]
                new_candidate_lp = token_log_probs[
                    jnp.arange(batch_size * sequence_length)[:, None],
                    flat_candidate_actions,
                ]
                return grpo_loss(
                    new_candidate_lp,
                    flat_old_candidate_lp,
                    flat_advantages,
                    epsilon=epsilon,
                    beta=beta,
                )

            new_params, new_opt_state = apply_epochs(
                params,
                opt_state,
                loss_fn,
                epochs,
            )
            return new_params, new_opt_state, 1.0 - behavior_rewards.mean()

        return step

    grpo_step = make_grpo_step(
        k_candidates=_k,
        epochs=ppo_epochs,
        epsilon=ppo_epsilon,
        beta=0.04,
    )
    grpo_no_kl_step = make_grpo_step(
        k_candidates=_k,
        epochs=ppo_epochs,
        epsilon=ppo_epsilon,
        beta=0.0,
    )
    group_pg_step = make_group_pg_step(
        k_candidates=_k,
        epochs=ppo_epochs,
    )
    grpo_token_step = make_grpo_token_step(
        k_candidates=_k,
        epochs=ppo_epochs,
        epsilon=ppo_epsilon,
        beta=0.04,
    )
    grpo_token_no_kl_step = make_grpo_token_step(
        k_candidates=_k,
        epochs=ppo_epochs,
        epsilon=ppo_epsilon,
        beta=0.0,
    )

    available = {
        "reinforce": reinforce_step,
        "dg": dg_step,
        "ppo": ppo_step,
        "tpo": tpo_step,
        "tpo_no_anchor": tpo_no_anchor_step,
        "tpo_token": tpo_token_step,
        "group_pg": group_pg_step,
        "grpo": grpo_step,
        "grpo_no_kl": grpo_no_kl_step,
        "grpo_token": grpo_token_step,
        "grpo_token_no_kl": grpo_token_no_kl_step,
    }
    unknown = [name for name in algorithms if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown transformer algorithm(s): {unknown}. Available: {list(available)}"
        )

    def run_seed(step_fn, seed):
        key = jr.PRNGKey(seed)
        params = init_params(key)
        opt_state = optimizer.init(params)

        def scan_step(carry, t):
            p, os_ = carry
            key_t = jr.fold_in(key, t)
            new_p, new_os, error = step_fn(p, os_, key_t)
            return (new_p, new_os), error

        (_, _), errors = jax.lax.scan(
            scan_step, (params, opt_state), jnp.arange(num_episodes)
        )
        return errors

    return {
        algo_name: jax.jit(
            jax.vmap(lambda seed, _sf=available[algo_name]: run_seed(_sf, seed))
        )
        for algo_name in algorithms
    }


def run_trial(
    *,
    num_seeds,
    sequence_length,
    vocab_size,
    num_episodes,
    batch_size,
    learning_rate,
    eta,
    ppo_epsilon,
    ppo_epochs,
    k_candidates=None,
    target_type,
    reward_type,
    algorithms,
    label="",
    seed_offset=0,
    tpo_eta=1.0,
):
    """Run core transformer algorithms on one task variant.

    Returns a dict mapping algorithm name to an array of shape
    ``(num_seeds, num_episodes)`` containing the per-episode error curve.
    """

    requested_algorithms = tuple(algorithms)
    run_fns = _build_trial_run_fns(
        requested_algorithms,
        sequence_length,
        vocab_size,
        num_episodes,
        batch_size,
        learning_rate,
        eta,
        ppo_epsilon,
        ppo_epochs,
        k_candidates,
        target_type,
        reward_type,
        tpo_eta=tpo_eta,
    )
    seeds = jnp.arange(seed_offset, seed_offset + num_seeds)
    results = {}
    for algo_name in requested_algorithms:
        print(f"    {label}{algo_name}...")
        results[algo_name] = run_fns[algo_name](seeds)
    return results
