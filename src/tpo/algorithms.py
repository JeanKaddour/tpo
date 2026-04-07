"""Shared policy-gradient losses used by the remaining experiments."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .core import dg_gate


class ClassificationBanditFeedback(NamedTuple):
    """Sampled one-step classification-bandit feedback."""

    log_probs: jax.Array
    probs: jax.Array
    actions: jax.Array
    rewards: jax.Array


def _broadcast_like(log_probs, values):
    """Broadcast 1-D per-episode values across a 2-D log-prob tensor."""
    values = jnp.asarray(values)
    if log_probs.ndim == 2 and values.ndim == 1:
        return values[:, None]
    return values


def _compute_surprisal(log_probs, surprisal_clip=None):
    """Return per-sample surprisal, optionally clipped for stability."""
    surprisal = -log_probs
    if surprisal_clip is not None:
        surprisal = jnp.clip(surprisal, -surprisal_clip, surprisal_clip)
    return surprisal


def sampled_action_log_probs(log_probs, actions):
    """Gather per-sample log-probabilities for integer actions."""

    return jnp.take_along_axis(
        log_probs, jnp.asarray(actions)[..., None], axis=-1
    ).squeeze(-1)


def sampled_classification_advantage(probs, rewards):
    """One-step bandit advantage with the paper's expected baseline."""

    baseline = jnp.sum(jnp.asarray(probs) ** 2, axis=-1)
    return jnp.asarray(rewards) - baseline


def sample_classification_bandit(logits, labels, key):
    """Sample one-step bandit feedback from classification logits."""

    log_probs = jax.nn.log_softmax(logits)
    probs = jax.nn.softmax(logits)
    actions = jax.random.categorical(key, logits)
    rewards = (actions == jnp.asarray(labels)).astype(jnp.float32)
    return ClassificationBanditFeedback(
        log_probs=log_probs,
        probs=probs,
        actions=actions,
        rewards=rewards,
    )


def one_hot_score_matrix(actions, values, num_classes):
    """Return score vectors with one active coordinate per sampled action."""

    return jnp.asarray(values)[:, None] * jax.nn.one_hot(actions, num_classes)


def classification_pg_loss_from_feedback(log_probs, probs, actions, rewards):
    """REINFORCE loss for sampled classification-bandit feedback."""

    advantages = sampled_classification_advantage(probs, rewards)
    logp_a = sampled_action_log_probs(log_probs, actions)
    return reinforce_loss(logp_a, advantages)


def classification_dg_loss_from_feedback(log_probs, probs, actions, rewards, eta=1.0):
    """DG loss for sampled classification-bandit feedback."""

    advantages = sampled_classification_advantage(probs, rewards)
    logp_a = sampled_action_log_probs(log_probs, actions)
    return dg_loss(logp_a, advantages, eta=eta)


def classification_tpo_loss_from_feedback(log_probs, probs, actions, rewards, eta=1.0):
    """Anchored TPO loss for sampled classification-bandit feedback."""

    advantages = sampled_classification_advantage(probs, rewards)
    scores = one_hot_score_matrix(actions, advantages, log_probs.shape[-1])
    return anchored_tpo_cross_entropy_loss(log_probs, scores, eta=eta)


def classification_group_pg_loss_from_feedback(log_probs, probs, actions, rewards):
    """Grouped scalar-weighted PG ablation for classification-bandit feedback."""

    advantages = sampled_classification_advantage(probs, rewards)
    scores = one_hot_score_matrix(actions, advantages, log_probs.shape[-1])
    return group_pg_cross_entropy_loss(log_probs, actions, scores)


def classification_grpo_loss_from_feedback(log_probs, actions, rewards, eps=1e-8):
    """Batch-standardized PG baseline for classification-bandit feedback."""

    return batch_standardized_pg_loss(log_probs, actions, rewards, eps=eps)


def classification_pg_loss(logits, labels, key):
    """REINFORCE loss for one-step classification bandits."""

    batch = sample_classification_bandit(logits, labels, key)
    return classification_pg_loss_from_feedback(
        batch.log_probs,
        batch.probs,
        batch.actions,
        batch.rewards,
    )


def classification_dg_loss(logits, labels, key, eta=1.0):
    """DG loss for one-step classification bandits."""

    batch = sample_classification_bandit(logits, labels, key)
    return classification_dg_loss_from_feedback(
        batch.log_probs,
        batch.probs,
        batch.actions,
        batch.rewards,
        eta=eta,
    )


def classification_tpo_loss(logits, labels, key, eta=1.0):
    """Anchored TPO loss for one-step classification bandits."""

    batch = sample_classification_bandit(logits, labels, key)
    return classification_tpo_loss_from_feedback(
        batch.log_probs,
        batch.probs,
        batch.actions,
        batch.rewards,
        eta=eta,
    )


def classification_group_pg_loss(logits, labels, key):
    """Grouped scalar-weighted PG ablation for one-step classification bandits."""

    batch = sample_classification_bandit(logits, labels, key)
    return classification_group_pg_loss_from_feedback(
        batch.log_probs,
        batch.probs,
        batch.actions,
        batch.rewards,
    )


def classification_grpo_loss(logits, labels, key, eps=1e-8):
    """Batch-standardized PG baseline for one-step classification bandits."""

    batch = sample_classification_bandit(logits, labels, key)
    return classification_grpo_loss_from_feedback(
        batch.log_probs,
        batch.actions,
        batch.rewards,
        eps=eps,
    )


def reinforce_loss(log_probs, advantages):
    """REINFORCE loss for flat or sequential log-prob tensors."""
    adv = jax.lax.stop_gradient(_broadcast_like(log_probs, advantages))
    if log_probs.ndim == 2:
        return -((adv * log_probs).sum(axis=1)).mean()
    return -(adv * log_probs).mean()


def dg_loss(log_probs, advantages, eta=1.0, surprisal_clip=None):
    """Delightful Policy Gradient loss."""
    adv = _broadcast_like(log_probs, advantages)
    surprisal = _compute_surprisal(log_probs, surprisal_clip=surprisal_clip)
    weights = jax.lax.stop_gradient(dg_gate(adv, surprisal, eta=eta) * adv)
    if log_probs.ndim == 2:
        return -((weights * log_probs).sum(axis=1)).mean()
    return -(weights * log_probs).mean()


def ppo_loss(log_probs, old_log_probs, advantages, epsilon=0.2):
    """PPO clipped surrogate loss."""
    adv = jax.lax.stop_gradient(advantages)
    old_lp = jax.lax.stop_gradient(old_log_probs)
    ratio = jnp.exp(log_probs - old_lp)

    if log_probs.ndim == 2:
        adv_broad = adv if adv.ndim == 2 else adv[:, None]
        clipped = jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
        return -(jnp.minimum(ratio * adv_broad, clipped * adv_broad).sum(axis=1)).mean()

    clipped = jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    return -jnp.minimum(ratio * adv, clipped * adv).mean()


def grpo_loss(log_probs, old_log_probs, advantages, epsilon=0.2, beta=0.04):
    """GRPO: clipped surrogate with optional reverse-KL penalty.

    Implements the grouped clipped policy-gradient loss used by the transformer
    experiments. ``beta=0`` disables the KL penalty entirely, yielding the
    no-KL variant while keeping PPO-style clipping. Advantages are expected to
    be group-relative (normalised per-prompt across candidates).
    """
    adv = jax.lax.stop_gradient(advantages)
    old_lp = jax.lax.stop_gradient(old_log_probs)
    # Clamp log-ratio to prevent Inf/NaN from stale off-policy data
    log_ratio = jnp.clip(log_probs - old_lp, -20.0, 20.0)
    ratio = jnp.exp(log_ratio)
    # Reverse KL approximation: exp(q - p) - (q - p) - 1
    kl = jnp.exp(-log_ratio) - (-log_ratio) - 1

    if log_probs.ndim == 2:
        adv_broad = adv if adv.ndim == 2 else adv[:, None]
        clipped = jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
        surrogate = jnp.minimum(ratio * adv_broad, clipped * adv_broad)
        return -((surrogate - beta * kl).sum(axis=1)).mean()

    clipped = jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    surrogate = jnp.minimum(ratio * adv, clipped * adv)
    return -(surrogate - beta * kl).mean()


def tpo_skill(scores):
    """Compute the whitened TPO skill over the last dimension."""
    scores = jnp.asarray(scores)
    centered = scores - scores.mean(axis=-1, keepdims=True)
    std = centered.std(axis=-1, keepdims=True)
    safe_std = jnp.where(std > 1e-6, std, 1.0)
    return jnp.where(std > 1e-6, centered / safe_std, centered)


def tpo_target(log_scores_old, scores, eta=1.0):
    """Compute the TPO mirror-descent target distribution.

    eta scales the whitened scores: skill / eta.  eta=1 (the default) uses
    z-scored scores directly; smaller eta sharpens, larger eta flattens.
    """
    skill = tpo_skill(scores)
    return jax.nn.softmax(
        jax.nn.log_softmax(log_scores_old, axis=-1) + skill / eta, axis=-1
    )


def tpo_target_no_anchor(scores, eta=1.0):
    """Compute the no-anchor TPO target on the sampled candidate simplex."""
    skill = tpo_skill(scores)
    return jax.nn.softmax(skill / eta, axis=-1)


def anchored_tpo_cross_entropy_loss(log_probs, scores, eta=1.0):
    """Cross-entropy loss to the anchored TPO target for one-step bandits."""

    q = jax.lax.stop_gradient(
        tpo_target(jax.lax.stop_gradient(log_probs), scores, eta=eta)
    )
    return -(q * log_probs).sum(axis=-1).mean()


def group_pg_cross_entropy_loss(log_probs, actions, scores):
    """Scalar-weighted policy-gradient ablation using TPO's grouped signal."""

    skill = jax.lax.stop_gradient(tpo_skill(scores))
    action_skill = sampled_action_log_probs(skill, actions)
    logp_a = sampled_action_log_probs(log_probs, actions)
    return reinforce_loss(logp_a, action_skill)


def batch_standardized_pg_loss(log_probs, actions, rewards, eps=1e-8):
    """Single-sample GRPO-like bandit loss with batch-standardized rewards."""

    logp_a = sampled_action_log_probs(log_probs, actions)
    rewards = jnp.asarray(rewards)
    advantages = jax.lax.stop_gradient(
        (rewards - rewards.mean()) / (rewards.std() + eps)
    )
    return -(advantages * logp_a).mean()
