import numpy as np
import jax.numpy as jnp
from jax import random as jr

from tpo.algorithms import (
    classification_group_pg_loss,
    classification_group_pg_loss_from_feedback,
    classification_tpo_loss,
    classification_tpo_loss_from_feedback,
    sample_classification_bandit,
    tpo_skill,
    tpo_target,
    tpo_target_no_anchor,
)


def test_tpo_skill_zero_variance_returns_centered_scores() -> None:
    scores = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)
    skill = np.asarray(tpo_skill(scores))
    np.testing.assert_allclose(skill, np.zeros_like(scores))


def test_tpo_target_is_normalized_and_anchor_sensitive() -> None:
    log_scores_old = np.log(np.array([[0.7, 0.2, 0.1]], dtype=np.float32))
    scores = np.array([[1.0, 0.0, -1.0]], dtype=np.float32)

    target = np.asarray(tpo_target(jnp.asarray(log_scores_old), scores))
    no_anchor = np.asarray(tpo_target_no_anchor(scores))

    np.testing.assert_allclose(
        target.sum(axis=1), np.array([1.0], dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        no_anchor.sum(axis=1), np.array([1.0], dtype=np.float32), atol=1e-6
    )
    assert target[0, 0] > no_anchor[0, 0]


def test_classification_tpo_loss_matches_shared_feedback_path() -> None:
    logits = np.array(
        [[2.0, 0.5, -1.0], [0.1, -0.2, 1.3]],
        dtype=np.float32,
    )
    labels = np.array([0, 2], dtype=np.int32)
    key = jr.PRNGKey(0)

    batch = sample_classification_bandit(logits, labels, key)
    direct = np.asarray(classification_tpo_loss(logits, labels, key, eta=0.7))
    shared = np.asarray(
        classification_tpo_loss_from_feedback(
            batch.log_probs,
            batch.probs,
            batch.actions,
            batch.rewards,
            eta=0.7,
        )
    )

    np.testing.assert_allclose(direct, shared, atol=1e-6)


def test_classification_group_pg_loss_matches_shared_feedback_path() -> None:
    logits = np.array(
        [[1.5, -0.5, 0.0], [0.2, 0.7, -1.3]],
        dtype=np.float32,
    )
    labels = np.array([1, 0], dtype=np.int32)
    key = jr.PRNGKey(1)

    batch = sample_classification_bandit(logits, labels, key)
    direct = np.asarray(classification_group_pg_loss(logits, labels, key))
    shared = np.asarray(
        classification_group_pg_loss_from_feedback(
            batch.log_probs,
            batch.probs,
            batch.actions,
            batch.rewards,
        )
    )

    np.testing.assert_allclose(direct, shared, atol=1e-6)
