"""Shared utilities: DG gate, cosine misalignment, pytree helpers."""

import jax
import jax.numpy as jnp


def dg_gate(advantage, surprisal, eta=1.0):
    """Compute the Delightful gate w = σ(U·ℓ / η).

    Args:
        advantage: per-sample advantages U.
        surprisal: per-sample action surprisal ℓ = -log π(a).
        eta: temperature parameter (default 1.0).

    Returns:
        Gate values in [0, 1].
    """
    advantage = jnp.asarray(advantage)
    surprisal = jnp.asarray(surprisal)
    chi = advantage * surprisal  # delight
    return jax.nn.sigmoid(chi / eta)


def cosine_misalignment(g1, g2):
    """Return 1 - cos(g1, g2) for flat vectors."""
    dot = jnp.sum(g1 * g2)
    norm1 = jnp.sqrt(jnp.sum(g1**2) + 1e-12)
    norm2 = jnp.sqrt(jnp.sum(g2**2) + 1e-12)
    cos = dot / (norm1 * norm2)
    return 1.0 - cos


def flatten_pytree(pytree):
    """Flatten a JAX pytree to a single 1-D array."""
    leaves = jax.tree_util.tree_leaves(pytree)
    if not leaves:
        return jnp.array([], dtype=jnp.float32)
    return jnp.concatenate([jnp.ravel(jnp.asarray(leaf)) for leaf in leaves])
