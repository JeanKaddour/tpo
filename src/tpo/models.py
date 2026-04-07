"""Flax neural network modules for MNIST and Transformer experiments."""

import flax.linen as nn

from ._typing import Array, MetricSeries


class PolicyMLP(nn.Module):
    """MLP policy for MNIST contextual bandit: images → logits."""

    hidden_sizes: tuple[int, ...] = (50, 50)
    num_actions: int = 10

    @nn.compact
    def __call__(self, x: MetricSeries) -> Array:
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        return nn.Dense(self.num_actions)(x)


class CausalTransformer(nn.Module):
    """Decoder-only Transformer for Token Reversal.

    Pre-norm architecture: LayerNorm → Attention → residual,
                          LayerNorm → FFN → residual.
    """

    vocab_size: int = 2
    d_model: int = 64
    num_heads: int = 2
    num_layers: int = 2
    max_seq_len: int = 20  # 2 * H
    ffn_mult: int = 4

    @nn.compact
    def __call__(self, tokens: Array) -> Array:
        """Forward pass.

        Args:
            tokens: integer token IDs, shape (seq_len,) or (B, seq_len).

        Returns:
            log-softmax logits, same leading dims + (vocab_size,).
        """
        single = tokens.ndim == 1
        if single:
            tokens = tokens[None]

        B, T = tokens.shape
        # Token + positional embeddings
        tok_emb = nn.Embed(self.vocab_size, self.d_model)(tokens)
        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.max_seq_len, self.d_model),
        )
        x = tok_emb + pos_emb[:T]

        # Causal mask
        causal_mask = nn.make_causal_mask(tokens)  # (B, 1, T, T)

        for _ in range(self.num_layers):
            # Pre-norm self-attention
            y = nn.LayerNorm()(x)
            y = nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                deterministic=True,
            )(y, mask=causal_mask)
            x = x + y

            # Pre-norm FFN
            y = nn.LayerNorm()(x)
            y = nn.Dense(self.ffn_mult * self.d_model)(y)
            y = nn.gelu(y)
            y = nn.Dense(self.d_model)(y)
            x = x + y

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        log_probs = nn.log_softmax(logits)

        if single:
            return log_probs[0]
        return log_probs
