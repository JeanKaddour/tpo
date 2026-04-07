"""Typed experiment configuration presets for the standalone TPO artifact."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping, TypeVar, Union


@dataclass(frozen=True)
class TransformerModelConfig:
    """Architecture config for the causal transformer experiments."""

    d_model: int = 64
    num_heads: int = 2
    num_layers: int = 2
    ffn_mult: int = 4


TRANSFORMER_MODEL = TransformerModelConfig()


CoreExperimentName = Literal[
    "tabular_single",
    "tabular_multi",
    "mnist",
    "transformer",
    "transformer_rlvr",
]
ExtendedExperimentName = Literal[
    "vocab_sweep",
    "transformer_variations",
    "rlvr_sweep",
    "mnist_mechanism",
    "terminal_reward_ablations",
    "causal_ablations",
]
ExperimentName = Literal[
    "tabular_single",
    "tabular_multi",
    "mnist",
    "transformer",
    "transformer_rlvr",
    "vocab_sweep",
    "transformer_variations",
    "rlvr_sweep",
    "mnist_mechanism",
    "terminal_reward_ablations",
    "causal_ablations",
]


EXPERIMENT_ORDER: tuple[CoreExperimentName, ...] = (
    "tabular_single",
    "tabular_multi",
    "mnist",
    "transformer",
    "transformer_rlvr",
)


@dataclass(frozen=True)
class MnistConfig:
    seed: int = 0
    num_seeds: int = 20
    num_steps: int = 10_000
    batch_size: int = 100
    learning_rate: float = 1e-3
    eta: float = 1.0
    eval_every: int = 200
    save_dir: str = "figures"


@dataclass(frozen=True)
class TabularSingleConfig:
    num_seeds: int = 100
    num_actions: int = 100
    batch_size: int = 100
    alpha: float = 0.1
    eta: float = 1.0
    num_steps: int = 150
    save_dir: str = "figures"


@dataclass(frozen=True)
class TabularMultiConfig:
    num_seeds: int = 100
    num_contexts: int = 100
    num_actions: int = 10
    alpha: float = 0.1
    eta: float = 1.0
    num_steps: int = 1_000
    save_dir: str = "figures"


@dataclass(frozen=True)
class TransformerConfig:
    num_seeds: int = 20
    sequence_length: int = 10
    vocab_size: int = 2
    num_episodes: int = 1_000
    batch_size: int = 100
    learning_rate: float = 1e-4
    eta: float = 1.0
    ppo_epsilon: float = 0.2
    ppo_epochs: int = 4
    k_candidates: int = 8
    tpo_eta: float = 1.0
    save_dir: str = "figures"


@dataclass(frozen=True)
class TransformerRlvrConfig:
    num_seeds: int = 20
    sequence_length: int = 10
    vocab_size: int = 2
    num_episodes: int = 2_000
    batch_size: int = 100
    learning_rate: float = 1e-4
    eta: float = 1.0
    ppo_epsilon: float = 0.2
    ppo_epochs: int = 4
    k_candidates: int = 8
    tpo_eta: float = 1.0
    save_dir: str = "figures"


@dataclass(frozen=True)
class TransformerVariationsConfig:
    num_seeds: int = 10
    sequence_length: int = 10
    vocab_size: int = 2
    num_episodes: int = 1_000
    batch_size: int = 100
    learning_rate: float = 1e-4
    eta: float = 1.0
    ppo_epsilon: float = 0.2
    ppo_epochs: int = 4
    k_candidates: int = 8
    tpo_eta: float = 1.0
    save_dir: str = "figures"


@dataclass(frozen=True)
class TerminalRewardAblationsConfig:
    num_seeds: int = 10
    sequence_length: int = 8
    vocab_size: int = 2
    num_episodes: int = 2_000
    batch_size: int = 256
    learning_rate: float = 1e-4
    eta: float = 1.0
    ppo_epsilon: float = 0.2
    ppo_epochs: int = 4
    k_candidates: int = 32
    tpo_eta: float = 1.0
    save_dir: str = "figures"


CausalAblationsConfig = TerminalRewardAblationsConfig


ExperimentConfig = Union[
    MnistConfig,
    TabularSingleConfig,
    TabularMultiConfig,
    TransformerConfig,
    TransformerRlvrConfig,
    TransformerVariationsConfig,
    TerminalRewardAblationsConfig,
    CausalAblationsConfig,
]


_FULL_PRESETS: Mapping[ExperimentName, ExperimentConfig] = {
    "mnist": MnistConfig(),
    "tabular_single": TabularSingleConfig(),
    "tabular_multi": TabularMultiConfig(),
    "transformer": TransformerConfig(),
    "transformer_rlvr": TransformerRlvrConfig(),
    "transformer_variations": TransformerVariationsConfig(),
    "terminal_reward_ablations": TerminalRewardAblationsConfig(),
    "causal_ablations": CausalAblationsConfig(),
}

_SMOKE_PRESETS: Mapping[ExperimentName, ExperimentConfig] = {
    "mnist": MnistConfig(num_seeds=2, num_steps=100, eval_every=20),
    "tabular_single": TabularSingleConfig(num_seeds=2, num_steps=20),
    "tabular_multi": TabularMultiConfig(num_seeds=2, num_steps=50),
    "transformer": TransformerConfig(num_seeds=2, num_episodes=50),
    "transformer_rlvr": TransformerRlvrConfig(
        num_seeds=2,
        num_episodes=200,
        sequence_length=5,
        batch_size=256,
        k_candidates=32,
    ),
    "transformer_variations": TransformerVariationsConfig(num_seeds=2, num_episodes=50),
    "terminal_reward_ablations": TerminalRewardAblationsConfig(
        num_seeds=2, num_episodes=100
    ),
    "causal_ablations": CausalAblationsConfig(num_seeds=2, num_episodes=100),
}


ConfigT = TypeVar("ConfigT")


def experiment_configs(
    smoke: bool, save_dir: str
) -> dict[ExperimentName, ExperimentConfig]:
    """Return typed experiment configs for the requested run mode."""

    presets = _SMOKE_PRESETS if smoke else _FULL_PRESETS
    return {
        name: replace(config, save_dir=save_dir) for name, config in presets.items()
    }


def coerce_config(
    config: ConfigT | None,
    config_type: type[ConfigT],
    overrides: Mapping[str, object] | None = None,
) -> ConfigT:
    """Accept either a config object or keyword overrides for that config type."""

    options = {} if overrides is None else dict(overrides)
    if config is not None:
        if options:
            raise TypeError(
                f"Pass either a `{config_type.__name__}` instance or keyword overrides, not both."
            )
        return config
    return config_type(**options)
