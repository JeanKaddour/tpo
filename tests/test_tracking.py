import numpy as np
import pytest

from tpo.tracking import WandbConfig, log_per_algorithm_runs, resolve_wandb_config


def test_resolve_wandb_config_defaults_to_tpo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "token")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_RUN_GROUP", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    config = resolve_wandb_config()

    assert config == WandbConfig(enabled=True, project="tpo", entity=None, group=None)


def test_log_per_algorithm_runs_respects_disabled_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "token")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    log_per_algorithm_runs(
        "mnist",
        object(),
        [("tpo", {"classification_error": np.ones((2, 3), dtype=np.float32)})],
        x_name="step",
        x_values=np.arange(3),
    )
