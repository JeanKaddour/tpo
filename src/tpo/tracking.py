"""Shared experiment reporting and optional Weights & Biases logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence
import os
import uuid

import numpy as np

WANDB_PUBLIC_ENV_KEYS = (
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
)
WANDB_RUNTIME_ENV_KEYS = (*WANDB_PUBLIC_ENV_KEYS, "WANDB_MODE", "WANDB_BASE_URL")


@dataclass(frozen=True)
class CurveReport:
    """Aggregated series that should be logged together."""

    name: str
    x_name: str
    x_values: Sequence[float]
    series: Mapping[str, Sequence[float]]
    title: str | None = None
    plot_series: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentReport:
    """Structured outputs produced by one concrete experiment run."""

    name: str
    config: Mapping[str, object]
    summary: Mapping[str, float]
    curves: tuple[CurveReport, ...] = field(default_factory=tuple)
    artifact_paths: tuple[Path, ...] = field(default_factory=tuple)
    raw_errors: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class WandbConfig:
    """Resolved Weights & Biases configuration for the current invocation."""

    enabled: bool
    project: str | None = None
    entity: str | None = None
    group: str | None = None


def collect_wandb_env(
    environ: Mapping[str, str] | None = None,
    *,
    runtime: bool = False,
) -> dict[str, str]:
    """Return configured WandB environment variables with blank values removed."""
    source = os.environ if environ is None else environ
    keys = WANDB_RUNTIME_ENV_KEYS if runtime else WANDB_PUBLIC_ENV_KEYS
    result: dict[str, str] = {}
    for key in keys:
        value = _env_value(source, key)
        if value is not None:
            result[key] = value
    return result


def resolve_wandb_config(
    *,
    disabled: bool = False,
    environ: Mapping[str, str] | None = None,
) -> WandbConfig:
    """Resolve whether WandB should be enabled and with which settings."""
    source = os.environ if environ is None else environ
    mode = _env_value(source, "WANDB_MODE")
    if disabled or mode == "disabled":
        return WandbConfig(enabled=False)

    enabled = _env_value(source, "WANDB_API_KEY") is not None or mode == "offline"
    if not enabled:
        return WandbConfig(enabled=False)

    return WandbConfig(
        enabled=True,
        project=_env_value(source, "WANDB_PROJECT") or "tpo",
        entity=_env_value(source, "WANDB_ENTITY"),
        group=_env_value(source, "WANDB_RUN_GROUP"),
    )


def with_default_group(config: WandbConfig, prefix: str = "all") -> WandbConfig:
    """Attach a generated group id when logging is enabled and none is set."""
    if not config.enabled or config.group:
        return config
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    group = f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"
    return replace(config, group=group)


def init_wandb_run(
    experiment_name: str,
    config: object,
    wandb_config: WandbConfig,
    smoke: bool = False,
):
    """Initialize a WandB run before the experiment starts.

    Returns the run object so experiments can log progressively.
    """
    import wandb

    config_payload = _config_as_dict(config)
    tags = [experiment_name]
    if smoke:
        tags.append("smoke")

    run = wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        group=wandb_config.group,
        job_type=experiment_name,
        name=f"{experiment_name}-smoke" if smoke else experiment_name,
        config=config_payload,
        tags=tags,
        reinit="finish_previous",
    )
    return run


def log_experiment_report(
    report: ExperimentReport,
    wandb_config: WandbConfig,
    run_metadata: Mapping[str, object] | None = None,
    run=None,
) -> None:
    """Log one experiment report to WandB if logging is enabled.

    If *run* is provided (pre-initialized), uses that run and finishes it.
    Otherwise creates a new run.
    """
    if not wandb_config.enabled:
        return

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WandB logging is enabled, but the `wandb` package is not installed. "
            "Install project dependencies to use WandB logging."
        ) from exc

    metadata = {} if run_metadata is None else dict(run_metadata)
    own_run = run is None
    if own_run:
        config_payload = dict(report.config)
        config_payload.update(
            {
                key: value
                for key, value in metadata.items()
                if key not in {"elapsed_seconds"}
                and isinstance(value, (str, int, float, bool))
            }
        )

        tags = [report.name]
        if metadata.get("smoke"):
            tags.append("smoke")
        if execution_env := metadata.get("execution_env"):
            tags.append(str(execution_env))

        run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            group=wandb_config.group,
            job_type=report.name,
            name=f"{report.name}-smoke" if metadata.get("smoke") else report.name,
            config=config_payload,
            tags=tags,
            reinit="finish_previous",
        )
    if run is None:
        return

    try:
        # Log per-step metrics first (wandb database / built-in charts).
        _log_step_metrics(run, report.curves)

        # Then log tables and line-series preview charts.
        for curve in report.curves:
            _log_curve(run, wandb, curve)

        summary_payload = {key: float(value) for key, value in report.summary.items()}
        elapsed = metadata.get("elapsed_seconds")
        if elapsed is not None:
            summary_payload["runtime_seconds"] = float(elapsed)
        run.summary.update(summary_payload)

        _log_artifacts(run, wandb, report)
    finally:
        run.finish()


def _config_as_dict(config: object) -> dict:
    """Convert a dataclass config to a dict, returning {} for non-dataclasses."""
    return asdict(config) if hasattr(config, "__dataclass_fields__") else {}


def _env_value(source: Mapping[str, str], key: str) -> str | None:
    """Read an env var and treat empty strings as unset."""
    value = source.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _as_array(values: Sequence[float]) -> np.ndarray:
    """Convert metric series to a flat numpy array."""
    return np.asarray(values, dtype=float).reshape(-1)


def log_per_algorithm_runs(
    experiment_name: str,
    config: object,
    algorithms: Sequence[tuple[str, dict[str, np.ndarray]]],
    x_name: str,
    x_values: np.ndarray,
) -> None:
    """Create a separate wandb run per algorithm and log per-step metrics.

    This logs one run per algorithm. Here we log seed-aggregated statistics (mean, se, per-seed
    values) so runs can be overlaid in the wandb UI.

    Args:
        experiment_name: e.g. "mnist", "tabular_multi", "transformer".
        config: experiment config dataclass (logged as wandb config).
        algorithms: sequence of ``(algo_name, metrics_dict)`` where
            ``metrics_dict`` maps metric names to arrays of shape
            ``(num_seeds, num_steps)``.
        x_name: name for the x-axis (e.g. "step", "episode").
        x_values: 1-D array of x-axis values.
    """
    if not resolve_wandb_config(environ=os.environ).enabled:
        return
    try:
        import wandb
    except ImportError:
        return

    config_dict = _config_as_dict(config)
    project = os.environ.get("WANDB_PROJECT", "tpo")
    entity = os.environ.get("WANDB_ENTITY")
    x_arr = _as_array(x_values)

    for algo_name, metrics in algorithms:
        run = wandb.init(
            project=project,
            entity=entity,
            name=f"{experiment_name}/{algo_name}",
            config={**config_dict, "algorithm": algo_name},
            tags=[experiment_name, algo_name],
            reinit="finish_previous",
        )
        try:
            for idx in range(len(x_arr)):
                step_data = {x_name: float(x_arr[idx])}
                for metric_name, arr in metrics.items():
                    arr_np = np.asarray(arr)
                    if arr_np.ndim == 2:
                        # (num_seeds, num_steps) — log mean and se
                        vals = arr_np[:, idx]
                        step_data[f"{metric_name}/mean"] = float(vals.mean())
                        step_data[f"{metric_name}/se"] = float(
                            vals.std() / np.sqrt(len(vals))
                        )
                    elif arr_np.ndim == 1:
                        step_data[metric_name] = float(arr_np[idx])
                run.log(step_data, step=int(x_arr[idx]))
        finally:
            run.finish()


def _log_step_metrics(run, curves: tuple[CurveReport, ...]) -> None:
    """Log per-step metrics from all curves for wandb database queryability.

    Each series value is logged at every step so users can build arbitrary
    charts and queries from the wandb UI.  Metrics are namespaced by curve
    name (e.g. ``classification_error/pg_mean``).
    """
    if not curves:
        return

    # Gather all curves' data indexed by step value.  Multiple curves may
    # share the same x-axis (e.g. error and misalignment at same steps).
    step_data: dict[int, dict[str, float]] = {}
    for curve in curves:
        x_values = _as_array(curve.x_values)
        series = {name: _as_array(values) for name, values in curve.series.items()}
        for idx in range(len(x_values)):
            step = int(x_values[idx])
            if step not in step_data:
                step_data[step] = {}
            for name, values in series.items():
                step_data[step][f"{curve.name}/{name}"] = float(values[idx])

    # Log in step order so wandb's x-axis is monotonic.
    for step in sorted(step_data):
        run.log(step_data[step], step=step)


def _log_curve(run, wandb, curve: CurveReport) -> None:
    """Log a curve as both a table and a preview chart."""
    x_values = _as_array(curve.x_values)
    series = {name: _as_array(values) for name, values in curve.series.items()}
    series_names = tuple(series)
    expected_length = len(x_values)
    for name, values in series.items():
        if len(values) != expected_length:
            raise ValueError(
                f"Curve '{curve.name}' series '{name}' has length {len(values)}; "
                f"expected {expected_length}."
            )

    x_list = x_values.tolist()
    series_lists = {name: values.tolist() for name, values in series.items()}
    columns = [curve.x_name, *series_names]
    rows = [
        [x_list[idx], *(series_lists[name][idx] for name in series_names)]
        for idx in range(expected_length)
    ]
    table = wandb.Table(columns=columns, data=rows)
    run.log({f"tables/{curve.name}": table})

    plot_keys = curve.plot_series or series_names
    plot_ys = [series_lists[name] for name in plot_keys]
    plot = wandb.plot.line_series(
        xs=x_list,
        ys=plot_ys,
        keys=list(plot_keys),
        title=curve.title or curve.name,
        xname=curve.x_name,
    )
    run.log({f"plots/{curve.name}": plot})


def _log_artifacts(run, wandb, report: ExperimentReport) -> None:
    """Attach output files to the current WandB run."""
    artifact_paths = [Path(path) for path in report.artifact_paths]
    if not artifact_paths:
        return

    artifact = wandb.Artifact(
        f"{report.name}-{run.id}-artifacts", type="experiment-artifacts"
    )
    for path in artifact_paths:
        artifact.add_file(str(path), name=path.name)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            run.log({f"figures/{path.stem}": wandb.Image(str(path))})
    run.log_artifact(artifact)
