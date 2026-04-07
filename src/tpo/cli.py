"""Local-first CLI for the standalone TPO paper artifact."""

import argparse
from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import time

from .config import (
    EXPERIMENT_ORDER,
    experiment_configs,
)
from .runtime import bootstrap_runtime
from .tracking import (
    ExperimentReport,
    log_experiment_report,
    resolve_wandb_config,
    with_default_group,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent

for env_path in (PROJECT_ROOT / ".env", REPO_ROOT / ".env"):
    if env_path.exists():
        with open(env_path) as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


EXPERIMENT_CHOICES = (
    *EXPERIMENT_ORDER,
    "vocab_sweep",
    "transformer_variations",
    "rlvr_sweep",
    "mnist_mechanism",
    "terminal_reward_ablations",
    "causal_ablations",
    "all",
)

LOCAL_CLI_PROG = "python -m tpo.cli"


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="TPO paper experiments")
    parser.add_argument(
        "experiment",
        nargs="?",
        default="all",
        choices=EXPERIMENT_CHOICES,
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run a reduced smoke configuration"
    )
    parser.add_argument(
        "--save-dir",
        default="figures",
        help="Directory for output figures and raw data",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--no-verbose", action="store_true", help="Reduce progress output"
    )
    parser.add_argument(
        "--algorithms", default=None, help="Comma-separated list of algorithms to run"
    )
    parser.add_argument(
        "--match",
        default=None,
        choices=["prompts", "interactions", "both"],
        help="Budget matching mode for sweep-style commands",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override vocab size for `transformer`",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=None, help="Override sequence length"
    )
    parser.add_argument(
        "--sequence-lengths",
        default=None,
        help="Comma-separated sequence lengths for `rlvr_sweep`",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=None, help="Override number of random seeds"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=None, help="Override episode count"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override step count for bandit runs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument(
        "--k-candidates",
        type=int,
        default=None,
        help="Override grouped candidate count",
    )
    parser.add_argument(
        "--eta", type=float, default=None, help="Override reward-scaling temperature"
    )
    parser.add_argument(
        "--tpo-eta", type=float, default=None, help="Override TPO target temperature"
    )
    parser.add_argument(
        "--ppo-epochs", type=int, default=None, help="Override PPO/TPO epoch count"
    )
    return parser


def parse_args(
    argv: tuple[str, ...] | list[str] | None = None, prog: str | None = None
) -> argparse.Namespace:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        _validate_match_selection(args.experiment, args.match)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def selected_experiments(experiment: str) -> tuple[str, ...]:
    if experiment == "all":
        return EXPERIMENT_ORDER
    return (experiment,)


def _parse_algorithms(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values or None


def _parse_sequence_lengths(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return values or None


def _default_match(experiment: str, match: str | None) -> str:
    _validate_match_selection(experiment, match)
    if match is not None:
        return match
    if experiment in {"transformer_variations", "rlvr_sweep"}:
        return "both"
    return "prompts"


_MATCH_CHOICES = {
    "transformer_rlvr": ("prompts", "interactions"),
    "transformer_variations": ("prompts", "interactions", "both"),
    "rlvr_sweep": ("prompts", "interactions", "both"),
}


def _validate_match_selection(experiment: str, match: str | None) -> None:
    if match is None:
        return
    invalid: list[str] = []
    for name in selected_experiments(experiment):
        supported = _MATCH_CHOICES.get(name)
        if supported is not None and match not in supported:
            invalid.append(name)
    if not invalid:
        return
    names = ", ".join(invalid)
    supported = ", ".join(_MATCH_CHOICES[invalid[0]])
    raise ValueError(
        f"--match {match!r} is not supported for {names}; choose one of: {supported}"
    )


def _override_config(config, args: argparse.Namespace):
    updates = {}
    for field_name, value in (
        ("num_seeds", args.num_seeds),
        ("num_episodes", args.num_episodes),
        ("num_steps", args.num_steps),
        ("batch_size", args.batch_size),
        ("learning_rate", args.learning_rate),
        ("sequence_length", args.sequence_length),
        ("vocab_size", args.vocab_size),
        ("k_candidates", args.k_candidates),
        ("eta", args.eta),
        ("tpo_eta", args.tpo_eta),
        ("ppo_epochs", args.ppo_epochs),
    ):
        if value is not None and hasattr(config, field_name):
            updates[field_name] = value
    return replace(config, **updates) if updates else config


def _base_config(name: str, smoke: bool, save_dir: str):
    presets = experiment_configs(smoke=smoke, save_dir=save_dir)
    if name == "vocab_sweep":
        return presets["transformer"]
    if name == "rlvr_sweep":
        return presets["transformer_rlvr"]
    if name == "mnist_mechanism":
        return presets["mnist"]
    return presets[name]


@contextmanager
def _wandb_mode(no_wandb: bool):
    previous = os.environ.get("WANDB_MODE")
    if no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("WANDB_MODE", None)
        else:
            os.environ["WANDB_MODE"] = previous


def _run_one(name: str, args: argparse.Namespace):
    bootstrap_runtime()

    algorithms = _parse_algorithms(args.algorithms)
    verbose = not args.no_verbose

    if name == "tabular_single":
        from .experiments.tabular import run_single_context

        config = _override_config(_base_config(name, args.smoke, args.save_dir), args)
        return run_single_context(config)
    if name == "tabular_multi":
        from .experiments.tabular import run_multi_context

        config = _override_config(_base_config(name, args.smoke, args.save_dir), args)
        return run_multi_context(config)
    if name == "mnist":
        from .experiments.mnist import run_mnist

        config = _override_config(_base_config(name, args.smoke, args.save_dir), args)
        return run_mnist(config, algorithms=algorithms)
    if name == "mnist_mechanism":
        from .experiments.mnist_mechanism import run_mnist_mechanism

        config = _override_config(_base_config(name, args.smoke, args.save_dir), args)
        return run_mnist_mechanism(config, algorithms=algorithms)
    if name == "transformer":
        from .experiments.transformer import run_transformer

        config = _override_config(_base_config(name, args.smoke, args.save_dir), args)
        return run_transformer(config, algorithms=algorithms, verbose=verbose)
    if name == "vocab_sweep":
        from .experiments.vocab_sweep import run_vocab_sweep

        config = _override_config(_base_config(name, args.smoke, args.save_dir), args)
        return run_vocab_sweep(config, algorithms=algorithms, verbose=verbose)
    if name == "transformer_variations":
        from .experiments.transformer_variations import run_transformer_variations

        config = _override_config(_base_config(name, args.smoke, args.save_dir), args)
        return run_transformer_variations(
            config,
            algorithms=algorithms,
            match=_default_match(name, args.match),
        )
    if name == "transformer_rlvr":
        from .experiments.transformer_rlvr import run_transformer_rlvr

        config = _override_config(_base_config(name, args.smoke, args.save_dir), args)
        return run_transformer_rlvr(
            config,
            algorithms=algorithms,
            match=_default_match(name, args.match),
            verbose=verbose,
        )
    if name == "rlvr_sweep":
        from .experiments.rlvr_sweep import run_rlvr_sweep

        config = _override_config(_base_config(name, args.smoke, args.save_dir), args)
        sequence_lengths = _parse_sequence_lengths(args.sequence_lengths)
        kwargs = {}
        if sequence_lengths is not None:
            kwargs["sequence_lengths"] = sequence_lengths
        return run_rlvr_sweep(
            config,
            algorithms=algorithms,
            match=_default_match(name, args.match),
            verbose=verbose,
            **kwargs,
        )
    if name == "terminal_reward_ablations":
        from .experiments.terminal_reward_ablations import run_terminal_reward_ablations

        config = _override_config(_base_config(name, args.smoke, args.save_dir), args)
        return run_terminal_reward_ablations(
            config, algorithms=algorithms, verbose=verbose
        )
    if name == "causal_ablations":
        from .experiments.causal_ablations import run_causal_ablations

        config = _override_config(_base_config(name, args.smoke, args.save_dir), args)
        return run_causal_ablations(config, algorithms=algorithms, verbose=verbose)

    raise ValueError(f"Unknown experiment: {name}")


def run_experiments(args: argparse.Namespace) -> None:
    experiments = selected_experiments(args.experiment)
    wandb_config = resolve_wandb_config(disabled=args.no_wandb)
    if len(experiments) > 1:
        wandb_config = with_default_group(wandb_config)

    with _wandb_mode(args.no_wandb):
        for name in experiments:
            print(f"\n{'=' * 60}")
            print(f"  Experiment: {name}")
            print(f"{'=' * 60}")
            started = time.time()
            result = _run_one(name, args)
            elapsed = time.time() - started

            if isinstance(result, ExperimentReport):
                log_experiment_report(
                    result,
                    wandb_config,
                    run_metadata={
                        "elapsed_seconds": elapsed,
                        "smoke": args.smoke,
                        "execution_env": os.environ.get("TPO_RUN_CONTEXT", "local"),
                    },
                )
            print(f"  Completed in {elapsed:.1f}s")


def main(argv: tuple[str, ...] | list[str] | None = None) -> None:
    args = parse_args(argv, prog=LOCAL_CLI_PROG)
    run_experiments(args)


if __name__ == "__main__":
    main()
