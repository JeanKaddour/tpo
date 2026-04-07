import pytest

from tpo.cli import _default_match, parse_args, selected_experiments


def test_all_only_expands_to_fast_regression_suite() -> None:
    assert selected_experiments("all") == (
        "tabular_single",
        "tabular_multi",
        "mnist",
        "transformer",
        "transformer_rlvr",
    )


def test_match_defaults_are_command_specific() -> None:
    assert _default_match("transformer_variations", None) == "both"
    assert _default_match("rlvr_sweep", None) == "both"
    assert _default_match("transformer_rlvr", None) == "prompts"


def test_transformer_rlvr_rejects_match_both() -> None:
    with pytest.raises(ValueError, match="not supported"):
        _default_match("transformer_rlvr", "both")


def test_parse_args_rejects_transformer_rlvr_match_both(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["transformer_rlvr", "--match", "both"])

    assert excinfo.value.code == 2
    assert "transformer_rlvr" in capsys.readouterr().err
