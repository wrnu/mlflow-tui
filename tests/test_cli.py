from __future__ import annotations

import pytest

from mlflow_tui.cli import parse_args


def test_parse_defaults() -> None:
    args = parse_args([])
    assert args.tracking_uri is None
    assert args.demo is False
    assert args.refresh == 3.0
    assert args.experiment is None


def test_parse_flags() -> None:
    args = parse_args(
        ["--tracking-uri", "http://localhost:5000", "--refresh", "1.5", "--experiment", "LT-JEPA"]
    )
    assert args.tracking_uri == "http://localhost:5000"
    assert args.refresh == 1.5
    assert args.experiment == "LT-JEPA"


def test_parse_demo() -> None:
    args = parse_args(["--demo"])
    assert args.demo is True


def test_version_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--version"])
    assert exit_info.value.code == 0
    assert "mlflow-tui" in capsys.readouterr().out
