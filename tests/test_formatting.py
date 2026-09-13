from __future__ import annotations

from rich.text import Text

from mlflow_tui.formatting import (
    downsample,
    format_age,
    format_bytes,
    format_duration,
    format_value,
    gpu_label,
    is_mlflow_filter,
    pick_metric_columns,
    run_duration_ms,
    sparkline,
    status_text,
    visible_tags,
)


def test_downsample_keeps_endpoints() -> None:
    values = list(range(10))
    sampled = downsample(values, 5)
    assert sampled[0] == 0
    assert sampled[-1] == 9
    assert len(sampled) == 5


def test_downsample_noop_when_small() -> None:
    assert downsample([1, 2, 3], 10) == [1, 2, 3]


def test_sparkline_flat_and_rising() -> None:
    assert sparkline([1, 1, 1], 4) == "▁" * 4
    rising = sparkline([0, 1, 2, 3, 4, 5, 6, 7], 8)
    assert len(rising) == 8
    assert rising[0] == "▁"
    assert rising[-1] == "█"


def test_format_value_scales() -> None:
    assert format_value(0) == "0"
    assert format_value(42_100_000) == "42.1M"
    assert format_value(0.0381) == "0.0381"
    assert format_value(3e-4).startswith("3.00e-04") or format_value(3e-4).startswith("3.0e-04")


def test_format_bytes_and_duration() -> None:
    assert format_bytes(None) == "-"
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048).endswith("KB")
    assert format_duration(None) == "-"
    assert format_duration(5_000) == "5s"
    assert format_duration(125_000) == "2m 05s"
    assert format_age(1_000, now_ms=61_000) == "1m"


def test_run_duration_uses_now_when_running() -> None:
    assert run_duration_ms(1_000, None, now_ms=11_000) == 10_000
    assert run_duration_ms(1_000, 4_000) == 3_000


def test_status_and_gpu_and_tags() -> None:
    running = status_text("RUNNING")
    assert isinstance(running, Text)
    assert "running" in running.plain
    assert gpu_label({"gpu": "91%"}, {}) == "91%"
    assert gpu_label({}, {"gpu_util": 88.2}) == "88%"
    tags = visible_tags(
        {
            "mlflow.user": "warren",
            "mlflow.log-model.history": "nope",
            "device": "A100",
        }
    )
    keys = [key for key, _ in tags]
    assert "user" in keys
    assert "device" in keys
    assert "log-model.history" not in keys


def test_metric_column_preference() -> None:
    assert pick_metric_columns(["accuracy", "loss", "lr"]) == ["loss", "accuracy", "lr"]


def test_mlflow_filter_detection() -> None:
    assert is_mlflow_filter("metrics.loss < 0.1")
    assert is_mlflow_filter('tags."mlflow.user" = "warren"')
    assert not is_mlflow_filter("run-0042")


def test_line_chart_contains_axes_and_title() -> None:
    from mlflow_tui.formatting import render_line_chart

    chart = render_line_chart(
        [0.10, 0.08, 0.06, 0.04, 0.038],
        width=40,
        height=10,
        title="loss",
        x0=0,
        x1=79,
    )
    assert "loss" in chart
    assert "┤" in chart
    assert "└" in chart
    assert "●" in chart


def test_line_chart_falls_back_to_sparkline() -> None:
    from mlflow_tui.formatting import SPARK_CHARS, render_line_chart

    tiny = render_line_chart([1, 2, 3, 4], width=8, height=3, title="loss")
    assert tiny.startswith("loss")
    assert any(ch in tiny for ch in SPARK_CHARS)
