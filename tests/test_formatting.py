from __future__ import annotations

import math

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
    ).plain
    assert "loss" in chart
    assert "last" in chart
    assert "┤" in chart
    assert "└" in chart
    assert any("\u2800" <= char <= "\u28ff" for char in chart)
    assert "79" in chart


def test_line_chart_falls_back_to_sparkline() -> None:
    from mlflow_tui.formatting import SPARK_CHARS, render_line_chart

    tiny = render_line_chart([1, 2, 3, 4], width=8, height=3, title="loss").plain
    assert tiny.startswith("loss")
    assert any(ch in tiny for ch in SPARK_CHARS)


def test_nice_ticks_use_1_2_5() -> None:
    from mlflow_tui.formatting import nice_ticks

    ticks = nice_ticks(0.0, 1.0, count=5)
    assert ticks[0] >= 0
    assert ticks[-1] <= 1.0000001
    assert ticks == sorted(ticks)
    steps = [round(ticks[i + 1] - ticks[i], 10) for i in range(len(ticks) - 1)]
    assert len(set(steps)) == 1


def test_line_chart_keeps_spike() -> None:
    from mlflow_tui.formatting import render_line_chart

    values = [0.0] * 80
    values[40] = 10.0
    lines = [
        line
        for line in render_line_chart(values, width=52, height=14, title="spike").plain.splitlines()
        if "│" in line or "┤" in line
    ]
    top = "".join(lines[:3])
    assert any("\u2800" <= char <= "\u28ff" for char in top)


def test_line_chart_maps_x_from_steps() -> None:
    from mlflow_tui.formatting import render_line_chart

    chart = render_line_chart(
        [1.0, 2.0, 3.0],
        xs=[100, 500, 900],
        width=48,
        height=12,
        title="loss",
        x0=100,
        x1=900,
    ).plain
    assert "100" in chart
    assert "900" in chart
    assert "500" in chart
    assert (
        "39.5"
        not in render_line_chart([0.0] * 80, width=52, height=12, title="spike", x0=0, x1=79).plain
    )


def test_nice_log_ticks_use_1_2_5() -> None:
    from mlflow_tui.formatting import nice_log_ticks

    ticks = nice_log_ticks(0.001, 1.0, count=5)
    assert ticks == sorted(ticks)
    assert all(tick > 0 for tick in ticks)
    for tick in ticks:
        exponent = math.floor(math.log10(tick))
        mantissa = tick / (10**exponent)
        assert any(abs(mantissa - coef) < 1e-6 for coef in (1.0, 2.0, 5.0))
    decades = nice_log_ticks(1e-8, 1e8, count=5)
    assert all(abs(math.log10(tick) - round(math.log10(tick))) < 1e-9 for tick in decades)


def test_line_chart_log_y_uses_symlog_for_signed_values() -> None:
    from mlflow_tui.formatting import render_line_chart

    positive = render_line_chart(
        [0.001, 0.01, 0.1, 1.0],
        width=48,
        height=12,
        title="loss",
        log_y=True,
    ).plain
    assert "loss" in positive
    assert "log" in positive.splitlines()[0]
    assert "symlog" not in positive.splitlines()[0]

    signed = render_line_chart(
        [-2.0, -0.2, 0.0, 0.2, 2.0],
        width=48,
        height=12,
        title="delta",
        log_y=True,
    ).plain
    assert "symlog" in signed.splitlines()[0]
    assert any("\u2800" <= char <= "\u28ff" for char in signed)
    assert "0" in signed
    assert "n=5" in signed


def test_line_chart_log_y_plots_all_negative_series() -> None:
    from mlflow_tui.formatting import render_line_chart

    chart = render_line_chart(
        [-8.0, -2.0, -0.5, -0.1],
        width=48,
        height=12,
        title="reward",
        log_y=True,
    ).plain
    assert "symlog" in chart.splitlines()[0]
    assert any("\u2800" <= char <= "\u28ff" for char in chart)
    assert "n=4" in chart


def test_zoom_and_pan_keep_the_window_inside_the_data() -> None:
    from mlflow_tui.formatting import clamp_view, pan_view, view_window, zoom_view

    start, span = zoom_view(0.0, 1.0, 0.5)
    assert abs(span - 0.5) < 1e-9
    assert abs(start - 0.25) < 1e-9
    start, span = pan_view(start, span, 1.0)
    assert start + span <= 1.0 + 1e-9
    start, span = clamp_view(-1.0, 3.0)
    assert start == 0.0
    assert span == 1.0
    lo, hi = view_window(0.0, 100.0, 0.25, 0.5)
    assert lo == 25.0
    assert hi == 75.0


def test_line_chart_zoom_uses_the_visible_x_window() -> None:
    from mlflow_tui.formatting import render_line_chart

    chart = render_line_chart(
        [1.0, 2.0, 3.0, 4.0],
        xs=[0, 100, 200, 400],
        width=48,
        height=12,
        title="loss",
        x0=0,
        x1=400,
        x_start=0.0,
        x_span=0.5,
        y_start=0.0,
        y_span=0.5,
    ).plain
    header = chart.splitlines()[0]
    assert "zoom" in header
    assert "x 2.0×" in header
    assert "y 2.0×" in header
    assert "0" in chart
    assert "200" in chart
    assert "400" not in chart.splitlines()[-1]
