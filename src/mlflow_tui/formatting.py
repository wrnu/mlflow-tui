from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TypeVar

from rich.cells import cell_len, set_cell_size
from rich.text import Text

T = TypeVar("T")

SPARK_CHARS = "▁▂▃▄▅▆▇█"
EMA_WEIGHT = 0.8

STATUS_STYLE: dict[str, tuple[str, str, str]] = {
    "RUNNING": ("●", "running", "bold #3ddc97"),
    "FINISHED": ("✓", "done", "#3ddc97"),
    "FAILED": ("✕", "failed", "bold #ff6b6b"),
    "KILLED": ("■", "killed", "#ffd166"),
    "SCHEDULED": ("○", "queued", "#7ee0ff"),
}

PREFERRED_METRICS = (
    "loss",
    "val_loss",
    "train_loss",
    "rmse",
    "mae",
    "accuracy",
    "val_accuracy",
    "f1",
    "auc",
    "reward",
)


def downsample(values: Sequence[T], max_points: int) -> list[T]:
    n = len(values)
    if max_points <= 0 or n == 0:
        return []
    if n <= max_points:
        return list(values)
    if max_points == 1:
        return [values[-1]]
    return [values[round(i * (n - 1) / (max_points - 1))] for i in range(max_points)]


def ema(values: Sequence[float], weight: float = EMA_WEIGHT) -> list[float]:
    """TensorBoard-style debiased exponential moving average."""
    if not values:
        return []
    if weight <= 0:
        return [float(v) for v in values]
    if weight >= 1:
        return [float(values[0])] * len(values)
    last = 0.0
    out: list[float] = []
    for index, value in enumerate(values, start=1):
        last = last * weight + (1.0 - weight) * float(value)
        debias = 1.0 - weight**index
        out.append(last / debias if debias else last)
    return out


def sparkline(values: Sequence[float], width: int = 24) -> str:
    if not values or width <= 0:
        return ""
    if len(values) == 1:
        sampled = [values[0]] * width
    else:
        last = max(width - 1, 1)
        sampled = [values[round(i * (len(values) - 1) / last)] for i in range(width)]
    lo = min(sampled)
    hi = max(sampled)
    span = hi - lo
    if span == 0:
        return SPARK_CHARS[0] * len(sampled)
    last_char = len(SPARK_CHARS) - 1
    return "".join(
        SPARK_CHARS[min(last_char, int((value - lo) / span * last_char))] for value in sampled
    )


def format_value(value: float) -> str:
    av = abs(value)
    if av == 0:
        return "0"
    if av >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if av >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if av >= 10_000:
        return f"{value / 1_000:.1f}k"
    if av >= 100:
        return f"{value:.1f}"
    if av >= 10:
        return f"{value:.2f}"
    if av >= 1:
        return f"{value:.3f}"
    if av >= 0.001:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    return f"{value:.2e}"


def format_bytes(size: int | None) -> str:
    if size is None:
        return "-"
    if size < 1024:
        return f"{size} B"
    units = ("KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        value /= 1024
        if value < 1024:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PB"


def format_duration(ms: int | None) -> str:
    if ms is None or ms < 0:
        return "-"
    seconds = int(ms / 1000)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s" if seconds else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def format_age(start_time_ms: int | None, now_ms: int | None = None) -> str:
    if not start_time_ms:
        return "-"
    now = now_ms if now_ms is not None else _now_ms()
    return format_duration(max(0, now - start_time_ms))


def run_duration_ms(
    start_time: int | None,
    end_time: int | None,
    now_ms: int | None = None,
) -> int | None:
    if not start_time:
        return None
    end = end_time if end_time is not None else (now_ms if now_ms is not None else _now_ms())
    return max(0, end - start_time)


def status_text(status: str) -> Text:
    icon, label, style = STATUS_STYLE.get(status.upper(), ("•", status.lower(), "dim"))
    return Text.assemble((icon, style), " ", (label, style))


def pick_metric_columns(metric_names: Sequence[str], limit: int = 3) -> list[str]:
    names = list(dict.fromkeys(metric_names))
    chosen: list[str] = []
    for preferred in PREFERRED_METRICS:
        if preferred in names:
            chosen.append(preferred)
        if len(chosen) >= limit:
            return chosen
    for name in names:
        if name not in chosen:
            chosen.append(name)
        if len(chosen) >= limit:
            break
    return chosen


def gpu_label(tags: dict[str, str], metrics: dict[str, float]) -> str:
    for key in ("gpu", "gpu_util", "gpu_utilization"):
        if key in tags:
            return tags[key]
    for key in ("gpu_util", "gpu_utilization", "gpu"):
        if key in metrics:
            return f"{metrics[key]:.0f}%"
    return "-"


def is_mlflow_filter(text: str) -> bool:
    stripped = text.strip()
    tokens = ("metrics.", "params.", "tags.", "attribute.", "attributes.")
    return any(token in stripped for token in tokens)


def visible_tags(tags: dict[str, str]) -> list[tuple[str, str]]:
    useful_internal = {
        "mlflow.user",
        "mlflow.source.name",
        "mlflow.source.type",
        "mlflow.runName",
    }
    items = []
    for key, value in sorted(tags.items()):
        if key.startswith("mlflow.") and key not in useful_internal:
            continue
        items.append((key.removeprefix("mlflow."), value))
    return items


_BRAILLE = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)
_LINE_STYLE = "#7ee0ff"
_RAW_STYLE = "dim #3d5563"
_LAST_STYLE = "bold #3ddc97"
_AXIS_STYLE = "#6b8796"
_X_LABEL_STYLE = "#c5d4de"
_GRID_STYLE = "#243848"
_TITLE_STYLE = "bold #c5d4de"


def format_tick(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if value == 0:
        return "0"
    av = abs(value)
    if av >= 1_000_000 or av < 1e-4:
        return f"{value:.2e}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_x(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return format_tick(value)


def nice_log_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Log-axis ticks on 1/2/5 × 10^n (decades only when the span is wide)."""
    if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0 or hi <= 0:
        return [1.0]
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        return [lo]
    exp_min = math.floor(math.log10(lo))
    exp_max = math.ceil(math.log10(hi))
    ticks: list[float] = []
    for exp in range(exp_min, exp_max + 1):
        for coef in (1.0, 2.0, 5.0):
            tick = coef * (10**exp)
            if lo <= tick <= hi:
                ticks.append(tick)
    if len(ticks) > max(count, 6):
        decades = [tick for tick in ticks if abs(math.log10(tick) - round(math.log10(tick))) < 1e-9]
        if len(decades) >= 2:
            ticks = decades
    return ticks or [lo, hi]


def nice_symlog_ticks(lo: float, hi: float, linthresh: float, count: int = 5) -> list[float]:
    """Symlog ticks: 0, ±1/2/5 × 10^n, matching Matplotlib's signed log axis."""
    if hi < lo:
        lo, hi = hi, lo
    ticks: list[float] = []
    if lo <= 0 <= hi:
        ticks.append(0.0)
    thresh = max(abs(linthresh), 1e-300)
    max_abs = max(abs(lo), abs(hi), thresh)
    for tick in nice_log_ticks(thresh, max_abs, count):
        if lo <= tick <= hi:
            ticks.append(tick)
        if lo <= -tick <= hi:
            ticks.append(-tick)
    uniq: list[float] = []
    for tick in sorted(ticks):
        if not uniq or abs(tick - uniq[-1]) > max(1e-12, abs(tick) * 1e-9):
            uniq.append(tick)
    return uniq or [lo, hi]


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Axis ticks on 1/2/5 × 10^n so labels match the data instead of row interpolation."""
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0]
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    if span == 0 or count < 2:
        return [lo]
    raw = span / (count - 1)
    exp = math.floor(math.log10(raw)) if raw > 0 else 0
    base = 10**exp
    step = 10 * base
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if candidate * base >= raw * 0.55:
            step = candidate * base
            break
    start = math.ceil(lo / step - 1e-12) * step
    ticks: list[float] = []
    value = start
    for _ in range(24):
        if value > hi + step * 1e-9:
            break
        ticks.append(value)
        value += step
    return ticks or [lo, hi]


def _padded_bounds(lo: float, hi: float, pad: float = 0.06) -> tuple[float, float]:
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        delta = max(abs(lo) * 0.1, 1e-6)
        return lo - delta, hi + delta
    span = hi - lo
    lo_p = lo - span * pad
    hi_p = hi + span * pad
    if lo >= 0:
        lo_p = max(0.0, lo_p)
    if hi <= 0:
        hi_p = min(0.0, hi_p)
    return lo_p, hi_p


MIN_VIEW_SPAN = 0.04
ZOOM_IN = 0.7
ZOOM_OUT = 1.0 / ZOOM_IN
PAN_STEP = 0.25


def clamp_view(start: float, span: float) -> tuple[float, float]:
    span = min(1.0, max(MIN_VIEW_SPAN, span))
    start = min(max(0.0, start), 1.0 - span)
    return start, span


def zoom_view(start: float, span: float, factor: float) -> tuple[float, float]:
    center = start + span / 2
    span = min(1.0, max(MIN_VIEW_SPAN, span * factor))
    return clamp_view(center - span / 2, span)


def pan_view(start: float, span: float, delta: float) -> tuple[float, float]:
    return clamp_view(start + delta * span, span)


def view_window(lo: float, hi: float, start: float, span: float) -> tuple[float, float]:
    if hi < lo:
        lo, hi = hi, lo
    full = hi - lo
    if full == 0:
        return lo, hi
    left = lo + full * start
    return left, left + full * span


def view_is_zoomed(x_span: float, y_span: float) -> bool:
    return x_span < 1.0 - 1e-9 or y_span < 1.0 - 1e-9


def _auto_linthresh(values: Sequence[float]) -> float:
    abs_nz = [abs(value) for value in values if value != 0 and math.isfinite(value)]
    if not abs_nz:
        return 1.0
    return max(min(abs_nz), 1e-12)


def _symlog(value: float, linthresh: float, *, base: float = 10.0, linscale: float = 1.0) -> float:
    """Matplotlib SymmetricalLogTransform: log |y|, linear in (-linthresh, linthresh)."""
    if not math.isfinite(value):
        return 0.0
    thresh = max(abs(linthresh), 1e-300)
    linscale_adj = linscale / (1.0 - base**-1)
    if abs(value) <= thresh:
        return value * linscale_adj
    return math.copysign(
        thresh * (linscale_adj + math.log(abs(value) / thresh) / math.log(base)),
        value,
    )


def _y_log_mode(values: Sequence[float]) -> str:
    if all(value > 0 for value in values):
        return "log"
    return "symlog"


def _axis_y(value: float, *, mode: str, linthresh: float) -> float:
    if mode == "log":
        return math.log10(value) if value > 0 else 0.0
    if mode == "symlog":
        return _symlog(value, linthresh)
    return value


def _padded_axis_bounds(
    lo: float, hi: float, *, mode: str, linthresh: float, pad: float = 0.06
) -> tuple[float, float]:
    t_lo = _axis_y(lo, mode=mode, linthresh=linthresh)
    t_hi = _axis_y(hi, mode=mode, linthresh=linthresh)
    if t_hi < t_lo:
        t_lo, t_hi = t_hi, t_lo
    if t_hi == t_lo:
        t_lo -= 0.5
        t_hi += 0.5
    else:
        span = t_hi - t_lo
        t_lo -= span * pad
        t_hi += span * pad
    return t_lo, t_hi


def _scale_free(value: float, lo: float, hi: float, size: int) -> int:
    if size <= 1 or hi == lo:
        return 0
    return int(round((value - lo) / (hi - lo) * (size - 1)))


def _value_plot_row(
    value: float,
    t_lo: float,
    t_hi: float,
    plot_h: int,
    *,
    mode: str,
    linthresh: float,
) -> int:
    """Character row for a y value, using the same braille grid as the line."""
    canvas_h = plot_h * 4
    py = (canvas_h - 1) - _scale_free(
        _axis_y(value, mode=mode, linthresh=linthresh), t_lo, t_hi, canvas_h
    )
    return max(0, min(plot_h - 1, py // 4))


def _plot_dot(cells: list[int], plot_w: int, plot_h: int, px: int, py: int) -> None:
    canvas_w = plot_w * 2
    canvas_h = plot_h * 4
    if px < 0 or py < 0 or px >= canvas_w or py >= canvas_h:
        return
    col, sub_x = divmod(px, 2)
    row, sub_y = divmod(py, 4)
    cells[row * plot_w + col] |= _BRAILLE[sub_y][sub_x]


def _draw_dots(
    cells: list[int], plot_w: int, plot_h: int, x0: int, y0: int, x1: int, y1: int
) -> None:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        _plot_dot(cells, plot_w, plot_h, x, y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _stroke_line(
    cells: list[int],
    plot_w: int,
    plot_h: int,
    xs: list[float],
    ys: list[float],
    x_lo: float,
    x_hi: float,
    t_lo: float,
    t_hi: float,
    *,
    mode: str,
    linthresh: float,
    envelope: bool = True,
) -> int:
    canvas_w = plot_w * 2
    canvas_h = plot_h * 4
    if envelope:
        sampled_x, sampled_y = _minmax_series(xs, ys, canvas_w)
    else:
        sampled_x = downsample(xs, canvas_w)
        sampled_y = downsample(ys, canvas_w)
    points: list[tuple[int, int]] = []
    for x_val, y_val in zip(sampled_x, sampled_y, strict=False):
        px = _scale_free(x_val, x_lo, x_hi, canvas_w)
        py = (canvas_h - 1) - _scale_free(
            _axis_y(y_val, mode=mode, linthresh=linthresh), t_lo, t_hi, canvas_h
        )
        points.append((px, py))
    if not points:
        return -1
    for i in range(len(points) - 1):
        _draw_dots(cells, plot_w, plot_h, *points[i], *points[i + 1])
    last_px, last_py = points[-1]
    last_col, last_row = last_px // 2, last_py // 4
    if 0 <= last_row < plot_h and 0 <= last_col < plot_w:
        return last_row * plot_w + last_col
    return -1


def _minmax_series(
    xs: list[float], ys: list[float], buckets: int
) -> tuple[list[float], list[float]]:
    n = len(ys)
    if n <= max(buckets * 2, 2):
        return xs, ys
    out_x: list[float] = []
    out_y: list[float] = []
    for bucket in range(buckets):
        start = int(bucket * n / buckets)
        end = int((bucket + 1) * n / buckets)
        if start >= end:
            continue
        chunk_y = ys[start:end]
        i_min = min(range(len(chunk_y)), key=chunk_y.__getitem__)
        i_max = max(range(len(chunk_y)), key=chunk_y.__getitem__)
        for index in sorted({0, i_min, i_max, len(chunk_y) - 1}):
            out_x.append(xs[start + index])
            out_y.append(chunk_y[index])
    return out_x, out_y


def _x_label_line(plot_w: int, labels: list[tuple[float, str]]) -> str:
    buf = [" "] * max(0, plot_w)
    if plot_w <= 0:
        return ""
    used: list[tuple[int, int]] = []

    def priority(frac: float) -> tuple[int, float]:
        if frac <= 0:
            return (0, 0.0)
        if frac >= 1:
            return (0, 1.0)
        return (1, abs(frac - 0.5))

    for frac, text in sorted(labels, key=lambda item: priority(item[0])):
        if not text:
            continue
        text = text[:plot_w]
        if frac <= 0:
            start = 0
        elif frac >= 1:
            start = max(0, plot_w - len(text))
        else:
            start = int(round(frac * (plot_w - 1) - len(text) / 2))
            start = max(0, min(start, plot_w - len(text)))
        end = min(plot_w, start + len(text))
        if any(start < right and end > left for left, right in used):
            continue
        for i, char in enumerate(text[: end - start]):
            buf[start + i] = char
        used.append((start, end))
    return "".join(buf)


def _clip_cells(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    if width == 1:
        return "…"
    return set_cell_size(text, width - 1) + "…"


def _append_header(chart: Text, parts: list[tuple[str, str]], width: int) -> None:
    used = 0
    for text, style in parts:
        remain = width - used
        if remain <= 0:
            break
        piece = _clip_cells(text, remain)
        if not piece:
            break
        chart.append(piece, style=style)
        used += cell_len(piece)
    chart.append("\n")


def render_line_chart(
    values: Sequence[float],
    *,
    width: int,
    height: int,
    title: str = "",
    x0: int | None = None,
    x1: int | None = None,
    xs: Sequence[float] | None = None,
    log_y: bool = False,
    x_start: float = 0.0,
    x_span: float = 1.0,
    y_start: float = 0.0,
    y_span: float = 1.0,
    smooth: bool = False,
) -> Text:
    """Braille line chart: 2×4 dots per cell, ticks on nice numbers, x from real steps."""
    if not values:
        return Text("No metric history", style="dim")
    series_y = [float(v) for v in values]
    if xs is None:
        series_x = [float(i) for i in range(len(series_y))]
    else:
        series_x = [float(v) for v in xs]
        if len(series_x) != len(series_y):
            series_x = [float(i) for i in range(len(series_y))]
    log_mode = _y_log_mode(series_y) if log_y else "linear"
    linthresh = _auto_linthresh(series_y) if log_mode == "symlog" else 1.0
    x_start, x_span = clamp_view(x_start, x_span)
    y_start, y_span = clamp_view(y_start, y_span)
    zoomed = view_is_zoomed(x_span, y_span)
    trend_y = ema(series_y) if smooth else series_y
    if width < 16 or height < 5:
        head = f"{title}  " if title else ""
        if log_y:
            head = f"{head}{log_mode}  "
        if smooth:
            head = f"{head}smooth  "
        if log_y:
            spark_src = [_axis_y(y, mode=log_mode, linthresh=linthresh) for y in trend_y]
        else:
            spark_src = trend_y
        return Text(f"{head}{sparkline(spark_src, max(8, width - 2))}", style=_LINE_STYLE)

    if log_y:
        t_full_lo, t_full_hi = _padded_axis_bounds(
            min(series_y), max(series_y), mode=log_mode, linthresh=linthresh
        )
        t_lo, t_hi = view_window(t_full_lo, t_full_hi, y_start, y_span)
        if log_mode == "log":
            ticks = nice_log_ticks(10**t_lo, 10**t_hi, count=min(6, max(3, height // 3)))
        else:
            ticks = [
                tick
                for tick in nice_symlog_ticks(
                    min(series_y),
                    max(series_y),
                    linthresh,
                    count=min(6, max(3, height // 3)),
                )
                if t_lo <= _axis_y(tick, mode=log_mode, linthresh=linthresh) <= t_hi
            ]
            if not ticks:
                ticks = nice_symlog_ticks(
                    min(series_y), max(series_y), linthresh, count=min(6, max(3, height // 3))
                )
    else:
        t_full_lo, t_full_hi = _padded_bounds(min(series_y), max(series_y))
        t_lo, t_hi = view_window(t_full_lo, t_full_hi, y_start, y_span)
        ticks = nice_ticks(t_lo, t_hi, count=min(6, max(3, height // 3)))
    x_full_lo = float(x0) if x0 is not None else series_x[0]
    x_full_hi = float(x1) if x1 is not None else series_x[-1]
    if x_full_hi == x_full_lo:
        x_full_lo -= 1
        x_full_hi += 1
    x_lo, x_hi = view_window(x_full_lo, x_full_hi, x_start, x_span)

    gutter = max(6, min(10, max(len(format_tick(tick)) for tick in (ticks or [0.0]))))
    plot_h = max(4, height - 3)
    plot_w = max(8, width - gutter - 2)

    raw_cells = [0] * (plot_h * plot_w)
    last_index = _stroke_line(
        raw_cells,
        plot_w,
        plot_h,
        series_x,
        series_y,
        x_lo,
        x_hi,
        t_lo,
        t_hi,
        mode=log_mode,
        linthresh=linthresh,
        envelope=not smooth,
    )
    smooth_cells: list[int] | None = None
    if smooth:
        smooth_cells = [0] * (plot_h * plot_w)
        last_index = _stroke_line(
            smooth_cells,
            plot_w,
            plot_h,
            series_x,
            list(trend_y),
            x_lo,
            x_hi,
            t_lo,
            t_hi,
            mode=log_mode,
            linthresh=linthresh,
            envelope=False,
        )

    tick_rows = {
        _value_plot_row(tick, t_lo, t_hi, plot_h, mode=log_mode, linthresh=linthresh): tick
        for tick in ticks
    }

    chart = Text()
    header_parts: list[tuple[str, str]] = []
    if title:
        header_parts.append((f"{title}  ", _TITLE_STYLE))
    header_parts.append((f"last {format_value(series_y[-1])}  ", _LAST_STYLE))
    header_parts.append((f"n={len(series_y)}", _AXIS_STYLE))
    if log_y:
        header_parts.append((f"  {log_mode}", _TITLE_STYLE))
    if smooth:
        header_parts.append(("  smooth", _TITLE_STYLE))
    if zoomed:
        header_parts.append((f"  zoom x {1 / x_span:.1f}×  y {1 / y_span:.1f}×", _TITLE_STYLE))
    lo, hi = min(series_y), max(series_y)
    header_parts.append((f"  min {format_value(lo)}  max {format_value(hi)}", _AXIS_STYLE))
    _append_header(chart, header_parts, width)

    for row in range(plot_h):
        tick = tick_rows.get(row)
        if tick is not None:
            chart.append(f"{format_tick(tick):>{gutter}}", style=_AXIS_STYLE)
            chart.append(" ┤", style=_AXIS_STYLE)
        else:
            chart.append(" " * gutter, style=_AXIS_STYLE)
            chart.append(" │", style=_AXIS_STYLE)
        for col in range(plot_w):
            index = row * plot_w + col
            smooth_bits = smooth_cells[index] if smooth_cells is not None else 0
            raw_bits = raw_cells[index]
            if smooth_bits:
                style = _LAST_STYLE if index == last_index else _LINE_STYLE
                chart.append(chr(0x2800 + smooth_bits), style=style)
            elif raw_bits:
                if smooth:
                    style = _RAW_STYLE
                elif index == last_index:
                    style = _LAST_STYLE
                else:
                    style = _LINE_STYLE
                chart.append(chr(0x2800 + raw_bits), style=style)
            elif tick is not None:
                chart.append("┈", style=_GRID_STYLE)
            else:
                chart.append(" ")
        if row != plot_h - 1:
            chart.append("\n")

    chart.append("\n")
    chart.append(" " * gutter + " └" + "─" * plot_w, style=_AXIS_STYLE)
    chart.append("\n")
    left = x_lo
    right = x_hi
    x_ticks = nice_ticks(x_lo, x_hi, count=3)
    span = right - left
    x_labels: list[tuple[float, str]] = []
    if span == 0:
        x_labels = [(0.0, _format_x(left)), (1.0, _format_x(right))]
    else:
        x_labels.append((0.0, _format_x(left)))
        for tick in x_ticks:
            frac = (tick - left) / span
            if 0.08 < frac < 0.92:
                x_labels.append((frac, _format_x(tick)))
        x_labels.append((1.0, _format_x(right)))
    chart.append(" " * (gutter + 2), style=_AXIS_STYLE)
    chart.append(_x_label_line(plot_w, x_labels), style=_X_LABEL_STYLE)
    return chart
