from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TypeVar

from rich.text import Text

T = TypeVar("T")

SPARK_CHARS = "▁▂▃▄▅▆▇█"

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


def _draw_line(grid: list[list[str]], x0: int, y0: int, x1: int, y1: int) -> None:
    height = len(grid)
    width = len(grid[0]) if grid else 0
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if 0 <= y < height and 0 <= x < width:
            if dy == 0:
                grid[y][x] = "─"
            elif dx == 0:
                grid[y][x] = "│"
            elif (sy > 0) == (sx > 0):
                grid[y][x] = "╲"
            else:
                grid[y][x] = "╱"
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def render_line_chart(
    values: Sequence[float],
    *,
    width: int,
    height: int,
    title: str = "",
    x0: int | None = None,
    x1: int | None = None,
) -> str:
    if not values:
        return "No metric history"
    if width < 16 or height < 5:
        head = f"{title}  " if title else ""
        return f"{head}{sparkline(values, max(8, width - 2))}"

    plot_h = max(3, height - (2 if title else 1) - 1)
    plot_w = max(8, width - 9)
    sampled = downsample(list(values), plot_w)
    lo = min(sampled)
    hi = max(sampled)
    if hi == lo:
        hi = lo + max(abs(lo) * 0.1, 1e-6)
    pad = (hi - lo) * 0.08
    lo -= pad
    hi += pad
    span = hi - lo

    coords: list[tuple[int, int]] = []
    last = max(len(sampled) - 1, 1)
    for i, value in enumerate(sampled):
        col = round(i * (plot_w - 1) / last)
        row = round((hi - value) / span * (plot_h - 1))
        coords.append((col, row))

    grid = [[" " for _ in range(plot_w)] for _ in range(plot_h)]
    for i in range(len(coords) - 1):
        _draw_line(grid, *coords[i], *coords[i + 1])
    for col, row in coords:
        if 0 <= row < plot_h and 0 <= col < plot_w:
            grid[row][col] = "●"

    lines: list[str] = []
    if title:
        lines.append(title)
    label_every = max(1, (plot_h - 1) // 3)
    for row in range(plot_h):
        y_val = hi - span * row / max(plot_h - 1, 1)
        label = f"{format_value(y_val):>7}"
        tick = "┤" if row % label_every == 0 or row == plot_h - 1 else "│"
        lines.append(f"{label} {tick}{''.join(grid[row])}")

    left = str(x0 if x0 is not None else 0)
    right = str(x1 if x1 is not None else len(values) - 1)
    lines.append(" " * 8 + "└" + "─" * plot_w)
    gap = max(1, plot_w - len(left))
    lines.append(" " * 8 + left + right.rjust(gap))
    return "\n".join(lines)
