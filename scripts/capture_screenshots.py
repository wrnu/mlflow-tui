#!/usr/bin/env python3
"""Capture README screenshots from the demo workspace.

Requires `rsvg-convert` (librsvg) to rasterize Textual's SVG screenshots.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from mlflow_tui.app import MLFlowTui
from mlflow_tui.demo import DemoTrackingStore
from mlflow_tui.widgets.plot import MetricPlot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
SIZE = (140, 40)


async def _wait_for_plot(app: MLFlowTui, pilot, *, attempts: int = 50) -> None:
    plot = app.query_one("#plot", MetricPlot)
    for _ in range(attempts):
        await pilot.pause()
        if plot._ys:
            await pilot.pause()
            return
    raise RuntimeError("demo plot never loaded")


def _svg_to_png(svg_path: Path) -> None:
    png_path = svg_path.with_suffix(".png")
    subprocess.run(
        ["rsvg-convert", "--zoom", "2", "-o", str(png_path), str(svg_path)],
        check=True,
    )


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)
    async with app.run_test(size=SIZE) as pilot:
        await _wait_for_plot(app, pilot)

        app.save_screenshot("dashboard.svg", path=str(OUT))

        await pilot.press("f")
        await pilot.pause()
        app.save_screenshot("graph.svg", path=str(OUT))

    for name in ("dashboard", "graph"):
        _svg_to_png(OUT / f"{name}.svg")
        (OUT / f"{name}.svg").unlink()


if __name__ == "__main__":
    asyncio.run(main())
