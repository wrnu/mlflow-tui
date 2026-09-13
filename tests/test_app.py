from __future__ import annotations

import asyncio

from textual.widgets import DataTable, OptionList

from mlflow_tui.app import MLFlowTui
from mlflow_tui.demo import DemoTrackingStore


def test_demo_app_mounts_core_widgets() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test() as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.experiments and app.runs:
                    break
            names = [exp.name for exp in app.experiments]
            assert names == ["LT-JEPA", "transformer-baseline", "ablations"]
            assert app.query_one("#experiments", OptionList).option_count == 3
            assert app.query_one("#runs", DataTable).row_count >= 3
            assert app.selected_run_id == "0042"

    asyncio.run(_run())
