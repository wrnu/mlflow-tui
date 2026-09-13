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


def test_graph_focus_hides_chrome_and_restores() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.runs:
                    break
            sidebar = app.query_one("#sidebar")
            runs = app.query_one("#runs", DataTable)
            assert sidebar.display
            assert runs.display
            await pilot.press("f")
            assert app.focused_view
            assert app.screen.has_class("graph-focus")
            assert not sidebar.display
            assert not runs.display
            assert app.query_one("#plot").has_focus
            await pilot.press("escape")
            assert not app.focused_view
            assert sidebar.display
            assert runs.display

    asyncio.run(_run())


def test_mouse_click_cycles_metric_and_double_click_focuses_plot() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.runs and app.plot_metric:
                    break
            first = app.plot_metric
            assert first is not None
            await pilot.click("#plot", offset=(4, 2))
            await pilot.pause()
            assert app.plot_metric != first
            await pilot.click("#plot", offset=(4, 2), times=2)
            await pilot.pause()
            assert app.focused_view

    asyncio.run(_run())


def test_mouse_can_mark_run_without_keyboard_focus() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.selected_run_id:
                    break
            run_id = app.selected_run_id
            assert run_id is not None
            app.query_one("#plot").focus()
            await pilot.click("#runs", offset=(8, 2), control=True)
            await pilot.pause()
            assert app.marked_run_ids

    asyncio.run(_run())
