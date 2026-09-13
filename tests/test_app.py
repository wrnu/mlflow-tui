from __future__ import annotations

import asyncio

from textual.events import Paste
from textual.widgets import DataTable, Input, OptionList

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
            assert app.query_one("#sidebar").size.width <= 22

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


def test_mouse_down_selects_a_run_row() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.selected_run_id and len(app.runs) > 1:
                    break
            first = app.selected_run_id
            await pilot.mouse_down("#runs", offset=(8, 4))
            await pilot.pause()
            assert app.selected_run_id is not None
            assert app.selected_run_id != first

    asyncio.run(_run())


def test_unfocused_paste_does_not_enter_filter() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test() as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.experiments:
                    break
            filt = app.query_one("#filter", Input)
            assert not filt.has_focus
            app.post_message(Paste("hello-from-paste"))
            await pilot.pause()
            assert filt.value == ""

    asyncio.run(_run())


def test_unfocused_printable_keys_do_not_enter_filter() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test() as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.experiments:
                    break
            filt = app.query_one("#filter", Input)
            assert not filt.has_focus
            await pilot.press("h", "e", "l", "l", "o")
            await pilot.pause()
            assert filt.value == ""
            filt.focus()
            await pilot.press("h", "e", "l", "l", "o")
            await pilot.pause()
            assert filt.value == "hello"

    asyncio.run(_run())
