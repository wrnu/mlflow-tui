from __future__ import annotations

import asyncio

from rich.text import Text
from textual.events import Paste
from textual.widgets import DataTable, Input, OptionList

from mlflow_tui.app import MLFlowTui
from mlflow_tui.demo import DemoTrackingStore
from mlflow_tui.screens.delete import DeleteRunScreen
from mlflow_tui.screens.help import HelpScreen
from mlflow_tui.widgets.footer import WrappingFooter
from mlflow_tui.widgets.plot import MetricPlot

DASHBOARD_FOOTER = (
    "Next pane",
    "Prev pane",
    "Filter",
    "Sidebar",
    "Next",
    "Prev",
    "Focus",
    "Keys",
    "Quit",
)
FOCUS_FOOTER = (
    "Back",
    "Next",
    "Prev",
    "LogY",
    "Smooth",
    "Keys",
    "Quit",
)


def _footer_descriptions(footer: WrappingFooter) -> list[str]:
    return [item.description for item in footer.query("FooterKey")]


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


def test_runs_table_keeps_a_usable_height_when_short() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(80, 22)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.runs:
                    break
            runs = app.query_one("#runs", DataTable)
            assert runs.size.height >= 8

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


def test_e_toggles_the_experiments_sidebar() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.runs:
                    break
            sidebar = app.query_one("#sidebar")
            assert sidebar.display
            await pilot.press("e")
            await pilot.pause()
            assert app.sidebar_hidden
            assert app.screen.has_class("sidebar-hidden")
            assert not sidebar.display
            await pilot.press("slash")
            await pilot.pause()
            assert not app.sidebar_hidden
            assert sidebar.display
            assert isinstance(app.focused, Input)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("e")
            await pilot.pause()
            assert app.sidebar_hidden
            assert not sidebar.display
            assert not isinstance(app.focused, Input)

    asyncio.run(_run())


def test_runs_arrow_keys_keep_rows_and_move_cursor() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.runs and app.selected_run_id:
                    break
            table = app.query_one("#runs", DataTable)
            table.focus()
            await pilot.pause()
            count = table.row_count
            first = table.cursor_row
            assert count >= 2
            await pilot.press("down")
            await pilot.pause()
            assert table.row_count == count
            assert table.cursor_row != first
            await pilot.press("up")
            await pilot.pause()
            assert table.row_count == count
            assert table.cursor_row == first
            await pilot.press("j")
            await pilot.pause()
            assert table.cursor_row != first
            await pilot.press("i")
            await pilot.pause()
            assert table.cursor_row == first

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


def test_l_toggles_log_scale_on_plot() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            plot = app.query_one("#plot", MetricPlot)
            for _ in range(40):
                await pilot.pause()
                if plot._ys:
                    break
            assert not plot.log_y
            await pilot.press("l")
            await pilot.pause()
            assert plot.log_y
            rendered = plot.render()
            plain = rendered.plain if isinstance(rendered, Text) else str(rendered)
            assert "log" in plain
            await pilot.press("l")
            await pilot.pause()
            assert not plot.log_y

    asyncio.run(_run())


def test_s_toggles_ema_smoothing_on_plot() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            plot = app.query_one("#plot", MetricPlot)
            for _ in range(40):
                await pilot.pause()
                if plot._ys:
                    break
            assert not plot.smooth
            await pilot.press("s")
            await pilot.pause()
            assert plot.smooth
            rendered = plot.render()
            plain = rendered.plain if isinstance(rendered, Text) else str(rendered)
            assert "smooth" in plain
            await pilot.press("s")
            await pilot.pause()
            assert not plot.smooth

    asyncio.run(_run())


def test_m_and_n_step_metrics_forward_and_back() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.runs and app.plot_metric and len(app.metric_keys) > 1:
                    break
            first = app.plot_metric
            await pilot.press("m")
            await pilot.pause()
            assert app.plot_metric != first
            await pilot.press("n")
            await pilot.pause()
            assert app.plot_metric == first

    asyncio.run(_run())


def test_question_mark_opens_help_and_dashboard_footer_pairs_navigation() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            footer = app.query_one(WrappingFooter)
            for _ in range(40):
                await pilot.pause()
                if app.runs:
                    break
            assert tuple(_footer_descriptions(footer)) == DASHBOARD_FOOTER
            await pilot.press("question_mark")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            help_footer = app.screen.query_one(WrappingFooter)
            assert _footer_descriptions(help_footer) == ["Close"]
            table = app.screen.query_one("#help-keys", DataTable)
            assert table.row_count >= 10
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, HelpScreen)

    asyncio.run(_run())


def test_graph_focus_footer_shows_back_and_graph_keys() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            plot = app.query_one("#plot", MetricPlot)
            footer = app.query_one(WrappingFooter)
            for _ in range(40):
                await pilot.pause()
                if plot._ys:
                    break
            await pilot.press("f")
            await pilot.pause()
            assert app.focused_view
            assert plot.has_focus
            assert tuple(_footer_descriptions(footer)) == FOCUS_FOOTER
            await pilot.press("escape")
            await pilot.pause()
            assert not app.focused_view
            assert tuple(_footer_descriptions(footer)) == DASHBOARD_FOOTER

    asyncio.run(_run())


def test_w_and_b_cycle_panes() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.runs:
                    break
            first = app.focused
            await pilot.press("w")
            await pilot.pause()
            second = app.focused
            assert second is not first
            await pilot.press("b")
            await pilot.pause()
            assert app.focused is first
            await pilot.press("f")
            await pilot.pause()
            focused = app.focused
            await pilot.press("w")
            await pilot.pause()
            assert app.focused is focused

    asyncio.run(_run())


def test_graph_focus_zoom_and_pan_and_reset() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            plot = app.query_one("#plot", MetricPlot)
            for _ in range(40):
                await pilot.pause()
                if plot._ys:
                    break
            await pilot.press("f")
            await pilot.pause()
            assert app.focused_view
            assert plot.has_focus
            await pilot.press("equals")
            await pilot.pause()
            assert plot.x_span < 1.0
            assert plot.y_span < 1.0
            x_before = plot.x_start
            await pilot.press("right")
            await pilot.pause()
            assert plot.x_start > x_before
            await pilot.press("left_square_bracket")
            await pilot.pause()
            x_span = plot.x_span
            y_span_before = plot.y_span
            await pilot.press("shift+up")
            await pilot.pause()
            assert plot.y_span < y_span_before
            assert plot.x_span == x_span
            y_before = plot.y_start
            await pilot.press("i")
            await pilot.pause()
            assert plot.y_start > y_before
            await pilot.press("j")
            await pilot.pause()
            assert plot.y_start == y_before
            await pilot.press("0")
            await pilot.pause()
            assert plot.x_span == 1.0
            assert plot.y_span == 1.0
            assert plot.x_start == 0.0

    asyncio.run(_run())


def test_footer_wraps_to_multiple_rows_when_narrow() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            footer = app.query_one(WrappingFooter)
            for _ in range(40):
                await pilot.pause()
                if footer.size.width > 0:
                    break
            assert footer.size.height == 1
            await pilot.resize_terminal(36, 24)
            for _ in range(40):
                await pilot.pause()
                if footer.size.height >= 2:
                    break
            assert footer.size.height >= 2
            assert tuple(_footer_descriptions(footer)) == DASHBOARD_FOOTER

    asyncio.run(_run())


def test_d_opens_delete_confirm_and_escape_cancels() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.runs:
                    break
            assert app.selected_run_id == "0042"
            await pilot.press("d")
            await pilot.pause()
            assert isinstance(app.screen, DeleteRunScreen)
            footer = app.screen.query_one(WrappingFooter)
            assert _footer_descriptions(footer) == ["Delete", "Cancel"]
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, DeleteRunScreen)
            assert app.selected_run_id == "0042"
            assert [run.id for run in app.runs][0] == "0042"

    asyncio.run(_run())


def test_d_deletes_selected_run_after_confirm() -> None:
    app = MLFlowTui(store=DemoTrackingStore(seed=1), refresh_seconds=0)

    async def _run() -> None:
        async with app.run_test(size=(140, 42)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app.runs:
                    break
            assert app.selected_run_id == "0042"
            app.marked_run_ids.add("0042")
            await pilot.press("d")
            await pilot.pause()
            assert isinstance(app.screen, DeleteRunScreen)
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if app.selected_run_id == "0041":
                    break
            assert app.selected_run_id == "0041"
            assert all(run.id != "0042" for run in app.runs)
            assert "0042" not in app.marked_run_ids

    asyncio.run(_run())
