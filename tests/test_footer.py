from __future__ import annotations

from mlflow_tui.widgets.footer import footer_actions, footer_item_width, pack_footer_rows


def test_pack_footer_rows_wraps_when_items_overflow() -> None:
    assert pack_footer_rows([3, 3, 3], 7) == [[0, 1], [2]]
    assert pack_footer_rows([3, 3, 3], 9) == [[0, 1, 2]]
    assert pack_footer_rows([5, 5], 4) == [[0], [1]]


def test_pack_footer_rows_empty_and_unknown_width() -> None:
    assert pack_footer_rows([], 40) == []
    assert pack_footer_rows([4, 5, 6], 0) == [[0, 1, 2]]


def test_footer_item_width_includes_padding() -> None:
    assert footer_item_width("q", "Quit", compact=False) == 8
    assert footer_item_width("q", "Quit", compact=True) == 7


def test_footer_actions_depend_on_screen() -> None:
    dashboard = footer_actions(modal=False, graph_focus=False)
    focus = footer_actions(modal=False, graph_focus=True)
    assert "focus_next" in dashboard and "focus_previous" in dashboard
    assert "next_metric" in dashboard and "prev_metric" in dashboard
    assert "toggle_graph_focus" in dashboard
    assert "toggle_sidebar" in dashboard
    assert "toggle_sidebar" not in focus
    assert "exit_graph_focus" not in dashboard
    assert "zoom_in" not in dashboard
    assert "focus_next" not in focus
    assert "focus_filter" not in focus
    assert "toggle_graph_focus" not in focus
    assert focus[0] == "exit_graph_focus"
    assert "zoom_in" not in focus
    assert "reset_view" not in focus
    assert footer_actions(modal=True, graph_focus=True) == ("confirm", "dismiss")
