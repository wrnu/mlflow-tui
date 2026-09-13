from __future__ import annotations

from mlflow_tui.widgets.footer import footer_item_width, pack_footer_rows


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
