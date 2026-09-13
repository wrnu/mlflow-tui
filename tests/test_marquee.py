from __future__ import annotations

from rich.cells import cell_len

from mlflow_tui.marquee import ellipsize, marquee_offset, marquee_slice, sidebar_width


def test_sidebar_width_hugs_then_caps() -> None:
    assert sidebar_width(["ab"]) == 16
    assert sidebar_width(["transformer-baseline"]) == 22
    assert sidebar_width(["Default"]) <= 22


def test_ellipsize_fits_or_marks() -> None:
    assert ellipsize("hello", 10) == "hello"
    clipped = ellipsize("hello-world", 8)
    assert clipped.endswith("…")
    assert cell_len(clipped) == 8


def test_marquee_slice_scrolls_and_loops() -> None:
    assert cell_len(marquee_slice("abcdefghij", 4, 0)) == 4
    assert marquee_slice("abcdefghij", 4, 0) == "abcd"
    assert marquee_slice("abcdefghij", 4, 1) == "bcde"
    looped = marquee_slice("abcd", 4, 5, gap=2)
    assert cell_len(looped) == 4


def test_marquee_offset_pauses_then_advances() -> None:
    assert marquee_offset(1, 20, 8) == 0
    assert marquee_offset(6, 20, 8) == 0
    assert marquee_offset(7, 20, 8) == 1
    assert marquee_offset(0, 5, 20) == 0
