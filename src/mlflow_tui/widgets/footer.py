from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rich.cells import cell_len
from textual.geometry import NULL_OFFSET, Region, Size, Spacing
from textual.layout import ArrangeResult, Layout, WidgetPlacement
from textual.widgets._footer import Footer as _Footer

if TYPE_CHECKING:
    from textual.widget import Widget


def footer_item_width(key_display: str, description: str, *, compact: bool) -> int:
    """Visible width of a footer key, matching Textual's FooterKey padding."""
    if compact:
        desc = (1 + cell_len(description)) if description else 0
        return cell_len(key_display) + desc + 1
    extra = 3 if description else 2
    return cell_len(key_display) + cell_len(description) + extra


def pack_footer_rows(widths: Sequence[int], width: int) -> list[list[int]]:
    """Greedy wrap: keep items on a row until the next one would overflow."""
    if not widths:
        return []
    if width <= 0:
        return [list(range(len(widths)))]
    rows: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index, item_width in enumerate(widths):
        size = max(1, item_width)
        if current and used + size > width:
            rows.append(current)
            current = [index]
            used = size
        else:
            used += size
            current.append(index)
    if current:
        rows.append(current)
    return rows


class WrapLayout(Layout):
    """Left-to-right, then extra rows when the next child does not fit."""

    name = "wrap"

    def arrange(
        self, parent: Widget, children: list[Widget], size: Size, greedy: bool = True
    ) -> ArrangeResult:
        parent.pre_layout(self)
        displayed = [
            child for child in children if child.display and child.styles.overlay != "screen"
        ]
        if not displayed:
            return []
        viewport = parent.app.viewport_size
        widths = [
            max(1, child.get_content_width(size, viewport) if size.width else 8)
            for child in displayed
        ]
        rows = pack_footer_rows(widths, size.width)
        placements: list[WidgetPlacement] = []
        for row_index, row in enumerate(rows):
            x = 0
            for index in row:
                child = displayed[index]
                child_width = widths[index]
                placements.append(
                    WidgetPlacement(
                        Region(x, row_index, child_width, 1),
                        NULL_OFFSET,
                        Spacing(0, 0, 0, 0),
                        child,
                    )
                )
                x += child_width
        return placements


class WrappingFooter(_Footer):
    """Key bindings that wrap to extra rows instead of scrolling off-screen."""

    DEFAULT_CSS = """
    WrappingFooter {
        height: auto;
        min-height: 1;
        overflow: hidden hidden;
        scrollbar-size: 0 0;
    }
    WrappingFooter FooterKey.-command-palette {
        dock: none;
        padding-right: 0;
        border-left: none;
    }
    """
    _wrap_layout = WrapLayout()

    @property
    def layout(self) -> Layout:
        return self._wrap_layout
