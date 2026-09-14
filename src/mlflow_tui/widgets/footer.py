from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rich.cells import cell_len
from textual.app import ComposeResult
from textual.binding import Binding
from textual.geometry import NULL_OFFSET, Region, Size, Spacing
from textual.layout import ArrangeResult, Layout, WidgetPlacement
from textual.widgets._footer import Footer as _Footer
from textual.widgets._footer import FooterKey

if TYPE_CHECKING:
    from textual.widget import Widget

DASHBOARD_ACTIONS: tuple[str, ...] = (
    "focus_next",
    "focus_previous",
    "focus_filter",
    "next_metric",
    "prev_metric",
    "toggle_graph_focus",
    "show_help",
    "quit",
)
FOCUS_ACTIONS: tuple[str, ...] = (
    "exit_graph_focus",
    "next_metric",
    "prev_metric",
    "toggle_log_scale",
    "toggle_smooth",
    "show_help",
    "quit",
)
MODAL_ACTIONS: tuple[str, ...] = ("dismiss",)


def action_name(action: str) -> str:
    return action.rsplit(".", 1)[-1]


def footer_actions(*, modal: bool, graph_focus: bool) -> tuple[str, ...]:
    if modal:
        return MODAL_ACTIONS
    if graph_focus:
        return FOCUS_ACTIONS
    return DASHBOARD_ACTIONS


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


class FooterActionKey(FooterKey):
    """Footer button that runs the action instead of synthesizing a key chord."""

    def on_mouse_down(self) -> None:
        if self._disabled:
            self.app.bell()
            return
        name = action_name(self.action)
        namespace = self.screen if name == "dismiss" else self.app
        self.app.call_next(self.app.run_action, self.action, namespace)


class WrappingFooter(_Footer):
    """Tappable keys for the current screen, wrapping instead of scrolling away."""

    DEFAULT_CSS = """
    WrappingFooter {
        height: auto;
        min-height: 1;
        overflow: hidden hidden;
        scrollbar-size: 0 0;
    }
    WrappingFooter FooterKey.-command-palette {
        display: none;
    }
    """
    _wrap_layout = WrapLayout()

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("show_command_palette", False)
        super().__init__(*args, **kwargs)

    @property
    def layout(self) -> Layout:
        return self._wrap_layout

    def compose(self) -> ComposeResult:
        if not self._bindings_ready:
            return
        allowed = footer_actions(
            modal=self.screen.is_modal,
            graph_focus=bool(getattr(self.app, "focused_view", False)),
        )
        by_action: dict[str, Binding] = {}
        for node in (self.screen, self.app):
            for _key, binding in node._bindings:
                if not binding.show:
                    continue
                name = action_name(binding.action)
                if name not in allowed or name in by_action:
                    continue
                by_action[name] = binding
        for action in allowed:
            binding = by_action.get(action)
            if binding is None:
                continue
            yield FooterActionKey(
                binding.key,
                self.app.get_key_display(binding),
                binding.description,
                binding.action,
                tooltip=binding.tooltip,
            ).data_bind(compact=_Footer.compact)
