from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Label

from mlflow_tui.widgets.footer import WrappingFooter

KEY_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "App",
        (
            ("tab", "Cycle panes"),
            ("/", "Filter experiments and runs"),
            ("r", "Refresh"),
            ("m", "Cycle the plotted metric"),
            ("l", "Toggle log Y (symlog if the series includes 0 or negatives)"),
            ("space", "Mark a run"),
            ("c", "Compare marked runs"),
            ("y", "Copy the selected run ID"),
            ("f", "Focus the graph (zoom and pan keys work here)"),
            ("esc", "Leave graph focus"),
            ("?", "This help"),
            ("q", "Quit"),
        ),
    ),
    (
        "Graph  ·  press f first",
        (
            ("= / +", "Zoom in (both axes)"),
            ("-", "Zoom out (both axes)"),
            ("[", "Zoom X in"),
            ("]", "Zoom X out"),
            ("i", "Zoom Y in  (also shift+up)"),
            ("o", "Zoom Y out  (also shift+down)"),
            ("arrows", "Pan the zoomed window"),
            ("0", "Reset zoom"),
            ("wheel", "Zoom in / out"),
            ("drag", "Pan"),
        ),
    ),
    (
        "Mouse",
        (
            ("click run", "Select"),
            ("ctrl/double-click run", "Mark for compare"),
            ("click plot", "Next metric"),
            ("double-click plot", "Focus / unfocus graph"),
            ("click footer key", "Same as the binding"),
        ),
    ),
)


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close", show=False),
    ]

    CSS = """
    HelpScreen {
        align: center middle;
        pointer: pointer;
    }

    #help-box {
        width: 94%;
        height: 86%;
        background: #0e1620;
        border: tall #00b4d8;
        padding: 1 1;
    }

    #help-title {
        text-style: bold;
        color: #7ee0ff;
        padding-bottom: 1;
        height: auto;
    }

    #help-keys {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-box"):
            yield Label("Keys", id="help-title")
            yield DataTable(id="help-keys", cursor_type="row", zebra_stripes=True)
            yield WrappingFooter()

    def on_mount(self) -> None:
        table = self.query_one("#help-keys", DataTable)
        table.add_column("Key", key="key", width=22)
        table.add_column("Action", key="action", width=48)
        for section, rows in KEY_GROUPS:
            table.add_row(section, "", key=f"sec-{section}")
            for key, action in rows:
                table.add_row(key, action, key=f"{section}:{key}")
        table.focus()

    def on_click(self, event: events.Click) -> None:
        if event.widget is self:
            self.dismiss()
