from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label

from mlflow_tui.widgets.footer import WrappingFooter


class DeleteRunScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("enter", "confirm", "Delete"),
        Binding("y", "confirm", "Delete", show=False),
        Binding("escape", "dismiss", "Cancel"),
        Binding("n", "dismiss", "Cancel", show=False),
        Binding("q", "dismiss", "Cancel", show=False),
    ]

    CSS = """
    DeleteRunScreen {
        align: center middle;
        pointer: pointer;
    }

    #delete-box {
        width: 64;
        max-width: 94%;
        height: auto;
        background: #0e1620;
        border: tall #00b4d8;
        padding: 1 2;
    }

    #delete-title {
        text-style: bold;
        color: #7ee0ff;
        padding-bottom: 1;
    }

    #delete-name {
        text-style: bold;
        color: #f4f7fb;
    }

    #delete-id {
        color: #8aa0b2;
        padding-bottom: 1;
    }

    #delete-note {
        color: #c5d0da;
        padding-bottom: 1;
    }
    """

    def __init__(self, run_name: str, run_id: str) -> None:
        super().__init__()
        self.run_name = run_name
        self.run_id = run_id

    def compose(self) -> ComposeResult:
        with Vertical(id="delete-box"):
            yield Label("Delete run", id="delete-title")
            yield Label(self.run_name, id="delete-name")
            yield Label(self.run_id, id="delete-id")
            yield Label(
                "This marks the run deleted in MLflow (same as the tracking UI).",
                id="delete-note",
            )
            yield WrappingFooter()

    def action_confirm(self) -> None:
        self.dismiss(True)

    def on_click(self, event: events.Click) -> None:
        if event.widget is self:
            self.dismiss(False)
