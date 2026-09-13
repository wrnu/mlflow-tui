from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label

from mlflow_tui.formatting import format_value
from mlflow_tui.marquee import ellipsize
from mlflow_tui.models import RunSummary
from mlflow_tui.widgets.footer import WrappingFooter
from mlflow_tui.widgets.table import MarqueeDataTable


class CompareScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    CSS = """
    CompareScreen {
        align: center middle;
        pointer: pointer;
    }

    #compare-box {
        width: 94%;
        height: 86%;
        background: #0e1620;
        border: tall #00b4d8;
        padding: 1 1;
    }

    #compare-title {
        text-style: bold;
        color: #7ee0ff;
        padding-bottom: 1;
        text-overflow: ellipsis;
        text-wrap: nowrap;
    }
    """

    def __init__(self, runs: list[RunSummary]) -> None:
        super().__init__()
        self.runs = runs

    def compose(self) -> ComposeResult:
        with Vertical(id="compare-box"):
            names = ", ".join(run.name for run in self.runs)
            yield Label(f"Compare  {names}", id="compare-title")
            yield MarqueeDataTable(id="compare", cursor_type="row", zebra_stripes=True)
            yield WrappingFooter()

    def on_mount(self) -> None:
        table = self.query_one("#compare", MarqueeDataTable)
        keys = ["key", *[run.id for run in self.runs]]
        table.marquee_columns = set(keys)
        table.add_column("Key", key="key", width=16)
        for run in self.runs:
            table.add_column(ellipsize(run.name, 14), key=run.id, width=14)

        metric_keys = sorted({key for run in self.runs for key in run.metrics})
        param_keys = sorted({key for run in self.runs for key in run.params})

        table.add_row("── metrics ──", *([""] * len(self.runs)), key="sec-metrics")
        for key in metric_keys:
            values = [
                format_value(run.metrics[key]) if key in run.metrics else "-" for run in self.runs
            ]
            table.add_row(key, *values, key=f"m-{key}")
        table.add_row("── params ──", *([""] * len(self.runs)), key="sec-params")
        for key in param_keys:
            table.add_row(key, *[run.params.get(key, "-") for run in self.runs], key=f"p-{key}")

    def on_click(self, event: events.Click) -> None:
        if event.widget is self:
            self.dismiss()
