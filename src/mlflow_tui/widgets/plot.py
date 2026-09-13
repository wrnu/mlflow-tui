from __future__ import annotations

from textual.widgets import Static

from mlflow_tui.formatting import render_line_chart
from mlflow_tui.models import MetricPoint


class MetricPlot(Static):
    """Terminal line chart for a single metric series."""

    DEFAULT_CSS = """
    MetricPlot {
        width: 1fr;
        height: 1fr;
        overflow: hidden;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("Select a run to plot metrics.", markup=False, **kwargs)
        self._name = ""
        self._steps: list[int] = []
        self._ys: list[float] = []

    def clear_series(self) -> None:
        self._name = ""
        self._steps = []
        self._ys = []
        self.update("Select a run to plot metrics.")

    def set_series(self, name: str, points: list[MetricPoint]) -> None:
        self._name = name
        self._steps = [point.step for point in points]
        self._ys = [point.value for point in points]
        self._render_plot()

    def on_resize(self) -> None:
        self._render_plot()

    def _render_plot(self) -> None:
        if not self._ys:
            self.update("Select a run to plot metrics.")
            return
        width = max(self.size.width, 0)
        height = max(self.size.height, 0)
        chart = render_line_chart(
            self._ys,
            width=width,
            height=height,
            title=self._name,
            x0=self._steps[0] if self._steps else 0,
            x1=self._steps[-1] if self._steps else 0,
        )
        self.update(chart)
