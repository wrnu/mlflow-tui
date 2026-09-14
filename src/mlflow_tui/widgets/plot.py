from __future__ import annotations

from textual import events
from textual.binding import Binding
from textual.widgets import Static

from mlflow_tui.formatting import (
    PAN_STEP,
    ZOOM_IN,
    ZOOM_OUT,
    clamp_view,
    pan_view,
    render_line_chart,
    zoom_view,
)
from mlflow_tui.models import MetricPoint


class MetricPlot(Static):
    """Terminal line chart for a single metric series."""

    can_focus = True
    BINDINGS = [
        Binding("equals,plus", "zoom_in", "Zoom in", show=False),
        Binding("minus", "zoom_out", "Zoom out", show=False),
        Binding("left_square_bracket", "zoom_x_in", "Zoom X in", show=False),
        Binding("right_square_bracket", "zoom_x_out", "Zoom X out", show=False),
        Binding("shift+up", "zoom_y_in", "Zoom Y in", show=False),
        Binding("o,shift+down", "zoom_y_out", "Zoom Y out", show=False),
        Binding("left", "pan_left", "Pan left", show=False),
        Binding("right", "pan_right", "Pan right", show=False),
        Binding("up,i", "pan_up", "Pan up", show=False),
        Binding("down,j", "pan_down", "Pan down", show=False),
        Binding("0", "reset_view", "Reset zoom", show=False),
    ]

    DEFAULT_CSS = """
    MetricPlot {
        width: 1fr;
        height: 1fr;
        overflow: hidden;
        padding: 0 1;
        pointer: pointer;
        text-wrap: nowrap;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("Select a run to plot metrics.", markup=False, **kwargs)
        self._name = ""
        self._steps: list[int] = []
        self._ys: list[float] = []
        self._press: tuple[int, int] | None = None
        self._dragged = False
        self._last_size: tuple[int, int] | None = None
        self._queued_size: tuple[int, int] | None = None
        self._resize_timer = None
        self.log_y = False
        self.smooth = False
        self.x_start = 0.0
        self.x_span = 1.0
        self.y_start = 0.0
        self.y_span = 1.0

    def clear_series(self) -> None:
        self._name = ""
        self._steps = []
        self._ys = []
        self.reset_view(render=False)
        self.update("Select a run to plot metrics.")

    def set_series(self, name: str, points: list[MetricPoint]) -> None:
        if name != self._name:
            self.reset_view(render=False)
        self._name = name
        self._steps = [point.step for point in points]
        self._ys = [point.value for point in points]
        self._render_plot()

    def toggle_log_y(self) -> None:
        self.log_y = not self.log_y
        self._render_plot()

    def toggle_smooth(self) -> None:
        self.smooth = not self.smooth
        self._render_plot()

    def reset_view(self, *, render: bool = True) -> None:
        self.x_start = 0.0
        self.x_span = 1.0
        self.y_start = 0.0
        self.y_span = 1.0
        if render and self._ys:
            self._render_plot()

    def action_zoom_in(self) -> None:
        self._zoom("both", ZOOM_IN)

    def action_zoom_out(self) -> None:
        self._zoom("both", ZOOM_OUT)

    def action_zoom_x_in(self) -> None:
        self._zoom("x", ZOOM_IN)

    def action_zoom_x_out(self) -> None:
        self._zoom("x", ZOOM_OUT)

    def action_zoom_y_in(self) -> None:
        self._zoom("y", ZOOM_IN)

    def action_zoom_y_out(self) -> None:
        self._zoom("y", ZOOM_OUT)

    def action_pan_left(self) -> None:
        self._pan(-PAN_STEP, 0.0)

    def action_pan_right(self) -> None:
        self._pan(PAN_STEP, 0.0)

    def action_pan_up(self) -> None:
        self._pan(0.0, PAN_STEP)

    def action_pan_down(self) -> None:
        self._pan(0.0, -PAN_STEP)

    def action_reset_view(self) -> None:
        self.reset_view()

    def _zoom(self, axis: str, factor: float) -> None:
        if axis in {"x", "both"}:
            self.x_start, self.x_span = zoom_view(self.x_start, self.x_span, factor)
        if axis in {"y", "both"}:
            self.y_start, self.y_span = zoom_view(self.y_start, self.y_span, factor)
        self._render_plot()

    def _pan(self, dx: float, dy: float) -> None:
        self.x_start, self.x_span = pan_view(self.x_start, self.x_span, dx)
        self.y_start, self.y_span = pan_view(self.y_start, self.y_span, dy)
        self._render_plot()

    def on_resize(self) -> None:
        size = (self.size.width, self.size.height)
        if size == self._queued_size:
            return
        self._queued_size = size
        if self._resize_timer is not None:
            self._resize_timer.stop()
        self._resize_timer = self.set_timer(0.12, self._apply_resize)

    def _apply_resize(self) -> None:
        size = (self.size.width, self.size.height)
        if size == self._last_size:
            return
        self._last_size = size
        self._render_plot()

    def _call_app(self, action_name: str) -> None:
        action = getattr(self.app, action_name, None)
        if callable(action):
            action()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._press = (event.x, event.y)
        self._dragged = False

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._press is None:
            return
        dx = event.x - self._press[0]
        dy = event.y - self._press[1]
        if abs(dx) <= 1 and abs(dy) <= 1:
            return
        width = max(self.size.width, 1)
        height = max(self.size.height, 1)
        self._pan(-dx / width, dy / height)
        self._press = None
        self._dragged = True
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.stop()
        self._zoom("both", ZOOM_IN)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.stop()
        self._zoom("both", ZOOM_OUT)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        if self._dragged:
            self._dragged = False
            self._press = None
            return
        if self._press is not None:
            dx = abs(event.x - self._press[0])
            dy = abs(event.y - self._press[1])
            self._press = None
            if dx > 1 or dy > 1:
                return
        if event.chain >= 2:
            self._call_app("action_toggle_graph_focus")
        else:
            self._call_app("action_next_metric")

    def _render_plot(self) -> None:
        if not self._ys:
            self.update("Select a run to plot metrics.")
            return
        width = max(self.size.width, 0)
        height = max(self.size.height, 0)
        self.x_start, self.x_span = clamp_view(self.x_start, self.x_span)
        self.y_start, self.y_span = clamp_view(self.y_start, self.y_span)
        chart = render_line_chart(
            self._ys,
            xs=self._steps,
            width=width,
            height=height,
            title=self._name,
            x0=self._steps[0] if self._steps else 0,
            x1=self._steps[-1] if self._steps else 0,
            log_y=self.log_y,
            smooth=self.smooth,
            x_start=self.x_start,
            x_span=self.x_span,
            y_start=self.y_start,
            y_span=self.y_span,
        )
        self.update(chart)
