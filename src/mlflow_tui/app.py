from __future__ import annotations

from datetime import datetime, timezone
from time import monotonic

from rich.cells import cell_len
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.coordinate import Coordinate
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Header,
    Input,
    Label,
    OptionList,
    TabbedContent,
    TabPane,
    Tree,
)
from textual.widgets.option_list import Option
from textual.widgets.tree import TreeNode
from textual.worker import get_current_worker

from mlflow_tui.formatting import (
    format_age,
    format_bytes,
    format_duration,
    format_value,
    gpu_label,
    is_mlflow_filter,
    pick_metric_columns,
    run_duration_ms,
    sparkline,
    status_text,
    visible_tags,
)
from mlflow_tui.marquee import ellipsize, marquee_offset, marquee_slice, sidebar_width
from mlflow_tui.models import Artifact, Experiment, RunSummary, TrackingError, TrackingStore
from mlflow_tui.screens.compare import CompareScreen
from mlflow_tui.screens.help import HelpScreen
from mlflow_tui.terminal import install_quiet_driver, keep_probes_off, keep_terminal_quiet
from mlflow_tui.widgets.footer import WrappingFooter
from mlflow_tui.widgets.plot import MetricPlot
from mlflow_tui.widgets.table import MarqueeDataTable


def _option_experiment_id(option_id: str | None) -> str | None:
    if not option_id or option_id.startswith("__"):
        return None
    if option_id.startswith("exp:"):
        return option_id[4:] or None
    return option_id


class RunsTable(MarqueeDataTable):
    """Runs list: tap/click selects; ctrl-click or double-click marks for compare."""

    marquee_columns = {"name"}

    def _scroll_cursor_into_view(self, animate: bool = False) -> None:
        super()._scroll_cursor_into_view(animate=False)

    def watch_cursor_coordinate(self, old_value, cursor_coordinate) -> None:
        super().watch_cursor_coordinate(old_value, cursor_coordinate)
        self.refresh(repaint=True)

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        await super()._on_mouse_down(event)
        meta = event.style.meta
        row_index = meta.get("row")
        column_index = meta.get("column", 0)
        if row_index is None:
            try:
                row_index, column_index = self.hover_coordinate
            except Exception:
                return
        if row_index < 0 or row_index >= self.row_count:
            return
        self.cursor_coordinate = Coordinate(row_index, max(0, column_index))

    async def _on_click(self, event: events.Click) -> None:
        await super()._on_click(event)
        if event.ctrl or event.chain >= 2:
            action_name = "action_toggle_mark"
            toggle = getattr(self.app, action_name, None)
            if callable(toggle):
                toggle()


class ExperimentList(OptionList):
    """Experiment list: tap/click highlights, which loads that experiment's runs."""

    BINDINGS = [
        Binding("i", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
    ]

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        await super()._on_mouse_down(event)
        option = event.style.meta.get("option")
        if option is None:
            option = getattr(self, "_mouse_hovering_over", None)
        if option is None:
            return
        try:
            if self.get_option_at_index(option).disabled:
                return
        except Exception:
            return
        self.highlighted = option


class ArtifactTree(Tree):
    BINDINGS = [
        Binding("i", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
    ]


class MLFlowTui(App[None]):
    TITLE = "mlflow-tui"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding("w", "focus_next", "Next pane"),
        Binding("b", "focus_previous", "Prev pane"),
        Binding("slash", "focus_filter", "Filter"),
        Binding("e", "toggle_sidebar", "Sidebar"),
        Binding("m", "next_metric", "Next"),
        Binding("n", "prev_metric", "Prev"),
        Binding("l", "toggle_log_scale", "LogY"),
        Binding("s", "toggle_smooth", "Smooth"),
        Binding("f", "toggle_graph_focus", "Focus"),
        Binding("escape", "exit_graph_focus", "Back"),
        Binding("question_mark", "show_help", "Keys"),
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh", show=False),
        Binding("space", "toggle_mark", "Mark", show=False),
        Binding("c", "compare", "Compare", show=False),
        Binding("y", "copy_run_id", "Yank ID", show=False),
    ]
    ENABLE_COMMAND_PALETTE = False
    focused_view: reactive[bool] = reactive(False, init=False, bindings=True)
    sidebar_hidden: reactive[bool] = reactive(False, init=False, bindings=True)

    def __init__(
        self,
        store: TrackingStore,
        *,
        refresh_seconds: float = 3.0,
        initial_experiment: str | None = None,
        mouse_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.store = store
        self.refresh_seconds = refresh_seconds
        self.initial_experiment = initial_experiment
        self.mouse_enabled = mouse_enabled
        self.experiments: list[Experiment] = []
        self.runs: list[RunSummary] = []
        self.runs_by_id: dict[str, RunSummary] = {}
        self.selected_experiment_id: str | None = None
        self.selected_run_id: str | None = None
        self.marked_run_ids: set[str] = set()
        self.metric_keys: list[str] = []
        self.plot_metric: str | None = None
        self._runs_seq = 0
        self._exp_seq = 0
        self._detail_seq = 0
        self._error: str | None = None
        self._exp_marquee_id: str | None = None
        self._exp_marquee_ticks = 0
        self._key_noise: list[float] = []
        self._last_quiet = 0.0
        self._last_repair = 0.0
        self._resize_repair_timer = None
        self._detail_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("Experiments", classes="pane-title", id="experiments-title")
                yield Input(placeholder="Filter…", id="filter")
                yield ExperimentList(id="experiments")
            with Vertical(id="main"):
                yield Label("Runs", classes="pane-title", id="runs-title")
                yield RunsTable(id="runs", cursor_type="row", zebra_stripes=True)
                with Horizontal(id="detail"):
                    with Vertical(id="plot-pane"):
                        yield Label("Select a run", id="run-meta")
                        yield MetricPlot(id="plot")
                    with TabbedContent(id="kv-tabs"):
                        with TabPane("Params", id="tab-params"):
                            yield MarqueeDataTable(
                                id="params", cursor_type="row", zebra_stripes=True
                            )
                        with TabPane("Tags", id="tab-tags"):
                            yield MarqueeDataTable(id="tags", cursor_type="row", zebra_stripes=True)
                        with TabPane("Artifacts", id="tab-artifacts"):
                            yield ArtifactTree("artifacts", id="artifacts")
        yield WrappingFooter()

    def on_mount(self) -> None:
        self.sub_title = self.store.tracking_uri
        runs = self.query_one("#runs", DataTable)
        runs.cursor_type = "row"
        runs.zebra_stripes = True
        for table_id in ("params", "tags"):
            table = self.query_one(f"#{table_id}", MarqueeDataTable)
            table.marquee_columns = {"key", "value"}
            table.add_column("Key", key="key", width=14)
            table.add_column("Value", key="value", width=20)
        if self.refresh_seconds > 0:
            self.set_interval(self.refresh_seconds, self.silent_refresh)
        self.set_interval(0.14, self._tick_experiment_marquee)
        self.set_interval(2.0, self._keep_probes_quiet)
        self.query_one("#experiments", OptionList).focus()
        self.query_one("#experiments", OptionList).add_option(
            Option("Loading experiments…", id="__loading", disabled=True)
        )
        self.query_one(
            "#plot", MetricPlot
        ).tooltip = "Click: next metric · double-click: focus · f: zoom/pan · l: LogY · s: smooth"
        self.query_one("#run-meta", Label).tooltip = "Click to cycle the plotted metric"
        self.query_one(
            "#runs", DataTable
        ).tooltip = "Click to select · ctrl-click or double-click to mark"
        self.query_one(
            "#filter", Input
        ).tooltip = "Substring or MLflow filter, e.g. metrics.loss < 0.1"
        self._install_quiet_input()
        self.call_after_refresh(self._install_quiet_input)
        self.reload_experiments()

    def watch_sidebar_hidden(self, hidden: bool) -> None:
        self.screen.set_class(hidden, "sidebar-hidden")
        if hidden and not self.focused_view:
            focused = self.focused
            if focused is not None and any(
                node.id == "sidebar" for node in focused.ancestors_with_self
            ):
                self.query_one("#runs", DataTable).focus()

    def action_toggle_sidebar(self) -> None:
        if self.focused_view:
            return
        self.sidebar_hidden = not self.sidebar_hidden

    def watch_focused_view(self, focused: bool) -> None:
        self.screen.set_class(focused, "graph-focus")
        plot = self.query_one("#plot", MetricPlot)
        if focused:
            plot.focus()
        else:
            self.query_one("#runs", DataTable).focus()
        self.call_after_refresh(plot._render_plot)

    def action_toggle_graph_focus(self) -> None:
        if not self.focused_view and not self.selected_run_id:
            self.notify("Select a run first")
            return
        self.focused_view = not self.focused_view

    def action_exit_graph_focus(self) -> None:
        if self.focused_view:
            self.focused_view = False
            return
        if isinstance(self.focused, Input):
            self.query_one("#runs", DataTable).focus()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in {"focus_next", "focus_previous"} and self.focused_view:
            return False
        return True

    def action_focus_filter(self) -> None:
        if self.focused_view:
            return
        if self.sidebar_hidden:
            self.sidebar_hidden = False
            self.call_after_refresh(self.query_one("#filter", Input).focus)
            return
        self.query_one("#filter", Input).focus()

    def action_refresh(self) -> None:
        self.reload_experiments(silent=False)
        self.notify("Refreshing")

    def silent_refresh(self) -> None:
        if self.selected_experiment_id:
            self.reload_runs(self.selected_experiment_id, silent=True)
        else:
            self.reload_experiments(silent=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "filter":
            return
        self.reload_experiments(silent=True)
        if self.selected_experiment_id:
            self.reload_runs(self.selected_experiment_id, silent=True)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "filter":
            self.query_one("#runs", DataTable).focus()

    def on_click(self, event: events.Click) -> None:
        widget_id = getattr(event.widget, "id", None)
        if widget_id == "experiments-title":
            self.query_one("#experiments", OptionList).focus()
        elif widget_id == "runs-title":
            self.query_one("#runs", DataTable).focus()
        elif widget_id == "run-meta":
            self.action_next_metric()

    def on_resize(self, _event: events.Resize) -> None:
        keep_probes_off(getattr(self, "_driver", None))
        timer = getattr(self, "_resize_repair_timer", None)
        if timer is not None:
            timer.stop()
        self._resize_repair_timer = self.set_timer(0.2, self._repair_after_resize)

    def _repair_after_resize(self) -> None:
        self.refresh(repaint=True, layout=True)

    def on_app_focus(self, _event: events.AppFocus) -> None:
        keep_probes_off(getattr(self, "_driver", None))

    def on_app_blur(self, _event: events.AppBlur) -> None:
        keep_probes_off(getattr(self, "_driver", None))

    def on_key(self, event: events.Key) -> None:
        if isinstance(self.focused, Input):
            return
        if event.key in {
            "q",
            "r",
            "slash",
            "m",
            "l",
            "space",
            "c",
            "y",
            "f",
            "question_mark",
            "equals",
            "plus",
            "minus",
            "left_square_bracket",
            "right_square_bracket",
            "left_curly_bracket",
            "right_curly_bracket",
            "shift+up",
            "shift+down",
            "0",
            "escape",
            "tab",
            "enter",
            "up",
            "down",
            "left",
            "right",
            "home",
            "end",
            "pageup",
            "pagedown",
            "backspace",
            "delete",
            "shift+tab",
        }:
            return
        character = event.character or ""
        if not character or character.isalpha() or character.isspace():
            return
        event.stop()
        event.prevent_default()
        now = monotonic()
        self._key_noise.append(now)
        cutoff = now - 0.15
        self._key_noise = [stamp for stamp in self._key_noise if stamp >= cutoff]
        if len(self._key_noise) >= 8:
            self._keep_terminal_quiet()
            self._repair_screen()

    def on_paste(self, event: events.Paste) -> None:
        if not isinstance(self.focused, Input):
            event.stop()
            event.prevent_default()

    def _allow_mouse(self) -> bool:
        return self.mouse_enabled

    def _install_quiet_input(self) -> None:
        driver = getattr(self, "_driver", None)
        install_quiet_driver(driver, allow_mouse=self._allow_mouse)
        self._keep_terminal_quiet(force=True)

    def _build_driver(
        self, headless: bool, inline: bool, mouse: bool, size: tuple[int, int] | None
    ):
        driver = super()._build_driver(headless=headless, inline=inline, mouse=mouse, size=size)
        install_quiet_driver(driver, allow_mouse=self._allow_mouse)
        return driver

    def _on_terminal_supports_synchronized_output(self, _message: object) -> None:
        return

    def _keep_probes_quiet(self) -> None:
        keep_probes_off(getattr(self, "_driver", None))

    def _keep_terminal_quiet(self, *, force: bool = False) -> None:
        now = monotonic()
        if not force and now - self._last_quiet < 0.08:
            return
        self._last_quiet = now
        keep_terminal_quiet(getattr(self, "_driver", None), allow_mouse=self._allow_mouse())

    def _repair_screen(self) -> None:
        now = monotonic()
        if now - self._last_repair < 0.2:
            return
        self._last_repair = now
        self.refresh()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "experiments" or self.focused_view:
            return
        self.query_one("#runs", DataTable).focus()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "experiments":
            return
        option_id = getattr(event, "option_id", None) or getattr(event.option, "id", None)
        experiment_id = _option_experiment_id(option_id)
        if not experiment_id or experiment_id == self.selected_experiment_id:
            return
        self.selected_experiment_id = experiment_id
        self.selected_run_id = None
        self.reload_runs(self.selected_experiment_id)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "runs" or event.row_key is None:
            return
        run_id = str(getattr(event.row_key, "value", event.row_key))
        if run_id == self.selected_run_id:
            return
        self.selected_run_id = run_id
        timer = getattr(self, "_detail_timer", None)
        if timer is not None:
            timer.stop()
        self._detail_timer = self.set_timer(0.18, lambda: self.load_run_detail(run_id))

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        node = event.node
        data = node.data
        if data is None or not data.is_dir or not self.selected_run_id:
            return
        if node.children:
            return
        self.load_artifacts(self.selected_run_id, data.path, node)

    def action_toggle_mark(self) -> None:
        if not self.selected_run_id:
            return
        if self.selected_run_id in self.marked_run_ids:
            self.marked_run_ids.remove(self.selected_run_id)
        else:
            self.marked_run_ids.add(self.selected_run_id)
        self._render_runs(self.runs)

    def action_compare(self) -> None:
        runs = [
            self.runs_by_id[run_id] for run_id in self.marked_run_ids if run_id in self.runs_by_id
        ]
        if len(runs) < 2:
            self.notify("Mark at least two runs (space, ctrl-click, or double-click), then compare")
            return
        self.push_screen(CompareScreen(runs))

    def action_copy_run_id(self) -> None:
        if not self.selected_run_id:
            return
        self.copy_to_clipboard(self.selected_run_id)
        self.notify(f"Copied {self.selected_run_id}")

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_toggle_log_scale(self) -> None:
        plot = self.query_one("#plot", MetricPlot)
        plot.toggle_log_y()

    def action_toggle_smooth(self) -> None:
        plot = self.query_one("#plot", MetricPlot)
        plot.toggle_smooth()

    def action_next_metric(self) -> None:
        self._cycle_metric(1)

    def action_prev_metric(self) -> None:
        self._cycle_metric(-1)

    def _cycle_metric(self, delta: int) -> None:
        if not self.metric_keys:
            self.notify("No metrics on the selected run")
            return
        if self.plot_metric in self.metric_keys:
            index = (self.metric_keys.index(self.plot_metric) + delta) % len(self.metric_keys)
        else:
            index = 0
        self.plot_metric = self.metric_keys[index]
        if self.selected_run_id:
            self.load_run_detail(self.selected_run_id)

    def reload_experiments(self, *, silent: bool = False) -> None:
        self._exp_seq += 1
        self.load_experiments(self._exp_seq, silent)

    @work(thread=True, exclusive=True, group="experiments")
    def load_experiments(self, seq: int, silent: bool = False) -> None:
        try:
            experiments = self.store.list_experiments()
        except TrackingError as exc:
            if seq == self._exp_seq:
                self.call_from_thread(self._show_error, str(exc))
            return
        needle = self._filter_text()
        if needle and not is_mlflow_filter(needle):
            experiments = [exp for exp in experiments if needle in exp.name.lower()]
        worker = get_current_worker()
        if worker.is_cancelled or seq != self._exp_seq:
            return
        self.call_from_thread(self._apply_experiments, experiments, silent)

    def _apply_experiments(self, experiments: list[Experiment], silent: bool) -> None:
        self._error = None
        self.experiments = experiments
        self._exp_marquee_id = None
        self._exp_marquee_ticks = 0
        option_list = self.query_one("#experiments", OptionList)
        previous = self.selected_experiment_id
        option_list.clear_options()
        self._sync_sidebar_width()
        if not experiments:
            uri = getattr(self.store, "tracking_uri", "") or ""
            label = "No experiments"
            if uri:
                label = f"No experiments at {uri}"
            option_list.add_option(
                Option(
                    ellipsize(label, self._experiment_label_width()),
                    id="__empty",
                    disabled=True,
                )
            )
            self.query_one("#plot", MetricPlot).clear_series()
            self.query_one("#runs-title", Label).update("Runs")
            return
        label_width = self._experiment_label_width()
        for experiment in experiments:
            option_list.add_option(
                Option(
                    ellipsize(experiment.name, label_width),
                    id=f"exp:{experiment.id}",
                )
            )
        index = 0
        target = previous or self._experiment_id_by_name(self.initial_experiment)
        if target:
            for i, experiment in enumerate(experiments):
                if experiment.id == target or experiment.name == target:
                    index = i
                    break
        option_list.highlighted = index
        chosen = experiments[index].id
        if chosen != self.selected_experiment_id or not silent:
            self.selected_experiment_id = chosen
            self.reload_runs(chosen, silent=silent)
        self.call_after_refresh(self._relabel_experiments)

    def reload_runs(self, experiment_id: str, *, silent: bool = False) -> None:
        self._runs_seq += 1
        filter_string = ""
        needle = self._filter_text()
        if is_mlflow_filter(needle):
            filter_string = needle
        if not silent:
            self.query_one("#runs", DataTable).loading = True
        self.load_runs(self._runs_seq, experiment_id, filter_string)

    @work(thread=True, exclusive=True, group="runs")
    def load_runs(self, seq: int, experiment_id: str, filter_string: str) -> None:
        try:
            runs = self.store.list_runs(experiment_id, filter_string=filter_string)
        except TrackingError as exc:
            self.call_from_thread(self._show_error, str(exc))
            return
        needle = self._filter_text()
        if needle and not is_mlflow_filter(needle):
            runs = [
                run
                for run in runs
                if needle in run.name.lower()
                or needle in run.id.lower()
                or needle in run.status.lower()
            ]
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_runs, seq, experiment_id, runs)

    def _apply_runs(self, seq: int, experiment_id: str, runs: list[RunSummary]) -> None:
        if seq != self._runs_seq or experiment_id != self.selected_experiment_id:
            return
        table = self.query_one("#runs", DataTable)
        table.loading = False
        self.runs = runs
        self.runs_by_id = {run.id: run for run in runs}
        self._render_runs(runs)
        experiment = next((exp for exp in self.experiments if exp.id == experiment_id), None)
        name = experiment.name if experiment else experiment_id
        self.query_one("#runs-title", Label).update(f"Runs  ·  {name}  ·  {len(runs)}")
        if not runs:
            self.selected_run_id = None
            self.query_one("#plot", MetricPlot).clear_series()
            self.query_one("#run-meta", Label).update("No runs")
            return
        selected = self.selected_run_id if self.selected_run_id in self.runs_by_id else runs[0].id
        self.selected_run_id = selected
        try:
            table.move_cursor(row=table.get_row_index(selected))
        except (KeyError, ValueError):
            pass
        self.load_run_detail(selected)

    def _render_runs(self, runs: list[RunSummary]) -> None:
        table = self.query_one("#runs", DataTable)
        current = self.selected_run_id
        table.clear(columns=True)
        metric_names = [key for run in runs for key in run.metrics]
        metrics = pick_metric_columns(metric_names)
        show_gpu = any(gpu_label(run.tags, run.metrics) != "-" for run in runs)
        table.add_column("", key="mark", width=1)
        reserved = 1 + 8 + 6 + 8 + 7 * len(metrics) + (4 if show_gpu else 0) + 10
        name_width = 18
        if table.size.width:
            name_width = max(10, min(22, table.size.width - reserved))
        table.add_column("Name", key="name", width=name_width)
        table.add_column("Status", key="status", width=8)
        table.add_column("Age", key="age", width=6)
        table.add_column("Duration", key="duration", width=8)
        for metric in metrics:
            table.add_column(metric, key=f"metric-{metric}", width=8)
        if show_gpu:
            table.add_column("GPU", key="gpu", width=4)
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        for run in runs:
            mark = "▸" if run.id in self.marked_run_ids else " "
            row = [
                mark,
                run.name,
                status_text(run.status),
                format_age(run.start_time, now_ms),
                format_duration(run_duration_ms(run.start_time, run.end_time, now_ms)),
            ]
            for metric in metrics:
                row.append(format_value(run.metrics[metric]) if metric in run.metrics else "-")
            if show_gpu:
                row.append(gpu_label(run.tags, run.metrics))
            table.add_row(*row, key=run.id)
        if current and current in self.runs_by_id:
            try:
                table.move_cursor(row=table.get_row_index(current))
            except Exception:
                pass

    def load_run_detail(self, run_id: str) -> None:
        if run_id != self.selected_run_id:
            return
        run = self.runs_by_id.get(run_id)
        if not run:
            return
        keys = pick_metric_columns(list(run.metrics), limit=max(1, len(run.metrics)))
        self.metric_keys = keys or list(run.metrics)
        if self.plot_metric not in run.metrics:
            self.plot_metric = self.metric_keys[0] if self.metric_keys else None
        self._render_meta(run)
        self._render_kv("params", sorted(run.params.items()))
        self._render_kv("tags", visible_tags(run.tags))
        if self.plot_metric:
            self._detail_seq += 1
            self.fetch_metric_history(self._detail_seq, run_id, self.plot_metric)
        else:
            self.query_one("#plot", MetricPlot).clear_series()
        self.reset_artifacts(run)

    def _render_meta(self, run: RunSummary) -> None:
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        bits = [
            f"[b]{run.name}[/]",
            status_text(run.status).markup,
            f"[dim]{run.id}[/]",
        ]
        if "lr" in run.params:
            bits.append(f"lr {run.params['lr']}")
        if "epoch" in run.params:
            max_epochs = run.params.get("max_epochs")
            bits.append(f"epoch {run.params['epoch']}" + (f" / {max_epochs}" if max_epochs else ""))
        if "samples" in run.params:
            try:
                bits.append(f"samples {format_value(float(run.params['samples']))}")
            except ValueError:
                bits.append(f"samples {run.params['samples']}")
        if self.plot_metric and self.plot_metric in run.metrics:
            bits.append(
                f"{self.plot_metric} {format_value(run.metrics[self.plot_metric])} "
                f"{sparkline([run.metrics[self.plot_metric]], 8)}"
            )
        bits.append(f"age {format_age(run.start_time, now_ms)}")
        self.query_one("#run-meta", Label).update("  ·  ".join(bits))

    def _render_kv(self, table_id: str, items: list[tuple[str, str]]) -> None:
        table = self.query_one(f"#{table_id}", DataTable)
        table.clear()
        if not items:
            table.add_row("—", "none", key="empty")
            return
        for index, (key, value) in enumerate(items):
            table.add_row(key, value, key=str(index))

    def reset_artifacts(self, run: RunSummary) -> None:
        tree = self.query_one("#artifacts", Tree)
        tree.clear()
        tree.root.set_label(run.artifact_uri or "artifacts")
        tree.root.data = Artifact(path="", is_dir=True)
        tree.root.expand()
        self.load_artifacts(run.id, "", tree.root)

    @work(thread=True, exclusive=True, group="metrics")
    def fetch_metric_history(self, seq: int, run_id: str, key: str) -> None:
        try:
            points = self.store.metric_history(run_id, key)
        except TrackingError as exc:
            self.call_from_thread(self._show_error, str(exc))
            return
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_history, seq, run_id, key, points)

    def _apply_history(self, seq: int, run_id: str, key: str, points: list) -> None:
        if seq != self._detail_seq or run_id != self.selected_run_id:
            return
        self.query_one("#plot", MetricPlot).set_series(key, points)

    @work(thread=True, exclusive=False, group="artifacts")
    def load_artifacts(self, run_id: str, path: str, node: TreeNode[Artifact | None]) -> None:
        try:
            artifacts = self.store.list_artifacts(run_id, path)
        except TrackingError as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            return
        self.call_from_thread(self._apply_artifacts, run_id, node, artifacts)

    def _apply_artifacts(
        self, run_id: str, node: TreeNode[Artifact | None], artifacts: list[Artifact]
    ) -> None:
        if run_id != self.selected_run_id:
            return
        for artifact in artifacts:
            label = artifact.path.rsplit("/", 1)[-1] or artifact.path
            if artifact.is_dir:
                child = node.add(label, data=artifact, allow_expand=True)
                child.allow_expand = True
            else:
                size = format_bytes(artifact.size)
                node.add_leaf(f"{label}  [dim]{size}[/]", data=artifact)

    def _show_error(self, message: str) -> None:
        self._error = message
        self.query_one("#runs", DataTable).loading = False
        self.notify(message, severity="error")
        option_list = self.query_one("#experiments", OptionList)
        option_list.clear_options()
        option_list.add_option(Option(message, id="__error", disabled=True))

    def _sync_sidebar_width(self) -> None:
        names = [exp.name for exp in self.experiments]
        self.query_one("#sidebar").styles.width = sidebar_width(names)

    def _relabel_experiments(self) -> None:
        opts = self.query_one("#experiments", OptionList)
        width = self._experiment_label_width()
        for index, experiment in enumerate(self.experiments):
            if experiment.id == self._exp_marquee_id:
                continue
            try:
                opts.replace_option_prompt_at_index(index, ellipsize(experiment.name, width))
            except Exception:
                continue

    def _experiment_label_width(self) -> int:
        opts = self.query_one("#experiments", OptionList)
        return max(4, opts.size.width - 4)

    def _tick_experiment_marquee(self) -> None:
        try:
            opts = self.query_one("#experiments", OptionList)
        except NoMatches:
            return
        highlighted = opts.highlighted
        if highlighted is None:
            return
        try:
            option = opts.get_option_at_index(highlighted)
        except Exception:
            return
        exp_id = _option_experiment_id(option.id)
        if not exp_id:
            return
        experiment = next((item for item in self.experiments if item.id == exp_id), None)
        if not experiment:
            return
        width = self._experiment_label_width()
        if self._exp_marquee_id != exp_id:
            if self._exp_marquee_id:
                self._restore_experiment_prompt(self._exp_marquee_id)
            self._exp_marquee_id = exp_id
            self._exp_marquee_ticks = 0
        self._exp_marquee_ticks += 1
        if cell_len(experiment.name) <= width:
            return
        offset = marquee_offset(self._exp_marquee_ticks, cell_len(experiment.name), width)
        opts.replace_option_prompt_at_index(
            highlighted, marquee_slice(experiment.name, width, offset)
        )

    def _restore_experiment_prompt(self, exp_id: str) -> None:
        experiment = next((item for item in self.experiments if item.id == exp_id), None)
        if not experiment:
            return
        try:
            self.query_one("#experiments", OptionList).replace_option_prompt(
                f"exp:{exp_id}",
                ellipsize(experiment.name, self._experiment_label_width()),
            )
        except Exception:
            return

    def _filter_text(self) -> str:
        try:
            return self.query_one("#filter", Input).value.strip()
        except Exception:
            return ""

    def _experiment_id_by_name(self, name: str | None) -> str | None:
        if not name:
            return None
        for experiment in self.experiments:
            if experiment.name == name or experiment.id == name:
                return experiment.id
        return name
