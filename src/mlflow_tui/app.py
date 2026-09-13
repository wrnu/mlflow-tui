from __future__ import annotations

from datetime import datetime, timezone

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
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
from mlflow_tui.models import Artifact, Experiment, RunSummary, TrackingError, TrackingStore
from mlflow_tui.screens.compare import CompareScreen
from mlflow_tui.widgets.plot import MetricPlot


def _option_experiment_id(option_id: str | None) -> str | None:
    if not option_id or option_id.startswith("__"):
        return None
    if option_id.startswith("exp:"):
        return option_id[4:] or None
    return option_id


class RunsTable(DataTable):
    """Runs list: ctrl-click or double-click marks a run for compare."""

    async def _on_click(self, event: events.Click) -> None:
        await super()._on_click(event)
        if event.ctrl or event.chain >= 2:
            action_name = "action_toggle_mark"
            toggle = getattr(self.app, action_name, None)
            if callable(toggle):
                toggle()


class MLFlowTui(App[None]):
    TITLE = "mlflow-tui"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("slash", "focus_filter", "Filter"),
        Binding("m", "next_metric", "Metric"),
        Binding("space", "toggle_mark", "Mark"),
        Binding("c", "compare", "Compare"),
        Binding("y", "copy_run_id", "Yank ID"),
        Binding("f", "toggle_graph_focus", "Focus"),
        Binding("escape", "exit_graph_focus", "Back", show=False),
    ]
    focused_view: reactive[bool] = reactive(False, init=False)

    def __init__(
        self,
        store: TrackingStore,
        *,
        refresh_seconds: float = 3.0,
        initial_experiment: str | None = None,
    ) -> None:
        super().__init__()
        self.store = store
        self.refresh_seconds = refresh_seconds
        self.initial_experiment = initial_experiment
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("Experiments", classes="pane-title", id="experiments-title")
                yield Input(placeholder="Filter…  /  or metrics.loss < 0.1", id="filter")
                yield OptionList(id="experiments")
            with Vertical(id="main"):
                yield Label("Runs", classes="pane-title", id="runs-title")
                yield RunsTable(id="runs", cursor_type="row", zebra_stripes=True)
                with Horizontal(id="detail"):
                    with Vertical(id="plot-pane"):
                        yield Label("Select a run", id="run-meta")
                        yield MetricPlot(id="plot")
                    with TabbedContent(id="kv-tabs"):
                        with TabPane("Params", id="tab-params"):
                            yield DataTable(id="params", cursor_type="row", zebra_stripes=True)
                        with TabPane("Tags", id="tab-tags"):
                            yield DataTable(id="tags", cursor_type="row", zebra_stripes=True)
                        with TabPane("Artifacts", id="tab-artifacts"):
                            yield Tree("artifacts", id="artifacts")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = self.store.tracking_uri
        runs = self.query_one("#runs", DataTable)
        runs.cursor_type = "row"
        runs.zebra_stripes = True
        for table_id in ("params", "tags"):
            table = self.query_one(f"#{table_id}", DataTable)
            table.add_columns("Key", "Value")
        if self.refresh_seconds > 0:
            self.set_interval(self.refresh_seconds, self.silent_refresh)
        self.query_one("#experiments", OptionList).focus()
        self.query_one("#experiments", OptionList).add_option(
            Option("Loading experiments…", id="__loading", disabled=True)
        )
        self.query_one("#plot", MetricPlot).tooltip = (
            "Click: next metric · wheel: cycle · double-click: focus graph"
        )
        self.query_one("#run-meta", Label).tooltip = "Click to cycle the plotted metric"
        self.query_one("#runs", DataTable).tooltip = (
            "Click to select · ctrl-click or double-click to mark"
        )
        self.reload_experiments()

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

    def action_focus_filter(self) -> None:
        if self.focused_view:
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
        self.load_run_detail(run_id)

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
            self.notify(
                "Mark at least two runs (space, ctrl-click, or double-click), then compare"
            )
            return
        self.push_screen(CompareScreen(runs))

    def action_copy_run_id(self) -> None:
        if not self.selected_run_id:
            return
        self.copy_to_clipboard(self.selected_run_id)
        self.notify(f"Copied {self.selected_run_id}")

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
        self.notify(f"Plotting {self.plot_metric}")

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
        option_list = self.query_one("#experiments", OptionList)
        previous = self.selected_experiment_id
        option_list.clear_options()
        if not experiments:
            uri = getattr(self.store, "tracking_uri", "") or ""
            label = "No experiments"
            if uri:
                label = f"No experiments at {uri}"
            option_list.add_option(Option(label, id="__empty", disabled=True))
            self.query_one("#plot", MetricPlot).clear_series()
            self.query_one("#runs-title", Label).update("Runs")
            return
        for experiment in experiments:
            option_list.add_option(Option(experiment.name, id=f"exp:{experiment.id}"))
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
        table.add_column("Name", key="name")
        table.add_column("Status", key="status")
        table.add_column("Age", key="age")
        table.add_column("Duration", key="duration")
        for metric in metrics:
            table.add_column(metric, key=f"metric-{metric}")
        if show_gpu:
            table.add_column("GPU", key="gpu")
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
            table.add_row("—", "none")
            return
        for key, value in items:
            table.add_row(key, value)

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
