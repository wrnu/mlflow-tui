from __future__ import annotations

import random
import threading
import time
from dataclasses import replace

from mlflow_tui.models import Artifact, Experiment, MetricPoint, RunSummary


def _now_ms() -> int:
    return int(time.time() * 1000)


def _curve(
    start: float,
    end: float,
    steps: int,
    rng: random.Random,
    noise: float = 0.0015,
) -> list[float]:
    values: list[float] = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        value = start * (end / start) ** t + rng.gauss(0, noise)
        values.append(max(value, 1e-6))
    return values


class DemoTrackingStore:
    """In-memory tracking store so the TUI can be tried without an MLflow server."""

    def __init__(self, seed: int = 42) -> None:
        self.tracking_uri = "demo://"
        self._lock = threading.Lock()
        self._rng = random.Random(seed)
        self._experiments = [
            Experiment(id="1", name="LT-JEPA"),
            Experiment(id="2", name="transformer-baseline"),
            Experiment(id="3", name="ablations"),
        ]
        self._runs: dict[str, list[RunSummary]] = {}
        self._history: dict[tuple[str, str], list[MetricPoint]] = {}
        self._artifacts: dict[str, list[Artifact]] = {}
        self._build()

    def list_experiments(self) -> list[Experiment]:
        return list(self._experiments)

    def list_runs(self, experiment_id: str, filter_string: str = "") -> list[RunSummary]:
        with self._lock:
            self._tick_running()
            runs = [self._copy_run(run) for run in self._runs.get(experiment_id, [])]
        if not filter_string:
            return runs
        needle = filter_string.lower()
        return [
            run
            for run in runs
            if needle in run.name.lower()
            or needle in run.id.lower()
            or needle in run.status.lower()
            or any(needle in value.lower() for value in run.params.values())
        ]

    def metric_history(self, run_id: str, key: str) -> list[MetricPoint]:
        with self._lock:
            self._tick_running()
            return list(self._history.get((run_id, key), []))

    def list_artifacts(self, run_id: str, path: str = "") -> list[Artifact]:
        artifacts = self._artifacts.get(run_id, [])
        prefix = path.rstrip("/")
        out: list[Artifact] = []
        for artifact in artifacts:
            parent = artifact.path.rsplit("/", 1)[0] if "/" in artifact.path else ""
            if prefix:
                if parent == prefix:
                    out.append(artifact)
            elif "/" not in artifact.path:
                out.append(artifact)
        return out

    def _copy_run(self, run: RunSummary) -> RunSummary:
        return replace(
            run,
            metrics=dict(run.metrics),
            params=dict(run.params),
            tags=dict(run.tags),
        )

    def _build(self) -> None:
        now = _now_ms()
        hour = 3_600_000
        self._runs["1"] = [
            self._make_run(
                run_id="0042",
                name="run-0042",
                status="RUNNING",
                start_time=now - 12 * 60_000,
                end_time=None,
                loss_start=0.11,
                loss_end=0.0381,
                steps=80,
                acc=0.91,
                params={"lr": "3e-4", "epoch": "17", "max_epochs": "50", "samples": "42100000"},
                tags={"gpu": "91%", "mlflow.user": "warren", "device": "A100"},
                extra_metrics={"gpu_util": 91.0, "epoch": 17.0, "samples": 42_100_000},
            ),
            self._make_run(
                run_id="0041",
                name="run-0041",
                status="FINISHED",
                start_time=now - 2 * hour,
                end_time=now - hour,
                loss_start=0.12,
                loss_end=0.0402,
                steps=120,
                acc=0.90,
                params={"lr": "3e-4", "epoch": "50", "max_epochs": "50", "samples": "120000000"},
                tags={"mlflow.user": "warren", "device": "A100"},
            ),
            self._make_run(
                run_id="0040",
                name="run-0040",
                status="FAILED",
                start_time=now - 26 * hour,
                end_time=now - 25 * hour,
                loss_start=0.14,
                loss_end=0.0891,
                steps=40,
                acc=0.72,
                params={"lr": "1e-3", "epoch": "9", "max_epochs": "50"},
                tags={"mlflow.user": "warren", "device": "A100"},
            ),
            self._make_run(
                run_id="0039",
                name="run-0039",
                status="FINISHED",
                start_time=now - 30 * hour,
                end_time=now - 28 * hour,
                loss_start=0.13,
                loss_end=0.0441,
                steps=100,
                acc=0.89,
                params={"lr": "1e-4", "epoch": "50", "max_epochs": "50"},
                tags={"mlflow.user": "warren"},
            ),
        ]
        self._runs["2"] = [
            self._make_run(
                run_id=f"t-{idx}",
                name=f"baseline-{idx}",
                status="FINISHED",
                start_time=now - (8 - idx) * hour,
                end_time=now - (7 - idx) * hour,
                loss_start=0.22,
                loss_end=0.05 + idx * 0.004,
                steps=80,
                acc=0.86 - idx * 0.01,
                params={"lr": "5e-4", "layers": str(6 + idx), "d_model": "512"},
                tags={"mlflow.user": "warren"},
            )
            for idx in range(4)
        ]
        self._runs["3"] = [
            self._make_run(
                run_id=f"ab-{idx}",
                name=f"ablation-{['patch', 'mask', 'ema', 'lr', 'crop', 'wd'][idx]}",
                status="FINISHED" if idx % 3 else "KILLED",
                start_time=now - (12 - idx) * hour,
                end_time=now - (11 - idx) * hour,
                loss_start=0.18,
                loss_end=0.06 + idx * 0.003,
                steps=60,
                acc=0.84,
                params={"lr": "3e-4", "variant": f"v{idx}"},
                tags={"mlflow.user": "warren"},
            )
            for idx in range(6)
        ]

    def _make_run(
        self,
        *,
        run_id: str,
        name: str,
        status: str,
        start_time: int,
        end_time: int | None,
        loss_start: float,
        loss_end: float,
        steps: int,
        acc: float,
        params: dict[str, str],
        tags: dict[str, str],
        extra_metrics: dict[str, float] | None = None,
    ) -> RunSummary:
        loss_values = _curve(loss_start, loss_end, steps, self._rng)
        acc_values = _curve(max(0.2, acc - 0.2), acc, steps, self._rng, noise=0.004)
        loss_history = [
            MetricPoint(key="loss", value=value, timestamp=start_time + i * 1000, step=i)
            for i, value in enumerate(loss_values)
        ]
        acc_history = [
            MetricPoint(key="accuracy", value=value, timestamp=start_time + i * 1000, step=i)
            for i, value in enumerate(acc_values)
        ]
        self._history[(run_id, "loss")] = loss_history
        self._history[(run_id, "accuracy")] = acc_history
        metrics = {"loss": loss_values[-1], "accuracy": acc_values[-1]}
        if extra_metrics:
            metrics.update(extra_metrics)
            for key, value in extra_metrics.items():
                self._history[(run_id, key)] = [
                    MetricPoint(key=key, value=value, timestamp=start_time, step=steps - 1)
                ]
        self._artifacts[run_id] = [
            Artifact(path="model", is_dir=True),
            Artifact(path="model/weights.pt", is_dir=False, size=84_000_000),
            Artifact(path="model/config.json", is_dir=False, size=1_204),
            Artifact(path="metrics.json", is_dir=False, size=420),
            Artifact(path="stdout.log", is_dir=False, size=12_400),
        ]
        tags = dict(tags)
        tags.setdefault("mlflow.runName", name)
        return RunSummary(
            id=run_id,
            name=name,
            status=status,
            start_time=start_time,
            end_time=end_time,
            metrics=metrics,
            params=params,
            tags=tags,
            artifact_uri=f"demo://{run_id}",
        )

    def _tick_running(self) -> None:
        for runs in self._runs.values():
            for run in runs:
                if run.status != "RUNNING":
                    continue
                history = self._history[(run.id, "loss")]
                last = history[-1]
                nxt = max(0.01, last.value * 0.997 + self._rng.gauss(0, 0.00025))
                step = last.step + 1
                point = MetricPoint(key="loss", value=nxt, timestamp=_now_ms(), step=step)
                history.append(point)
                if len(history) > 400:
                    del history[: len(history) - 400]
                run.metrics["loss"] = nxt
                gpu = 86 + self._rng.random() * 10
                run.metrics["gpu_util"] = gpu
                run.tags["gpu"] = f"{gpu:.0f}%"
                epoch = min(50, int(run.params.get("epoch", "17")) + (1 if step % 8 == 0 else 0))
                run.params["epoch"] = str(epoch)
                run.metrics["epoch"] = float(epoch)


def open_demo_store() -> DemoTrackingStore:
    return DemoTrackingStore()
