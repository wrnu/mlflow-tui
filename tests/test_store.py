from __future__ import annotations

from mlflow_tui.demo import DemoTrackingStore
from mlflow_tui.store import run_from_mlflow


class _Info:
    run_id = "abc123def"
    run_name = "run-0042"
    status = "RUNNING"
    start_time = 1_000
    end_time = None
    artifact_uri = "file:///tmp/mlruns/1/abc"


class _Data:
    metrics = {"loss": 0.0381, "accuracy": 0.91}
    params = {"lr": "3e-4"}
    tags = {"mlflow.runName": "run-0042", "gpu": "91%"}


class _Run:
    info = _Info()
    data = _Data()


def test_run_from_mlflow() -> None:
    run = run_from_mlflow(_Run())
    assert run.id == "abc123def"
    assert run.name == "run-0042"
    assert run.status == "RUNNING"
    assert run.metrics["loss"] == 0.0381
    assert run.params["lr"] == "3e-4"


def test_demo_store_has_core_workspace() -> None:
    store = DemoTrackingStore(seed=1)
    experiments = store.list_experiments()
    assert [exp.name for exp in experiments] == ["LT-JEPA", "transformer-baseline", "ablations"]
    runs = store.list_runs("1")
    names = [run.name for run in runs]
    assert names[:3] == ["run-0042", "run-0041", "run-0040"]
    running = runs[0]
    assert running.status == "RUNNING"
    history = store.metric_history(running.id, "loss")
    assert len(history) > 10
    assert history[0].step == 0
    artifacts = store.list_artifacts(running.id)
    assert any(item.path == "model" and item.is_dir for item in artifacts)
    nested = store.list_artifacts(running.id, "model")
    assert any(item.path.endswith("weights.pt") for item in nested)


def test_demo_store_ticks_running_metrics() -> None:
    store = DemoTrackingStore(seed=1)
    first = store.list_runs("1")[0].metrics["loss"]
    later = store.list_runs("1")[0].metrics["loss"]
    assert later != first or store.metric_history("0042", "loss")[-1].step > 80


def test_demo_filter_is_local() -> None:
    store = DemoTrackingStore(seed=1)
    matches = store.list_runs("1", filter_string="failed")
    assert [run.name for run in matches] == ["run-0040"]
