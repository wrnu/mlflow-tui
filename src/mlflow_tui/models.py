from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Experiment:
    id: str
    name: str
    artifact_location: str = ""


@dataclass
class RunSummary:
    id: str
    name: str
    status: str
    start_time: int | None
    end_time: int | None
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    artifact_uri: str = ""


@dataclass(frozen=True)
class MetricPoint:
    key: str
    value: float
    timestamp: int
    step: int


@dataclass(frozen=True)
class Artifact:
    path: str
    is_dir: bool
    size: int | None = None


class TrackingError(Exception):
    """Raised when the tracking backend cannot complete a request."""


class TrackingStore(Protocol):
    tracking_uri: str

    def list_experiments(self) -> list[Experiment]: ...

    def list_runs(self, experiment_id: str, filter_string: str = "") -> list[RunSummary]: ...

    def metric_history(self, run_id: str, key: str) -> list[MetricPoint]: ...

    def list_artifacts(self, run_id: str, path: str = "") -> list[Artifact]: ...

    def delete_run(self, run_id: str) -> None: ...
