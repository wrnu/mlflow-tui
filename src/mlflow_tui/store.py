from __future__ import annotations

from typing import Any

from mlflow_tui.auth import apply_tracking_auth, auth_hint
from mlflow_tui.models import (
    Artifact,
    Experiment,
    MetricPoint,
    RunSummary,
    TrackingError,
    TrackingStore,
)


def _coerce_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def run_from_mlflow(run: Any) -> RunSummary:
    info = run.info
    data = run.data
    tags = dict(getattr(data, "tags", {}) or {})
    name = (
        getattr(info, "run_name", None)
        or tags.get("mlflow.runName")
        or str(getattr(info, "run_id", ""))[:8]
    )
    metrics = {
        str(key): _coerce_float(value)
        for key, value in dict(getattr(data, "metrics", {}) or {}).items()
    }
    params = {
        str(key): str(value) for key, value in dict(getattr(data, "params", {}) or {}).items()
    }
    return RunSummary(
        id=str(info.run_id),
        name=str(name),
        status=str(info.status),
        start_time=getattr(info, "start_time", None),
        end_time=getattr(info, "end_time", None),
        metrics=metrics,
        params=params,
        tags=tags,
        artifact_uri=str(getattr(info, "artifact_uri", "") or ""),
    )


class MlflowTrackingStore:
    def __init__(self, tracking_uri: str | None = None) -> None:
        try:
            from mlflow.entities import ViewType
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise TrackingError(
                "mlflow is not installed. Install mlflow-skinny or mlflow "
                "to talk to a tracking server."
            ) from exc

        kwargs: dict[str, str] = {}
        tracking_uri = apply_tracking_auth(tracking_uri, prompt_password=False)
        if tracking_uri:
            kwargs["tracking_uri"] = tracking_uri
        try:
            self._client = MlflowClient(**kwargs)
        except Exception as exc:
            raise TrackingError(f"Could not connect to MLflow: {exc}") from exc
        self._view_type = ViewType
        self.tracking_uri = tracking_uri or getattr(self._client, "tracking_uri", "") or "(default)"

    def list_experiments(self) -> list[Experiment]:
        try:
            pages = self._search_experiments()
        except Exception as exc:
            raise TrackingError(_with_auth_hint("Failed to list experiments", exc)) from exc
        return [
            Experiment(
                id=str(
                    getattr(exp, "experiment_id", None)
                    or getattr(exp, "experimentId", None)
                    or getattr(exp, "id", "")
                ),
                name=str(getattr(exp, "name", "") or ""),
                artifact_location=str(getattr(exp, "artifact_location", "") or ""),
            )
            for exp in pages
            if str(
                getattr(exp, "experiment_id", None)
                or getattr(exp, "experimentId", None)
                or getattr(exp, "id", "")
            )
        ]

    def _search_experiments(self) -> list[Any]:
        try:
            return self._paginate(
                lambda token: self._client.search_experiments(
                    view_type=self._view_type.ACTIVE_ONLY,
                    max_results=1000,
                    order_by=["last_update_time DESC"],
                    **({"page_token": token} if token else {}),
                ),
                limit=2000,
            )
        except Exception:
            return self._paginate(
                lambda token: self._client.search_experiments(
                    view_type=self._view_type.ACTIVE_ONLY,
                    max_results=1000,
                    **({"page_token": token} if token else {}),
                ),
                limit=2000,
            )

    def list_runs(self, experiment_id: str, filter_string: str = "") -> list[RunSummary]:
        try:
            pages = self._paginate(
                lambda token: self._client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=filter_string or "",
                    run_view_type=self._view_type.ACTIVE_ONLY,
                    max_results=500,
                    order_by=["attributes.start_time DESC"],
                    **({"page_token": token} if token else {}),
                ),
                limit=1000,
            )
        except Exception as exc:
            raise TrackingError(_with_auth_hint("Failed to list runs", exc)) from exc
        return [run_from_mlflow(run) for run in pages]

    def metric_history(self, run_id: str, key: str) -> list[MetricPoint]:
        try:
            history = list(self._client.get_metric_history(run_id, key))
        except Exception as exc:
            raise TrackingError(_with_auth_hint(f"Failed to load metric '{key}'", exc)) from exc
        return [
            MetricPoint(
                key=str(getattr(point, "key", key)),
                value=_coerce_float(getattr(point, "value", 0.0)),
                timestamp=int(getattr(point, "timestamp", 0) or 0),
                step=int(getattr(point, "step", 0) or 0),
            )
            for point in history
        ]

    def list_artifacts(self, run_id: str, path: str = "") -> list[Artifact]:
        try:
            files = list(self._client.list_artifacts(run_id, path or None))
        except TypeError:
            files = list(self._client.list_artifacts(run_id, path))
        except Exception as exc:
            raise TrackingError(_with_auth_hint("Failed to list artifacts", exc)) from exc
        artifacts = []
        for info in files:
            artifacts.append(
                Artifact(
                    path=str(getattr(info, "path", "") or ""),
                    is_dir=bool(getattr(info, "is_dir", False)),
                    size=getattr(info, "file_size", None),
                )
            )
        return artifacts

    def delete_run(self, run_id: str) -> None:
        try:
            self._client.delete_run(run_id)
        except Exception as exc:
            raise TrackingError(_with_auth_hint("Failed to delete run", exc)) from exc

    def _paginate(self, fetch: Any, *, limit: int) -> list[Any]:
        items: list[Any] = []
        token = None
        for _ in range(50):
            page = fetch(token)
            items.extend(page)
            token = getattr(page, "token", None)
            if isinstance(token, str):
                token = token.strip() or None
            if not token or len(items) >= limit:
                break
        return items[:limit]


def _with_auth_hint(prefix: str, exc: BaseException) -> str:
    detail = f"{prefix}: {exc}"
    hint = auth_hint(exc)
    if hint:
        return f"{detail}. {hint}"
    return detail


def open_store(tracking_uri: str | None) -> TrackingStore:
    return MlflowTrackingStore(tracking_uri)
