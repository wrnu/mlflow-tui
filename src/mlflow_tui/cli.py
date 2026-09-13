from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

from mlflow_tui import __version__


def _package_version() -> str:
    try:
        return version("mlflow-tui")
    except PackageNotFoundError:
        return __version__


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mlflow-tui",
        description="Terminal UI for MLflow tracking.",
    )
    parser.add_argument(
        "--tracking-uri",
        "-u",
        default=None,
        help="Tracking URI (file store or server). Defaults to MLFLOW_TRACKING_URI.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Open a built-in demo workspace (no tracking server required).",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=3.0,
        help="Seconds between live refreshes. Set 0 to disable. Default: 3.",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        default=None,
        help="Select an experiment by name or ID on startup.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.refresh < 0:
        print("error: --refresh must be >= 0", file=sys.stderr)
        raise SystemExit(2)

    if args.demo:
        from mlflow_tui.demo import DemoTrackingStore

        store = DemoTrackingStore()
    else:
        from mlflow_tui.store import open_store

        try:
            store = open_store(args.tracking_uri)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            print(
                "hint: try `mlflow-tui --demo` to explore the UI without a tracking server",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

    from mlflow_tui.app import MLFlowTui

    app = MLFlowTui(
        store=store,
        refresh_seconds=args.refresh,
        initial_experiment=args.experiment,
    )
    app.run()


if __name__ == "__main__":
    main()
