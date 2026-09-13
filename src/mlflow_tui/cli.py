from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

from mlflow_tui import __version__
from mlflow_tui.auth import apply_tracking_auth, auth_hint
from mlflow_tui.terminal import prepare_terminal, should_enable_mouse


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
        help="Tracking URI (file store or server). Defaults to MLFLOW_TRACKING_URI. "
        "This is -u; username is --username / --user.",
    )
    parser.add_argument(
        "--username",
        "--user",
        dest="username",
        default=None,
        help="Username for HTTP basic auth. Defaults to MLFLOW_TRACKING_USERNAME.",
    )
    parser.add_argument(
        "--password",
        "-p",
        dest="password",
        default=None,
        metavar="PASSWORD",
        help="HTTP basic auth password (command line). "
        "If omitted, MLFLOW_TRACKING_PASSWORD is used, otherwise you are prompted. "
        "With uv, put this after the command: uv run mlflow-tui -p 'secret' "
        "(uv run -p is --python).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token. Defaults to MLFLOW_TRACKING_TOKEN. "
        "Basic auth takes precedence if a username is also set.",
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
        "--mouse",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable mouse reporting. Web/mobile terminals (T3, VS Code) flood "
        "mouse-move sequences that show up as flashing glyphs; those default off. "
        "Desktop terminals keep mouse on. Override with --mouse / --no-mouse.",
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
            tracking_uri = apply_tracking_auth(
                args.tracking_uri,
                username=args.username,
                password=args.password,
                token=args.token,
            )
            store = open_store(tracking_uri)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            hint = auth_hint(exc)
            if hint:
                print(f"hint: {hint}", file=sys.stderr)
            print(
                "hint: try `mlflow-tui --demo` to explore the UI without a tracking server",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

    from mlflow_tui.app import MLFlowTui

    prepare_terminal()
    app = MLFlowTui(
        store=store,
        refresh_seconds=args.refresh,
        initial_experiment=args.experiment,
    )
    app.run(mouse=should_enable_mouse(args.mouse))


if __name__ == "__main__":
    main()
