from __future__ import annotations

import os
import sys

_WEB_TERMINALS = {
    "vscode",
    "vscodeinsiders",
    "cursor",
    "hyper",
}


def prepare_terminal() -> None:
    """Skip Kitty keyboard probes. Webviews echo those CSI replies as glyphs."""
    os.environ.setdefault("TEXTUAL_DISABLE_KITTY_KEY", "1")


def should_enable_mouse(explicit: bool | None = None) -> bool:
    """Mouse reporting uses any-event tracking (1003), which touch webviews flood."""
    if explicit is not None:
        return explicit
    env = os.environ.get("MLFLOW_TUI_MOUSE")
    if env is not None and env.strip() != "":
        return env.strip().lower() not in {"0", "false", "no", "off"}
    if not sys.stdin.isatty():
        return False
    term = os.environ.get("TERM", "")
    if term in {"", "dumb"}:
        return False
    program = os.environ.get("TERM_PROGRAM", "").lower()
    if program in _WEB_TERMINALS:
        return False
    return True
