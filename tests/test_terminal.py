from __future__ import annotations

import os
import sys

from mlflow_tui.terminal import prepare_terminal, should_enable_mouse


def test_explicit_mouse_flags_win(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TUI_MOUSE", "0")
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    assert should_enable_mouse(True) is True
    assert should_enable_mouse(False) is False


def test_env_disables_mouse(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TUI_MOUSE", "off")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    assert should_enable_mouse(None) is False


def test_web_preview_defaults_mouse_off(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TUI_MOUSE", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    assert should_enable_mouse(None) is False
    monkeypatch.setenv("TERM_PROGRAM", "cursor")
    assert should_enable_mouse(None) is False


def test_desktop_tty_defaults_mouse_on(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TUI_MOUSE", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    assert should_enable_mouse(None) is True


def test_prepare_terminal_disables_kitty_probes(monkeypatch) -> None:
    monkeypatch.delenv("TEXTUAL_DISABLE_KITTY_KEY", raising=False)
    prepare_terminal()
    assert os.environ.get("TEXTUAL_DISABLE_KITTY_KEY") == "1"
