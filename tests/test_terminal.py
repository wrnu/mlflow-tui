from __future__ import annotations

import os
import sys

from mlflow_tui.terminal import (
    install_quiet_driver,
    keep_terminal_quiet,
    prepare_terminal,
    should_enable_mouse,
)


def test_explicit_mouse_flags_win(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TUI_MOUSE", "0")
    assert should_enable_mouse(True) is True
    assert should_enable_mouse(False) is False


def test_env_disables_mouse(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TUI_MOUSE", "off")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert should_enable_mouse(None) is False


def test_mouse_defaults_on(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TUI_MOUSE", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert should_enable_mouse(None) is True
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert should_enable_mouse(None) is True


def test_dumb_term_defaults_mouse_off(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TUI_MOUSE", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert should_enable_mouse(None) is False


def test_prepare_terminal_disables_probes_and_truecolor(monkeypatch) -> None:
    monkeypatch.delenv("TEXTUAL_DISABLE_KITTY_KEY", raising=False)
    monkeypatch.delenv("TEXTUAL_COLOR_SYSTEM", raising=False)
    monkeypatch.setenv("COLORTERM", "truecolor")
    prepare_terminal()
    assert os.environ.get("TEXTUAL_DISABLE_KITTY_KEY") == "1"
    assert os.environ.get("TEXTUAL_COLOR_SYSTEM") == "256"
    assert os.environ.get("COLORTERM") == "256"


class _Driver:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self._mouse = True
        self._in_band_window_resize = True
        self.enable_calls = 0
        self.pixel_calls = 0
        self.processed: list[object] = []

    def write(self, data: str) -> None:
        self.writes.append(data)

    def flush(self) -> None:
        return

    def _enable_mouse_support(self) -> None:
        self.enable_calls += 1

    def _disable_mouse_support(self) -> None:
        self._mouse = False
        self.writes.append("disabled")

    def _enable_mouse_pixels(self) -> None:
        self.pixel_calls += 1

    def _query_in_band_window_resize(self) -> None:
        self.writes.append("query-inband")

    def _request_terminal_sync_mode_support(self) -> None:
        self.writes.append("query-sync")

    def process_message(self, message: object) -> None:
        self.processed.append(message)


class InBandWindowResize:
    pass


class MouseMove:
    pass


class MouseScrollDown:
    pass


class Click:
    pass


def test_install_quiet_driver_enables_taps_skips_probes() -> None:
    driver = _Driver()
    install_quiet_driver(driver, allow_mouse=lambda: True)
    assert driver._in_band_window_resize is False
    driver._enable_mouse_support()
    assert driver.enable_calls == 0
    blob = "".join(driver.writes)
    assert "1000h" in blob
    assert "1003h" in blob
    assert "1007h" in blob
    driver._enable_mouse_pixels()
    assert driver.pixel_calls == 0
    driver._query_in_band_window_resize()
    assert "query-inband" not in driver.writes
    driver.write("\x1b[?1016h")
    assert "1016h" not in "".join(driver.writes)
    driver.process_message(InBandWindowResize())
    assert driver.processed == []
    driver.process_message(MouseMove())
    driver.process_message(Click())
    driver.process_message(MouseScrollDown())
    assert len(driver.processed) == 3


def test_keep_terminal_quiet_enables_touch() -> None:
    driver = _Driver()
    keep_terminal_quiet(driver, allow_mouse=True)
    blob = "".join(driver.writes)
    assert "1000h" in blob
    assert "1003h" in blob
    assert "1007h" in blob
    assert "1003l" not in blob


def test_keep_terminal_quiet_turns_mouse_off() -> None:
    driver = _Driver()
    keep_terminal_quiet(driver, allow_mouse=False)
    assert driver._mouse is False
    blob = "".join(driver.writes)
    assert "1003l" in blob
    assert "1000l" in blob
    assert "disabled" in blob
