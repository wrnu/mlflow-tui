from __future__ import annotations

import os
import sys

from mlflow_tui.terminal import (
    finalize_driver_bytes,
    install_quiet_driver,
    keep_probes_off,
    keep_terminal_quiet,
    prepare_terminal,
    scrub_driver_bytes,
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


def test_prepare_terminal_disables_probes_keeps_color(monkeypatch) -> None:
    monkeypatch.setenv("TEXTUAL_DISABLE_KITTY_KEY", "0")
    monkeypatch.setenv("TEXTUAL_COLOR_SYSTEM", "truecolor")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.delenv("TEXTUAL_SMOOTH_SCROLL", raising=False)
    prepare_terminal()
    assert os.environ.get("TEXTUAL_DISABLE_KITTY_KEY") == "1"
    assert os.environ.get("TEXTUAL_COLOR_SYSTEM") == "truecolor"
    assert os.environ.get("COLORTERM") == "truecolor"
    assert os.environ.get("TEXTUAL_SMOOTH_SCROLL") == "0"


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


class TerminalSupportsSynchronizedOutput:
    pass


def test_scrub_driver_bytes_strips_sync_and_pixel_probes() -> None:
    assert "2026" not in scrub_driver_bytes("\x1b[?2026h")
    assert scrub_driver_bytes("\x1b[?2026hframe\x1b[?2026l") == "frame"
    assert "1016h" not in scrub_driver_bytes("\x1b[?1016h")
    assert scrub_driver_bytes("\x1b[>7u") == ""
    assert scrub_driver_bytes("ok") == "ok"


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
    driver.process_message(TerminalSupportsSynchronizedOutput())
    assert len(driver.processed) == 3


def test_keep_probes_off_does_not_toggle_mouse() -> None:
    driver = _Driver()
    driver._mouse = True
    keep_probes_off(driver)
    assert driver._mouse is True
    assert driver.writes == []


def test_quiet_sequences_never_query_sync_or_inband_resize() -> None:
    driver = _Driver()
    keep_terminal_quiet(driver, allow_mouse=True)
    blob = "".join(driver.writes)
    assert "2026" not in blob
    assert "2048" not in blob
    driver = _Driver()
    install_quiet_driver(driver, allow_mouse=lambda: True)
    driver._enable_mouse_support()
    blob = "".join(driver.writes)
    assert "2026" not in blob
    assert "2048" not in blob


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


def test_finalize_hides_cursor_and_drops_parked_cup() -> None:
    frame = "\x1b[10;1Hdone\x1b[34;40H"
    out = finalize_driver_bytes(frame)
    assert out.endswith("\x1b[?25l")
    assert "34;40H" not in out
    assert "\x1b[10;1Hdone" in out
    assert "\x1b[?25h" not in finalize_driver_bytes("ok\x1b[?25h")


def test_finalize_restores_cursor_when_leaving_alt_screen() -> None:
    out = finalize_driver_bytes("\x1b[?1049l\x1b[?25h", restore_cursor=True)
    assert "\x1b[?25h" in out
    assert not out.endswith("\x1b[?25l")


def test_finalize_strips_newlines_that_scroll_after_torn_cup() -> None:
    frame = "\x1b[10;1Hdone\n\x1b[11;1Hnext"
    out = finalize_driver_bytes(frame)
    assert "\n" not in out
    assert "\x1b[10;1Hdone" in out
    assert "\x1b[11;1Hnext" in out


def test_quiet_write_parks_hidden_cursor() -> None:
    driver = _Driver()
    install_quiet_driver(driver, allow_mouse=lambda: True)
    driver.write("\x1b[2;1Hcell\x1b[34;40H")
    blob = "".join(driver.writes)
    assert "34;40H" not in blob
    assert blob.endswith("\x1b[?25l")
    driver.write("\x1b[?1049l")
    driver.write("\x1b[?25h")
    assert "\x1b[?25h" in "".join(driver.writes)
