from __future__ import annotations

import os
import sys
from collections.abc import Callable

# Optional capability probes. Terminals that do not implement them echo the
# query (or the reply) into the alt screen as glyphs.
_PROBES_OFF = "\x1b[?1004l\x1b[?1016l\x1b[?2048l\x1b[?25l"
_MOUSE_OFF = "\x1b[?1000l\x1b[?1003l\x1b[?1006l\x1b[?1015l" + _PROBES_OFF


def prepare_terminal() -> None:
    """Skip Kitty keyboard probes. Unsupported terminals echo those CSI replies."""
    os.environ.setdefault("TEXTUAL_DISABLE_KITTY_KEY", "1")


def should_enable_mouse(explicit: bool | None = None) -> bool:
    """Mouse is on for a real tty unless the user or env disables it."""
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
    return True


def keep_terminal_quiet(driver: object, *, allow_mouse: bool) -> None:
    """Re-assert cbreak and optional-probe-off after a resize or mode reset."""
    if driver is None:
        return
    if not allow_mouse:
        disable = getattr(driver, "_disable_mouse_support", None)
        if callable(disable):
            disable()
        if hasattr(driver, "_mouse"):
            driver._mouse = False
    if hasattr(driver, "_in_band_window_resize"):
        driver._in_band_window_resize = False
    write = getattr(driver, "write", None)
    if callable(write):
        write(_PROBES_OFF if allow_mouse else _MOUSE_OFF)
        flush = getattr(driver, "flush", None)
        if callable(flush):
            flush()
    reassert_cbreak(driver)


def install_quiet_driver(
    driver: object,
    *,
    allow_mouse: Callable[[], bool],
) -> None:
    """Keep optional CSI probes off, and mouse off once the app has disabled it."""
    if driver is None or getattr(driver, "_mlflow_tui_quiet", False):
        return
    driver._mlflow_tui_quiet = True
    if hasattr(driver, "_in_band_window_resize"):
        driver._in_band_window_resize = False
    orig_enable = getattr(driver, "_enable_mouse_support", None)
    orig_disable = getattr(driver, "_disable_mouse_support", None)

    def enable_mouse() -> None:
        if allow_mouse() and callable(orig_enable):
            orig_enable()
            return
        if callable(orig_disable):
            orig_disable()
        write = getattr(driver, "write", None)
        if callable(write):
            write(_MOUSE_OFF)
            flush = getattr(driver, "flush", None)
            if callable(flush):
                flush()

    def drop() -> None:
        return

    driver._enable_mouse_support = enable_mouse
    driver._enable_mouse_pixels = drop
    driver._query_in_band_window_resize = drop
    driver._enable_in_band_window_resize = drop
    driver._request_terminal_sync_mode_support = drop
    orig_process = getattr(driver, "process_message", None)
    if callable(orig_process):

        def process_message(message: object) -> None:
            if type(message).__name__ == "InBandWindowResize":
                return
            orig_process(message)

        driver.process_message = process_message


def reassert_cbreak(driver: object) -> None:
    """Resize and focus changes can restore echo; put the PTY back in cbreak."""
    fileno = getattr(driver, "fileno", None)
    if not isinstance(fileno, int):
        return
    try:
        import termios
        import tty
    except ImportError:
        return
    try:
        attrs = termios.tcgetattr(fileno)
    except Exception:
        return
    patch_l = getattr(driver, "_patch_lflag", None)
    patch_i = getattr(driver, "_patch_iflag", None)
    try:
        if callable(patch_l):
            attrs[tty.LFLAG] = patch_l(attrs[tty.LFLAG])
        else:
            attrs[tty.LFLAG] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN)
        if callable(patch_i):
            attrs[tty.IFLAG] = patch_i(attrs[tty.IFLAG])
        termios.tcsetattr(fileno, termios.TCSANOW, attrs)
    except Exception:
        return
