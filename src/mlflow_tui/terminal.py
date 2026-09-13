from __future__ import annotations

import os
from collections.abc import Callable

# Optional probes. Incomplete terminals echo those CSI replies as text.
_PROBES_OFF = "\x1b[?1004l\x1b[?1016l\x1b[?2048l\x1b[?25l"
_MOUSE_OFF = "\x1b[?1000l\x1b[?1003l\x1b[?1006l\x1b[?1015l\x1b[?1007l" + _PROBES_OFF
# Button + any-event + SGR mouse. 1003 is what most mobile webviews use for taps.
# Alternate scroll (1007) sends wheel/touch to the app instead of the host buffer.
_MOUSE_ON = "\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1006h\x1b[?1007h" + _PROBES_OFF
_PIXEL_MOUSE_ON = ("?1015h", "?1016h")


def prepare_terminal() -> None:
    """Skip Kitty probes and truecolor. Those CSI replies show up as header junk."""
    os.environ.setdefault("TEXTUAL_DISABLE_KITTY_KEY", "1")
    os.environ.setdefault("TEXTUAL_COLOR_SYSTEM", "256")
    if os.environ.get("TEXTUAL_COLOR_SYSTEM") == "256":
        colorterm = os.environ.get("COLORTERM", "").strip().lower()
        if colorterm in {"truecolor", "24bit"}:
            os.environ["COLORTERM"] = "256"


def should_enable_mouse(explicit: bool | None = None) -> bool:
    """Clicks, taps, and wheel are on unless the user turns them off.

    Do not require a tty: some web/mobile PTYs report ``isatty() == False``.
    """
    if explicit is not None:
        return explicit
    env = os.environ.get("MLFLOW_TUI_MOUSE")
    if env is not None and env.strip() != "":
        return env.strip().lower() not in {"0", "false", "no", "off"}
    term = os.environ.get("TERM", "")
    if term in {"", "dumb"}:
        return False
    return True


def keep_terminal_quiet(driver: object, *, allow_mouse: bool) -> None:
    """Re-assert cbreak and 256-color-safe modes after resize or a host reset."""
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
        write(_MOUSE_ON if allow_mouse else _MOUSE_OFF)
        flush = getattr(driver, "flush", None)
        if callable(flush):
            flush()
    reassert_cbreak(driver)


def install_quiet_driver(
    driver: object,
    *,
    allow_mouse: Callable[[], bool],
) -> None:
    """Allow taps, clicks, and wheel; skip pixel-mouse and capability probes."""
    if driver is None or getattr(driver, "_mlflow_tui_quiet", False):
        return
    driver._mlflow_tui_quiet = True
    if hasattr(driver, "_in_band_window_resize"):
        driver._in_band_window_resize = False
    orig_disable = getattr(driver, "_disable_mouse_support", None)
    orig_write = getattr(driver, "write", None)

    def write_raw(data: str) -> None:
        if callable(orig_write):
            orig_write(data)
            flush = getattr(driver, "flush", None)
            if callable(flush):
                flush()

    def enable_mouse() -> None:
        if allow_mouse():
            if hasattr(driver, "_mouse"):
                driver._mouse = True
            write_raw(_MOUSE_ON)
            return
        if callable(orig_disable):
            orig_disable()
        if hasattr(driver, "_mouse"):
            driver._mouse = False
        write_raw(_MOUSE_OFF)

    def write(data: str) -> None:
        if isinstance(data, str) and any(marker in data for marker in _PIXEL_MOUSE_ON):
            return
        if callable(orig_write):
            orig_write(data)

    def drop() -> None:
        return

    driver.write = write
    driver._enable_mouse_support = enable_mouse
    driver._enable_mouse_pixels = drop
    driver._query_in_band_window_resize = drop
    driver._enable_in_band_window_resize = drop
    driver._request_terminal_sync_mode_support = drop
    orig_process = getattr(driver, "process_message", None)
    if callable(orig_process):

        def process_message(message: object) -> None:
            name = type(message).__name__
            if name == "InBandWindowResize":
                return
            orig_process(message)

        driver.process_message = process_message


def reassert_cbreak(driver: object) -> None:
    """Resize can restore echo; put the PTY back in cbreak."""
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
