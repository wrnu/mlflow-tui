from __future__ import annotations

import os
import re
from collections.abc import Callable

# Optional probes. Incomplete terminals echo those CSI replies as text.
# Do not send 2026/2048 at all: hosts that do not implement them reply with
# `$y2026;2$y2048;2`, which lands in the shell after quit.
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_LEAVE_ALT_SCREEN = "\x1b[?1049l"
_PROBES_OFF = "\x1b[?1004l\x1b[?1016l" + _HIDE_CURSOR
_MOUSE_OFF = "\x1b[?1000l\x1b[?1003l\x1b[?1006l\x1b[?1015l\x1b[?1007l" + _PROBES_OFF
# Button + any-event + SGR mouse. 1003 is what most mobile webviews use for taps.
# Alternate scroll (1007) sends wheel/touch to the app instead of the host buffer.
_MOUSE_ON = "\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1006h\x1b[?1007h" + _PROBES_OFF
# Pixel mouse, sync, in-band resize, and Kitty keyboard: incomplete hosts echo these.
_SCRUB_EXACT = (
    "\x1b[?1015h",
    "\x1b[?1016h",
    "\x1b[?2026h",
    "\x1b[?2026l",
    "\x1b[?2026$p",
    "\x1b[?2048h",
    "\x1b[?2048l",
    "\x1b[?2048$p",
    "\x1b[?1004h",
)
_SCRUB_RE = re.compile(r"\x1b\[[>][0-9]+u")
# Textual parks the hardware cursor with a trailing CUP after every frame.
_TRAILING_CUP = re.compile(r"\x1b\[\d+;\d+H\Z")
_DROP_MESSAGES = frozenset({"InBandWindowResize", "TerminalSupportsSynchronizedOutput"})
_DRIVER_PROBE_METHODS = (
    "_request_terminal_sync_mode_support",
    "_query_in_band_window_resize",
    "_enable_in_band_window_resize",
    "_disable_in_band_window_resize",
    "_enable_mouse_pixels",
)


def _noop(self, *args, **kwargs) -> None:
    return


def silence_capability_probes() -> None:
    """Stop Textual drivers from querying sync / in-band resize (CSI $p)."""
    for module_name, class_name in (
        ("textual.drivers.linux_driver", "LinuxDriver"),
        ("textual.drivers.linux_inline_driver", "LinuxInlineDriver"),
        ("textual.drivers.web_driver", "WebDriver"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            driver_cls = getattr(module, class_name)
        except Exception:
            continue
        if getattr(driver_cls, "_mlflow_tui_probes_silenced", False):
            continue
        driver_cls._mlflow_tui_probes_silenced = True
        for method in _DRIVER_PROBE_METHODS:
            if hasattr(driver_cls, method):
                setattr(driver_cls, method, _noop)


def prepare_terminal() -> None:
    """Skip Kitty probes and extra scroll reports. Keep the host color depth."""
    os.environ["TEXTUAL_DISABLE_KITTY_KEY"] = "1"
    os.environ["TEXTUAL_SMOOTH_SCROLL"] = "0"
    silence_capability_probes()
    force_full_compositor_frames()


def scrub_driver_bytes(data: str) -> str:
    """Strip capability probes that incomplete terminals echo as glyphs."""
    if not data:
        return data
    for token in _SCRUB_EXACT:
        if token in data:
            data = data.replace(token, "")
    if "\x1b[>" in data:
        data = _SCRUB_RE.sub("", data)
    return data


def force_full_compositor_frames() -> None:
    """Always repaint the whole screen. Partial CUP updates drift and drop rows."""
    try:
        from textual._compositor import Compositor
    except Exception:
        return
    if getattr(Compositor, "_mlflow_tui_full_frames", False):
        return
    Compositor._mlflow_tui_full_frames = True
    orig = Compositor.render_update

    def render_update(self, full=False, screen_stack=None, simplify=False):
        return orig(self, True, screen_stack, simplify)

    Compositor.render_update = render_update


def finalize_driver_bytes(data: str, *, restore_cursor: bool = False) -> str:
    """Hide the hardware cursor and drop the post-frame CUP that parks it.

    Textual appends ``CSI row;col H`` after every compositor update so the
    terminal cursor sits on the focused widget. Hosts that drop ``ESC`` then
    paint ``34;40H`` as text, and if ``?25l`` did not stick the cursor blinks
    on the runs table. Newlines after a torn CUP scroll the alt screen, so
    rows vanish; compositor lines are already positioned with CUP.
    """
    if not data:
        return data
    data = scrub_driver_bytes(data)
    if restore_cursor or not data:
        return data
    if _SHOW_CURSOR in data:
        data = data.replace(_SHOW_CURSOR, "")
    if "\n" in data:
        data = data.replace("\n", "")
    data = _TRAILING_CUP.sub("", data)
    if data and not data.endswith(_HIDE_CURSOR):
        data += _HIDE_CURSOR
    return data


def write_tty(fd: int, data: str) -> None:
    """Write a frame in one go, looping until the kernel takes every byte."""
    if not data:
        return
    view = memoryview(data.encode("utf-8"))
    while view:
        n = os.write(fd, view)
        if n <= 0:
            break
        view = view[n:]


def install_atomic_tty_writer(driver: object) -> None:
    """Bypass line/block-buffered stdio so frames are not flushed mid-CUP."""
    thread = getattr(driver, "_writer_thread", None)
    if thread is None or getattr(thread, "_mlflow_tui_atomic", False):
        return
    file = getattr(thread, "_file", None)
    if file is None:
        return
    try:
        fd = file.fileno()
        flush = getattr(file, "flush", None)
        if callable(flush):
            flush()
    except Exception:
        return
    thread._mlflow_tui_atomic = True

    def write(text: str) -> int:
        if text:
            write_tty(fd, text)
        return len(text) if isinstance(text, str) else 0

    def flush() -> None:
        return

    try:
        file.write = write
        file.flush = flush
    except Exception:
        thread._mlflow_tui_atomic = False


def drain_pending_input(driver: object) -> None:
    """Drop unread CSI replies so they cannot type themselves into the shell."""
    fileno = getattr(driver, "fileno", None)
    if not isinstance(fileno, int):
        return
    try:
        import termios

        termios.tcflush(fileno, termios.TCIFLUSH)
    except Exception:
        return


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


def keep_probes_off(driver: object) -> None:
    """Re-assert cbreak without writing CSI (CSI mid-frame tears the header)."""
    if driver is None:
        return
    if hasattr(driver, "_in_band_window_resize"):
        driver._in_band_window_resize = False
    if hasattr(driver, "_sync_available"):
        driver._sync_available = False
    reassert_cbreak(driver)


def keep_terminal_quiet(driver: object, *, allow_mouse: bool) -> None:
    """Re-assert cbreak and mouse modes after resize or a host reset."""
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
    silence_capability_probes()
    force_full_compositor_frames()
    if driver is None or getattr(driver, "_mlflow_tui_quiet", False):
        install_atomic_tty_writer(driver)
        return
    driver._mlflow_tui_quiet = True
    if hasattr(driver, "_in_band_window_resize"):
        driver._in_band_window_resize = False
    orig_disable = getattr(driver, "_disable_mouse_support", None)
    orig_write = getattr(driver, "write", None)
    orig_stop = getattr(driver, "stop_application_mode", None)

    def _prepare(data: str) -> str:
        if _LEAVE_ALT_SCREEN in data:
            driver._mlflow_tui_leaving = True
        return finalize_driver_bytes(
            data, restore_cursor=bool(getattr(driver, "_mlflow_tui_leaving", False))
        )

    def write_raw(data: str) -> None:
        data = _prepare(data) if isinstance(data, str) else data
        if not data:
            return
        install_atomic_tty_writer(driver)
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
        if not isinstance(data, str):
            if callable(orig_write):
                orig_write(data)
            return
        data = _prepare(data)
        if not data:
            return
        install_atomic_tty_writer(driver)
        if callable(orig_write):
            orig_write(data)

    def drop() -> None:
        return

    def stop_application_mode() -> None:
        if callable(orig_stop):
            orig_stop()
        drain_pending_input(driver)

    driver.write = write
    driver._enable_mouse_support = enable_mouse
    driver._enable_mouse_pixels = drop
    driver._query_in_band_window_resize = drop
    driver._enable_in_band_window_resize = drop
    driver._disable_in_band_window_resize = drop
    driver._request_terminal_sync_mode_support = drop
    driver.stop_application_mode = stop_application_mode
    orig_process = getattr(driver, "process_message", None)
    if callable(orig_process):

        def process_message(message: object) -> None:
            if type(message).__name__ in _DROP_MESSAGES:
                return
            orig_process(message)

        driver.process_message = process_message
    install_atomic_tty_writer(driver)


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
