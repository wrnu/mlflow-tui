from __future__ import annotations

from collections.abc import Sequence

from rich.cells import cell_len, set_cell_size

SIDEBAR_MIN = 16
SIDEBAR_MAX = 22
SIDEBAR_CHROME = 3
MARQUEE_GAP = 3
MARQUEE_PAUSE = 6


def ellipsize(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    if width == 1:
        return "…"
    return set_cell_size(text, width - 1) + "…"


def marquee_slice(text: str, width: int, offset: int, *, gap: int = MARQUEE_GAP) -> str:
    """Return a `width`-cell window of *text*, looping with a gap when it overflows."""
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    loop = text + (" " * max(1, gap))
    start = offset % cell_len(loop)
    doubled = loop + loop
    skipped = 0
    index = 0
    while index < len(doubled) and skipped < start:
        skipped += cell_len(doubled[index])
        index += 1
    return set_cell_size(doubled[index:], width)


def marquee_offset(offset: int, text_width: int, width: int, *, pause: int = MARQUEE_PAUSE) -> int:
    """Hold at the start of the name, then scroll. Returns the slice offset."""
    if text_width <= width:
        return 0
    loop = text_width + MARQUEE_GAP
    position = offset % (pause + loop)
    if position < pause:
        return 0
    return position - pause


def sidebar_width(names: Sequence[str], *, title: str = "Experiments") -> int:
    """Sidebar columns: hug the labels, then cap so long names truncate."""
    longest = max([cell_len(title), *(cell_len(name) for name in names)], default=0)
    return max(SIDEBAR_MIN, min(SIDEBAR_MAX, longest + SIDEBAR_CHROME))
