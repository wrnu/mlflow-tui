from __future__ import annotations

from rich.cells import cell_len
from textual.widgets import DataTable

from mlflow_tui.marquee import marquee_offset, marquee_slice


class MarqueeDataTable(DataTable[object]):
    """DataTable that scrolls overflowing cells on the highlighted row."""

    marquee_columns: set[str] = set()

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._source: dict[tuple[str, str], str] = {}
        self._active_row: str | None = None
        self._ticks = 0

    def on_mount(self) -> None:
        self.set_interval(0.14, self._tick_marquee)

    def clear(self, columns: bool = False):
        self._source = {}
        self._active_row = None
        self._ticks = 0
        return super().clear(columns=columns)

    def add_row(self, *cells: object, height: int | None = 1, key: str | None = None, label=None):
        result = super().add_row(*cells, height=height, key=key, label=label)
        if key is None:
            return result
        for column, cell in zip(self.ordered_columns, cells, strict=False):
            col_id = _column_id(column)
            if col_id in self.marquee_columns and isinstance(cell, str):
                self._source[(str(key), col_id)] = cell
        return result

    def _tick_marquee(self) -> None:
        if not self.row_count or not self.marquee_columns:
            return
        try:
            row = self.ordered_rows[self.cursor_row]
        except (IndexError, AttributeError):
            return
        row_key = _key_id(row.key)
        if row_key != self._active_row:
            self._restore_row(self._active_row)
            self._active_row = row_key
            self._ticks = 0
        self._ticks += 1
        for column in self.ordered_columns:
            col_id = _column_id(column)
            if col_id not in self.marquee_columns:
                continue
            full = self._source.get((row_key, col_id))
            if not full:
                continue
            width = max(1, column.width if not column.auto_width else column.content_width)
            if cell_len(full) <= width:
                continue
            offset = marquee_offset(self._ticks, cell_len(full), width)
            try:
                self.update_cell(row_key, col_id, marquee_slice(full, width, offset))
            except Exception:
                continue

    def _restore_row(self, row_key: str | None) -> None:
        if not row_key:
            return
        for column in self.ordered_columns:
            col_id = _column_id(column)
            full = self._source.get((row_key, col_id))
            if full is None:
                continue
            try:
                self.update_cell(row_key, col_id, full)
            except Exception:
                continue


def _column_id(column: object) -> str:
    return _key_id(getattr(column, "key", None))


def _key_id(key: object) -> str:
    value = getattr(key, "value", None)
    return str(value if value is not None else key)
