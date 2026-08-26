"""Phase 7 -- connection helper for both supported engines.

SQLite is the default and needs no setup. Set `RECOVERAI_DB_URL` to a postgresql:// URL
and the same stores run on Postgres instead. There is no silent fallback: if the URL is
set and the server is unreachable, connecting raises rather than quietly writing to a
local file, because a run that half-lands in the wrong engine is worse than one that
refuses to start.

The stores are written against one small surface -- execute / executemany /
executescript / commit / close, with `?` placeholders and dict-like rows. This module
translates that to whichever driver is live, so the SQL in the stores stays readable and
only genuinely dialect-specific constructs (autoincrement, upsert) are branched.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from backend.app.config import DB_PATH, DB_URL

SQLITE, POSTGRES = "sqlite", "postgres"


def engine_for(url: str | None = DB_URL) -> str:
    return POSTGRES if url else SQLITE


def qmark_to_pyformat(sql: str) -> str:
    """`?` -> `%s`, leaving anything inside single-quoted literals alone, and doubling a
    literal `%` so psycopg does not read it as a placeholder."""
    out, in_str = [], False
    for ch in sql:
        if ch == "'":
            in_str = not in_str
            out.append(ch)
        elif ch == "%" and not in_str:
            out.append("%%")
        elif ch == "?" and not in_str:
            out.append("%s")
        else:
            out.append(ch)
    return "".join(out)


class _PgCursor:
    """Iterable result that also answers fetchone(), matching sqlite3's cursor closely
    enough for the stores. Rows come back as plain dicts, so `row["col"]` and
    `dict(row)` both behave as they did on sqlite3.Row."""

    def __init__(self, cur):
        self._cur = cur
        self.lastrowid: int | None = None

    def __iter__(self):
        return iter(self._cur.fetchall())

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()


class Connection:
    """Thin wrapper presenting one API over sqlite3 and psycopg."""

    def __init__(self, dialect: str, raw: Any):
        self.dialect = dialect
        self.raw = raw

    # ------------------------------------------------------------------ helpers
    @property
    def is_pg(self) -> bool:
        return self.dialect == POSTGRES

    def _sql(self, sql: str) -> str:
        return qmark_to_pyformat(sql) if self.is_pg else sql

    # ------------------------------------------------------------------ execute
    def execute(self, sql: str, args: Sequence | None = None):
        if not self.is_pg:
            return self.raw.execute(sql, args or ())
        cur = self.raw.cursor()
        cur.execute(self._sql(sql), tuple(args or ()))
        return _PgCursor(cur)

    def execute_returning_id(self, sql: str, args: Sequence, id_column: str) -> int:
        """INSERT that yields the new primary key on either engine."""
        if not self.is_pg:
            return int(self.raw.execute(sql, args).lastrowid or 0)
        cur = self.raw.cursor()
        cur.execute(self._sql(f"{sql} RETURNING {id_column}"), tuple(args))
        row = cur.fetchone()
        return int(row[id_column]) if row else 0

    def executemany(self, sql: str, rows: Iterable[Sequence]) -> None:
        rows = list(rows)
        if not rows:
            return
        if not self.is_pg:
            self.raw.executemany(sql, rows)
            return
        cur = self.raw.cursor()
        cur.executemany(self._sql(sql), [tuple(r) for r in rows])

    def executescript(self, sql: str) -> None:
        if not self.is_pg:
            self.raw.executescript(sql)
            return
        cur = self.raw.cursor()
        for stmt in (s.strip() for s in sql.split(";")):
            if stmt:
                cur.execute(stmt)

    def commit(self) -> None:
        self.raw.commit()

    def close(self) -> None:
        self.raw.commit()
        self.raw.close()


def connect(path: Path | str = DB_PATH, url: str | None = DB_URL) -> Connection:
    """Postgres when `url` (or RECOVERAI_DB_URL) is set, SQLite otherwise.

    `path` is ignored on Postgres, so callers that pass a temp file -- the test suite
    does this constantly -- keep working unchanged on the default engine.
    """
    if url:
        import psycopg
        from psycopg.rows import dict_row
        raw = psycopg.connect(url, row_factory=dict_row, autocommit=False)
        return Connection(POSTGRES, raw)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(str(p), check_same_thread=False)
    raw.row_factory = sqlite3.Row
    raw.execute("PRAGMA journal_mode=WAL")
    raw.execute("PRAGMA synchronous=NORMAL")
    return Connection(SQLITE, raw)
