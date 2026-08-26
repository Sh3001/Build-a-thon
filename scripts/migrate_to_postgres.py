#!/usr/bin/env python3
"""Copy a SQLite RecoverAI database into Postgres.

    .venv/bin/python scripts/migrate_to_postgres.py \
        --url postgresql://localhost/recoverai

Row order matters for `audit_log`: the chain is verified in `seq` order, so rows are
read ordered and inserted with their original `seq` preserved. The sequence is then
reset past the high-water mark, otherwise the next append would collide with a key that
already exists. The script finishes by re-verifying the chain on the *destination*, so a
migration that silently corrupted an ordering fails loudly here rather than on the
dashboard.

Re-runnable: --replace truncates the destination tables first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.audit.store import FIELDS as AUDIT_FIELDS  # noqa: E402
from backend.app.audit.store import AuditStore  # noqa: E402
from backend.app.audit.store import schema_for as audit_schema  # noqa: E402
from backend.app.config import DB_PATH  # noqa: E402
from backend.app.database.db import POSTGRES, connect  # noqa: E402
from backend.app.database.store import CASE_NAMES  # noqa: E402
from backend.app.database.store import schema_for as case_schema  # noqa: E402

AUDIT_COLUMNS = ["seq", "prev_hash", "row_hash", *AUDIT_FIELDS]
BATCH = 2000


def copy_table(src, dst, table: str, columns: list[str], order: str | None = None) -> int:
    cols = ", ".join(columns)
    sql = f"SELECT {cols} FROM {table}" + (f" ORDER BY {order}" if order else "")
    rows = [tuple(r[c] for c in columns) for r in src.execute(sql)]
    if not rows:
        return 0
    marks = ", ".join("?" * len(columns))
    insert = f"INSERT INTO {table} ({cols}) VALUES ({marks})"
    for i in range(0, len(rows), BATCH):
        dst.executemany(insert, rows[i:i + BATCH])
    dst.commit()
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", type=Path, default=DB_PATH, help="source .db file")
    ap.add_argument("--url", required=True, help="destination postgresql:// URL")
    ap.add_argument("--replace", action="store_true",
                    help="truncate destination tables before copying")
    args = ap.parse_args()

    if not args.sqlite.exists():
        print(f"error: {args.sqlite} not found", file=sys.stderr)
        return 2

    src = connect(args.sqlite, url=None)
    dst = connect(args.sqlite, url=args.url)
    if dst.dialect != POSTGRES:
        print("error: --url did not produce a Postgres connection", file=sys.stderr)
        return 2

    print(f"  source      {args.sqlite}")
    print(f"  destination {args.url}\n")

    dst.executescript(case_schema(dst.dialect))
    dst.executescript(audit_schema(dst.dialect))
    dst.commit()

    if args.replace:
        for t in ("audit_log", "cases", "runs"):
            dst.execute(f"TRUNCATE TABLE {t} RESTART IDENTITY")
        dst.commit()
        print("  truncated destination tables\n")

    total = 0
    for table, cols, order in (
        ("cases", CASE_NAMES, None),
        ("audit_log", AUDIT_COLUMNS, "seq"),
        ("runs", ["key", "created_at", "payload"], None),
    ):
        n = copy_table(src, dst, table, cols, order)
        total += n
        print(f"  {table:<10} {n:>7,} rows  ->  ok")

    # seq values were carried over verbatim, so the identity sequence still points at 1.
    dst.execute("SELECT setval(pg_get_serial_sequence('audit_log','seq'), "
                "COALESCE((SELECT MAX(seq) FROM audit_log), 1))")
    dst.commit()

    ok, bad = AuditStore(conn=dst).verify()
    print(f"\n  chain verified on destination: {ok}"
          + ("" if ok else f"  (first bad seq {bad})"))
    src.close()
    dst.close()
    if not ok:
        print("\n  MIGRATION FAILED -- destination chain does not verify", file=sys.stderr)
        return 1
    print(f"\n  {total:,} rows migrated. Point the app at it with:\n"
          f"    export RECOVERAI_DB_URL={args.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
