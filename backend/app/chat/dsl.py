"""A small, closed query language over the `cases` table - and the code that runs it.

This replaces a fixed menu of canned questions. The earlier version matched a question to
one of nine shapes and dropped anything it could not express, so "how many cases used
whatsapp" silently became "how many cases", and "average amount" silently became a sum.
Answering a *different* question confidently is worse than refusing the original one.

Three properties, in order of importance:

1. **Nothing is dropped silently.** Every term the parser cannot bind becomes an entry in
   `unresolved`, and the answer says so. An answer whose filters do not match the question
   is not returned as if it did.
2. **Numbers are computed, never generated.** A model may emit a `DataQuery`; it never
   emits a figure. The SQL here produces every number.
3. **The surface is closed.** Fields, operators and aggregates are allowlisted against the
   real schema, and values are bound as parameters - a query naming an unknown column is
   rejected, not interpolated.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from backend.app.database.store import CASE_COLUMNS

#: Columns that may be filtered, grouped or aggregated, typed from the real schema.
NUMERIC = {c for c, t in CASE_COLUMNS if t.startswith(("REAL", "INTEGER"))}
TEXTUAL = {c for c, t in CASE_COLUMNS if t.startswith("TEXT")}
#: `strategy` is the arm selector, not a user-facing filter - it is set by the engine.
FIELDS = (NUMERIC | TEXTUAL) - {"strategy", "actions"}


class Agg(str, Enum):
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    LIST = "list"          #: return rows rather than a scalar


class Op(str, Enum):
    EQ = "="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    CONTAINS = "contains"


class Filter(BaseModel):
    field: str
    op: Op = Op.EQ
    value: float | str


class DataQuery(BaseModel):
    """One question, expressed against the case table."""

    agg: Agg = Agg.COUNT
    field: str | None = None          #: what to aggregate; unused for COUNT/LIST
    filters: list[Filter] = Field(default_factory=list)
    group_by: str | None = None
    order_by: str | None = None
    descending: bool = True
    limit: int = Field(default=10, ge=1, le=50)
    #: Terms in the question that could not be bound to a field. Never silently empty.
    unresolved: list[str] = Field(default_factory=list)

    def validate_against_schema(self) -> list[str]:
        """Returns human-readable problems. Empty means the query is runnable."""
        bad: list[str] = []
        if self.field and self.field not in FIELDS:
            bad.append(f"unknown field {self.field!r}")
        if self.agg in (Agg.SUM, Agg.AVG, Agg.MIN, Agg.MAX):
            if not self.field:
                bad.append(f"{self.agg.value} needs a field")
            elif self.field not in NUMERIC:
                bad.append(f"{self.field!r} is not numeric, so it cannot be "
                           f"{self.agg.value}med" if self.agg is Agg.SUM
                           else f"{self.field!r} is not numeric")
        if self.group_by and self.group_by not in FIELDS:
            bad.append(f"cannot group by {self.group_by!r}")
        if self.order_by and self.order_by not in FIELDS:
            bad.append(f"cannot order by {self.order_by!r}")
        for f in self.filters:
            if f.field not in FIELDS:
                bad.append(f"cannot filter on {f.field!r}")
            elif f.op in (Op.GT, Op.GTE, Op.LT, Op.LTE) and f.field not in NUMERIC:
                bad.append(f"{f.field!r} is text, so {f.op.value} does not apply")
        return bad


class Result(BaseModel):
    rows: list[dict] = Field(default_factory=list)
    scalar: float | None = None
    total_matched: int = 0
    sql: str = ""                    #: shown to the user; this is the audit trail


def _where(filters: list[Filter]) -> tuple[str, list]:
    """Build a parameterised WHERE. Field names come from the allowlist, values are bound."""
    clauses, args = [], []
    for f in filters:
        if f.op is Op.CONTAINS:
            clauses.append(f"LOWER({f.field}) LIKE ?")
            args.append(f"%{str(f.value).lower()}%")
        else:
            clauses.append(f"{f.field} {f.op.value} ?")
            args.append(f.value)
    return (" AND ".join(clauses), args)


def run(conn, q: DataQuery, strategy: str = "recoverai") -> Result:
    """Execute a validated query. Callers must have checked `validate_against_schema`."""
    where, args = _where(q.filters)
    scope = "strategy = ?" + (f" AND {where}" if where else "")
    args = [strategy, *args]

    matched = int(conn.execute(
        f"SELECT COUNT(*) c FROM cases WHERE {scope}", args).fetchone()["c"])

    if q.agg is Agg.LIST:
        order = q.order_by or "expected_recovery"
        sql = (f"SELECT transaction_id, customer_id, failure_code, status, amount_usd, "
               f"amount_recovered, retries, contacts FROM cases WHERE {scope} "
               f"ORDER BY {order} {'DESC' if q.descending else 'ASC'} LIMIT ?")
        rows = [dict(r) for r in conn.execute(sql, [*args, q.limit])]
        return Result(rows=rows, total_matched=matched, sql=sql)

    fn = {Agg.COUNT: "COUNT(*)", Agg.SUM: f"SUM({q.field})", Agg.AVG: f"AVG({q.field})",
          Agg.MIN: f"MIN({q.field})", Agg.MAX: f"MAX({q.field})"}[q.agg]

    if q.group_by:
        sql = (f"SELECT {q.group_by} AS bucket, {fn} AS value, COUNT(*) AS n "
               f"FROM cases WHERE {scope} GROUP BY {q.group_by} "
               f"ORDER BY value {'DESC' if q.descending else 'ASC'} LIMIT ?")
        rows = [dict(r) for r in conn.execute(sql, [*args, q.limit])]
        return Result(rows=rows, total_matched=matched, sql=sql)

    sql = f"SELECT {fn} AS value FROM cases WHERE {scope}"
    row = conn.execute(sql, args).fetchone()
    val = row["value"] if row else None
    return Result(scalar=None if val is None else float(val),
                  total_matched=matched, sql=sql)
