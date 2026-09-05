"""Loading cases into the system.

Two sources, one shape. `load_split` reads the processed CSV splits every experiment and
test runs on. `SqlSource` reads live cases out of an external SQL database, for a
deployment that has real failed payments rather than a generated dataset.

Four properties hold on the database path, and they are why this is a module rather than a
`pd.read_sql` at each call site:

**The connection is read-only.** This points at somebody's payments database. The session
is set read-only and the statement is rejected unless it is a single SELECT, so a mistake
in a mapping file cannot become a write.

**Outcome columns never survive.** `recovered` and `recovery_days` are labels. A live run
that could see them would be scoring the answer it was handed, so they are dropped on the
way in -- and the drop is reported rather than silent, because a source that carries them
is a source someone should look at.

**Only schema fields survive.** A production payments table holds emails, names and card
fingerprints. The mapping is an allowlist of `Transaction` fields, so a column nobody
mapped cannot ride along into a prompt, a log, or a model feature.

**An unmappable failure code is quarantined, not guessed.** `processor_codes.normalise`
answers UNKNOWN when it does not recognise a code, and UNKNOWN has no `FailureCode`. Such
a row comes back as a reject carrying its reason, rather than being defaulted to
`temporary_decline` -- which would turn a mapping gap into confident retries against
instruments nobody has classified.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from pydantic import ValidationError

from backend.app.adapters.processor_codes import normalise
from backend.app.config import DATA_PROCESSED, to_usd
from backend.app.models.schemas import Transaction


# --------------------------------------------------------------------------- csv splits

def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / f"{name}.csv")
    if "amount_usd" not in df.columns:
        df["amount_usd"] = [to_usd(a, c) for a, c in zip(df["amount"], df["currency"])]
    return df


def to_transactions(df: pd.DataFrame) -> list[dict]:
    """Rows as dicts. The `recovered`/`recovery_days` labels are dropped: a live run must
    not be able to see the historical outcome of the case it is working."""
    drop = [c for c in OUTCOME_COLUMNS if c in df.columns]
    return df.drop(columns=drop).to_dict("records")


# --------------------------------------------------------------------------- sql source

#: Labels. Never fed to a run, on any path.
OUTCOME_COLUMNS = ("recovered", "recovery_days")

#: The allowlist. Anything not named here is dropped rather than carried.
TRANSACTION_FIELDS: frozenset[str] = frozenset(Transaction.model_fields)

#: Features the scorer reads but a real source often will not have. Falling back to the
#: schema default is reasonable; doing it silently is not, so `FetchReport` counts them.
DERIVED_FEATURES = (
    "failure_count", "days_since_failure", "customer_tenure", "previous_success_rate",
    "previous_payment_attempts", "avg_transaction_value", "days_since_last_payment",
    "subscription_age", "overdue_days", "previous_recovery_count",
)

_SELECT_ONLY = re.compile(r"^\s*(?:with\b.+?\bselect|select)\b", re.I | re.S)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy|call|do|"
    r"vacuum|merge|replace)\b", re.I)


class SqlSourceError(RuntimeError):
    pass


def _text(value):
    """Decode bytes once, at the edge.

    psycopg hands back `bytes` rather than `str` for text columns whenever the connection
    encoding is SQL_ASCII, because it cannot know what the bytes actually mean -- and
    SQL_ASCII is still common in older payment databases. Left alone, every identifier
    becomes `b'txn_1'`, every enum lookup misses, and the whole table lands in the reject
    pile for a reason that has nothing to do with the data.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            # latin-1 cannot fail, and mangling one accented name is better than discarding
            # a payment. The alternative -- rejecting the row -- hides a recoverable case.
            return raw.decode("latin-1")
    return value


@dataclass(frozen=True)
class FetchReport:
    """What the fetch did, in numbers a person can check.

    Kept separate from the rows so a caller cannot use the data without the count of what
    was thrown away -- a loader that quietly drops a third of a table is the failure mode
    this guards against.
    """
    fetched: int = 0
    accepted: int = 0
    rejected: int = 0
    dropped_columns: tuple[str, ...] = ()
    unmapped_columns: tuple[str, ...] = ()
    defaulted_features: dict[str, int] = field(default_factory=dict)
    reject_reasons: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        out = [f"fetched {self.fetched}, accepted {self.accepted}, rejected {self.rejected}"]
        if self.dropped_columns:
            out.append(f"  dropped outcome columns: {', '.join(self.dropped_columns)}")
        if self.unmapped_columns:
            out.append(f"  ignored unmapped columns: {', '.join(self.unmapped_columns)}")
        for feat, n in sorted(self.defaulted_features.items(), key=lambda kv: -kv[1]):
            out.append(f"  {feat}: {n} rows fell back to the schema default")
        for reason, n in sorted(self.reject_reasons.items(), key=lambda kv: -kv[1]):
            out.append(f"  rejected {n}: {reason}")
        return "\n".join(out)


@dataclass(frozen=True)
class FetchResult:
    transactions: list[Transaction]
    rejects: list[dict]
    report: FetchReport

    def to_dicts(self) -> list[dict]:
        return [t.model_dump(mode="json") for t in self.transactions]

    def to_frame(self) -> pd.DataFrame:
        """For the ML path, which works in DataFrames. `amount_usd` is computed by the
        schema, so it arrives already normalised across currencies."""
        return pd.DataFrame(self.to_dicts())


@dataclass
class SqlSource:
    """An external SQL database as a source of failed payments.

    `column_map` maps *source column* to `Transaction` field. Identity for anything not
    named, so a source that already uses the schema's names needs no mapping at all.
    """
    url: str
    query: str
    provider: str = "stripe"
    column_map: dict[str, str] = field(default_factory=dict)
    limit: int = 5000
    connect_fn: Callable[[str], object] | None = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> SqlSource:
        e = env if env is not None else dict(os.environ)
        url = e.get("RECOVERAI_SOURCE_URL", "").strip()
        query = e.get("RECOVERAI_SOURCE_QUERY", "").strip()
        if not url or not query:
            raise SqlSourceError(
                "RECOVERAI_SOURCE_URL and RECOVERAI_SOURCE_QUERY must both be set; "
                "the query must be a single SELECT over your failed-payment table")
        raw_map = e.get("RECOVERAI_SOURCE_COLUMNS", "").strip()
        try:
            column_map = json.loads(raw_map) if raw_map else {}
        except json.JSONDecodeError as exc:
            raise SqlSourceError(f"RECOVERAI_SOURCE_COLUMNS is not valid JSON: {exc}") from exc
        return cls(url=url, query=query,
                   provider=e.get("RECOVERAI_SOURCE_PROVIDER", "stripe").strip().lower(),
                   column_map=column_map,
                   limit=int(e.get("RECOVERAI_SOURCE_LIMIT", "5000")))

    # ------------------------------------------------------------------ safety
    @staticmethod
    def check_read_only(sql: str) -> None:
        """A single SELECT, and nothing else.

        Two checks rather than one: the statement must *look* like a select, and must not
        contain a write keyword anywhere -- which catches `SELECT ... ; DROP TABLE`, since
        a stacked statement passes the first test and fails the second.
        """
        stripped = re.sub(r"--[^\n]*", " ", sql)
        stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.S)
        if not _SELECT_ONLY.match(stripped):
            raise SqlSourceError("source query must be a SELECT (or a WITH ... SELECT)")
        if ";" in stripped.strip().rstrip(";"):
            raise SqlSourceError("source query must be a single statement")
        found = _FORBIDDEN.search(stripped)
        if found:
            raise SqlSourceError(
                f"source query contains {found.group(0)!r}; this connection is read-only")

    def _connect(self):
        conn = self._open()
        # Belt and braces with check_read_only: the server itself refuses a write on this
        # session even if a future edit lets one past the statement check. Applied to every
        # connection however it was obtained -- an alternative driver injected through
        # `connect_fn` must not quietly lose the guarantee the psycopg path has.
        try:
            conn.read_only = True
        except (AttributeError, TypeError) as exc:
            raise SqlSourceError(
                f"{type(conn).__name__} does not support a read-only session; refusing to "
                f"query a payments database over a connection that could write") from exc
        return conn

    def _open(self):
        if self.connect_fn is not None:
            return self.connect_fn(self.url)
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:            # pragma: no cover - dependency present
            raise SqlSourceError("psycopg is required to read from a SQL source") from exc
        return psycopg.connect(self.url, row_factory=dict_row, autocommit=True)

    # ------------------------------------------------------------------ fetch
    def fetch(self, limit: int | None = None) -> FetchResult:
        self.check_read_only(self.query)
        cap = self.limit if limit is None else limit
        conn = self._connect()
        try:
            cur = conn.execute(self.query)
            rows = [dict(r) for r in cur.fetchmany(cap)]
        finally:
            close = getattr(conn, "close", None)
            if callable(close):
                close()
        return self.build(rows)

    # ------------------------------------------------------------------ mapping
    def build(self, rows: Iterable[dict]) -> FetchResult:
        """Rows in, validated transactions out. Pure, so the mapping is testable without
        a database."""
        accepted: list[Transaction] = []
        rejects: list[dict] = []
        dropped: set[str] = set()
        unmapped: set[str] = set()
        defaulted: dict[str, int] = {}
        reasons: dict[str, int] = {}
        fetched = 0

        for raw_row in rows:
            fetched += 1
            row = {k: _text(v) for k, v in raw_row.items()}
            mapped: dict = {}
            for src_col, value in row.items():
                field_name = self.column_map.get(src_col, src_col)
                if field_name in OUTCOME_COLUMNS:
                    dropped.add(src_col)
                    continue
                if field_name not in TRANSACTION_FIELDS:
                    unmapped.add(src_col)
                    continue
                if value is not None:
                    mapped[field_name] = value

            for feat in DERIVED_FEATURES:
                if feat not in mapped:
                    defaulted[feat] = defaulted.get(feat, 0) + 1

            ref = str(row.get(self._source_of("transaction_id"), "") or "?")

            code = self._failure_code(row, mapped)
            if code is None:
                raw = mapped.get("failure_code") or row.get(self._source_of("failure_code"))
                reason = (f"failure code {raw!r} is not mapped for provider "
                          f"{self.provider!r}; refusing to guess a cause")
                reasons[reason] = reasons.get(reason, 0) + 1
                rejects.append({"transaction_id": ref, "reason": reason, "row": row})
                continue
            mapped["failure_code"] = code

            if "days_since_failure" not in mapped and mapped.get("failed_at"):
                derived = self._days_since(mapped["failed_at"])
                if derived is not None:
                    mapped["days_since_failure"] = derived
                    defaulted["days_since_failure"] -= 1

            try:
                accepted.append(Transaction(**mapped))
            except ValidationError as exc:
                first = exc.errors()[0]
                reason = f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"
                reasons[reason] = reasons.get(reason, 0) + 1
                rejects.append({"transaction_id": ref, "reason": reason, "row": row})

        report = FetchReport(
            fetched=fetched, accepted=len(accepted), rejected=len(rejects),
            dropped_columns=tuple(sorted(dropped)),
            unmapped_columns=tuple(sorted(unmapped)),
            defaulted_features={k: v for k, v in defaulted.items() if v > 0},
            reject_reasons=reasons)
        return FetchResult(accepted, rejects, report)

    # ------------------------------------------------------------------ helpers
    def _source_of(self, field_name: str) -> str:
        for src, dst in self.column_map.items():
            if dst == field_name:
                return src
        return field_name

    def _failure_code(self, row: dict, mapped: dict) -> str | None:
        """Provider code to internal vocabulary, via the shared taxonomy. A value that is
        already a valid `FailureCode` passes through -- a source that speaks this system's
        language should not have to round-trip through a provider table."""
        raw = mapped.get("failure_code")
        if raw is not None:
            candidate = str(raw).strip().lower()
            try:
                return Transaction.model_fields["failure_code"].annotation(candidate).value
            except (ValueError, TypeError):
                pass
        message = row.get(self._source_of("failure_message")) or row.get("failure_message")
        result = normalise(self.provider, None if raw is None else str(raw),
                           None if message is None else str(message))
        code = result.failure_code
        return code.value if code is not None else None

    @staticmethod
    def _days_since(value) -> float | None:
        try:
            when = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - when).total_seconds() / 86400.0, 3)


# --------------------------------------------------------------------------- cli

def _cli(argv: list[str] | None = None) -> int:
    """`python -m backend.app.services.dataio` -- fetch a sample and report on it.

    Read-only and prints no row content by default: the point is to see whether the
    mapping holds before pointing a run at the source, not to dump customer data to a
    terminal.
    """
    import argparse

    ap = argparse.ArgumentParser(prog="python -m backend.app.services.dataio",
                                 description=_cli.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--show-rejects", type=int, default=3,
                    help="print this many rejected transaction ids and reasons")
    ap.add_argument("--json", action="store_true", help="emit accepted rows as JSON")
    a = ap.parse_args(argv)

    src = SqlSource.from_env()
    print(f"source   {src.url.split('@')[-1]}   provider={src.provider}")
    result = src.fetch(limit=a.limit)
    print(result.report.describe())
    for r in result.rejects[: a.show_rejects]:
        print(f"  - {r['transaction_id']}: {r['reason']}")
    if a.json:
        print(json.dumps(result.to_dicts(), indent=2, default=str))
    return 0 if result.report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
