"""Offline tests for the external SQL source. No database -- `build()` is pure, and
`fetch()` runs against a fake connection, so the mapping, the read-only guard and the
quarantine behaviour are all covered without a server.

The assertions worth reading are the ones about what does *not* come through: labels,
unmapped columns, and rows whose failure code nobody has classified.
"""
from __future__ import annotations

import pytest

from backend.app.models.enums import FailureCode, PaymentMethod
from backend.app.services.dataio import (
    DERIVED_FEATURES, OUTCOME_COLUMNS, SqlSource, SqlSourceError, TRANSACTION_FIELDS,
)

URL = "postgresql://reader:pw@db.internal:5432/payments"
QUERY = "SELECT * FROM failed_payments WHERE failed_at > now() - interval '30 days'"


def row(**over) -> dict:
    base = {
        "customer_id": "cus_1", "transaction_id": "txn_1", "amount": 4200.0,
        "currency": "USD", "payment_method": "card", "failure_code": "insufficient_funds",
    }
    base.update(over)
    return base


def source(**kw) -> SqlSource:
    return SqlSource(url=URL, query=QUERY, **kw)


# ------------------------------------------------------------------ read-only guard

@pytest.mark.parametrize("sql", [
    "SELECT * FROM failed_payments",
    "  select id from t",
    "WITH recent AS (SELECT * FROM p) SELECT * FROM recent",
    "SELECT * FROM t;",                       # one trailing semicolon is still one statement
])
def test_a_plain_select_is_allowed(sql):
    SqlSource.check_read_only(sql)


@pytest.mark.parametrize("sql", [
    "INSERT INTO t VALUES (1)",
    "UPDATE payments SET amount = 0",
    "DELETE FROM payments",
    "DROP TABLE payments",
    "TRUNCATE payments",
    "GRANT ALL ON payments TO public",
    "COPY payments TO '/tmp/x'",
])
def test_write_statements_are_refused(sql):
    with pytest.raises(SqlSourceError):
        SqlSource.check_read_only(sql)


def test_a_write_stacked_behind_a_select_is_refused():
    """Passes the "looks like a select" test and must still be caught."""
    with pytest.raises(SqlSourceError, match="single statement|read-only"):
        SqlSource.check_read_only("SELECT 1; DROP TABLE payments")


def test_a_write_hidden_in_a_comment_boundary_is_refused():
    with pytest.raises(SqlSourceError):
        SqlSource.check_read_only("SELECT 1 /* harmless */ ; DELETE FROM payments")


def test_a_commented_out_write_does_not_trip_the_guard():
    SqlSource.check_read_only("SELECT id FROM t -- delete from t later\n")


def test_fetch_refuses_before_it_connects():
    """The guard runs first, so a bad query never opens a session."""
    def explode(url):
        raise AssertionError("must not connect")
    with pytest.raises(SqlSourceError):
        SqlSource(url=URL, query="DELETE FROM payments", connect_fn=explode).fetch()


# ------------------------------------------------------------------ what survives

def test_outcome_labels_are_dropped_and_reported():
    """A live run that could see `recovered` would be scoring the answer."""
    res = source().build([row(recovered=1, recovery_days=3)])
    assert res.report.accepted == 1
    assert set(res.report.dropped_columns) == set(OUTCOME_COLUMNS)
    assert not hasattr(res.transactions[0], "recovered")
    assert "recovered" not in res.to_dicts()[0]


def test_unmapped_columns_never_reach_a_transaction():
    """A payments table has PII in it. The mapping is an allowlist, so a column nobody
    mapped cannot ride along into a prompt, a log, or a feature."""
    res = source().build([row(customer_email="a@b.com", cardholder_name="A Person",
                              card_fingerprint="fp_xyz")])
    emitted = res.to_dicts()[0]
    for leaked in ("customer_email", "cardholder_name", "card_fingerprint"):
        assert leaked not in emitted
        assert leaked in res.report.unmapped_columns


def test_only_schema_fields_are_in_the_allowlist():
    assert "customer_id" in TRANSACTION_FIELDS and "amount" in TRANSACTION_FIELDS
    assert "recovered" not in TRANSACTION_FIELDS


def test_column_mapping_renames_source_columns():
    src = source(column_map={"cust_ref": "customer_id", "txn_ref": "transaction_id",
                             "gross_amount": "amount", "decline_reason": "failure_code"})
    res = src.build([{"cust_ref": "cus_9", "txn_ref": "txn_9", "gross_amount": 100.0,
                      "currency": "USD", "payment_method": "card",
                      "decline_reason": "insufficient_funds"}])
    assert res.report.accepted == 1
    t = res.transactions[0]
    assert t.customer_id == "cus_9" and t.transaction_id == "txn_9" and t.amount == 100.0


def test_a_mapped_outcome_column_is_still_dropped():
    src = source(column_map={"was_paid": "recovered"})
    res = src.build([row(was_paid=1)])
    assert res.report.dropped_columns == ("was_paid",)
    assert res.report.accepted == 1


# ------------------------------------------------------------------ failure codes

def test_an_internal_failure_code_passes_straight_through():
    res = source().build([row(failure_code="expired_card")])
    assert res.transactions[0].failure_code is FailureCode.EXPIRED_CARD


def test_a_provider_code_is_normalised_through_the_shared_taxonomy():
    res = source(provider="stripe").build([row(failure_code="issuer_not_available")])
    assert res.transactions[0].failure_code is FailureCode.NETWORK_ERROR


def test_an_unmappable_code_is_quarantined_not_guessed():
    """The failure mode this exists to prevent: defaulting the unrecognised remainder to
    `temporary_decline` and retrying instruments nobody has classified."""
    res = source(provider="stripe").build([row(failure_code="wibble_declined")])
    assert res.report.accepted == 0 and res.report.rejected == 1
    reject = res.rejects[0]
    assert reject["transaction_id"] == "txn_1"
    assert "refusing to guess" in reject["reason"]
    assert "wibble_declined" in reject["reason"]


def test_an_unknown_provider_quarantines_everything_rather_than_guessing():
    res = source(provider="not_a_processor").build([row(failure_code="some_code")])
    assert res.report.rejected == 1


def test_a_free_text_message_can_rescue_a_row_the_code_could_not():
    res = source(provider="stripe").build(
        [row(failure_code="odd_code", failure_message="Not enough balance in account")])
    assert res.report.accepted == 1
    assert res.transactions[0].failure_code is FailureCode.INSUFFICIENT_FUNDS


def test_rejects_are_counted_by_reason():
    res = source(provider="stripe").build([
        row(transaction_id="a", failure_code="nope_1"),
        row(transaction_id="b", failure_code="nope_2"),
    ])
    assert res.report.rejected == 2 and sum(res.report.reject_reasons.values()) == 2


# ------------------------------------------------------------------ validation

def test_a_row_missing_a_required_field_is_rejected_with_the_field_named():
    bad = row()
    del bad["customer_id"]
    res = source().build([bad])
    assert res.report.rejected == 1
    assert "customer_id" in res.rejects[0]["reason"]


def test_an_unknown_payment_method_is_rejected_not_coerced():
    res = source().build([row(payment_method="cheque")])
    assert res.report.rejected == 1


def test_one_bad_row_does_not_lose_the_good_ones():
    res = source().build([row(transaction_id="ok_1"),
                          row(transaction_id="bad", payment_method="cheque"),
                          row(transaction_id="ok_2")])
    assert res.report.accepted == 2 and res.report.rejected == 1
    assert {t.transaction_id for t in res.transactions} == {"ok_1", "ok_2"}


# ------------------------------------------------------------------ derived features

def test_features_the_source_lacks_are_counted_not_hidden():
    """Falling back to a schema default is reasonable; doing it invisibly degrades scoring
    with no trace, so the count is part of the report."""
    res = source().build([row()])
    assert res.report.accepted == 1
    for feat in DERIVED_FEATURES:
        if feat != "days_since_failure":
            assert res.report.defaulted_features[feat] == 1


def test_days_since_failure_is_derived_from_a_timestamp_when_absent():
    from datetime import datetime, timedelta, timezone
    when = datetime.now(timezone.utc) - timedelta(days=5)
    res = source().build([row(failed_at=when)])
    assert res.transactions[0].days_since_failure == pytest.approx(5.0, abs=0.01)
    assert "days_since_failure" not in res.report.defaulted_features


def test_a_supplied_value_is_not_overwritten_by_derivation():
    res = source().build([row(days_since_failure=2.0, failed_at="2020-01-01T00:00:00")])
    assert res.transactions[0].days_since_failure == 2.0


def test_amount_usd_is_computed_across_currencies():
    res = source().build([row(amount=1000.0, currency="INR")])
    t = res.transactions[0]
    assert t.amount_usd != t.amount and t.amount_usd > 0


# ------------------------------------------------------------------ fetch plumbing

class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.asked = None

    def fetchmany(self, n):
        self.asked = n
        return self.rows[:n]


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.read_only = None
        self.closed = False
        self.seen = None

    def execute(self, sql):
        self.seen = sql
        return FakeCursor(self.rows)

    def close(self):
        self.closed = True


def test_fetch_sets_the_session_read_only_and_closes_it():
    conn = FakeConn([row()])
    src = SqlSource(url=URL, query=QUERY, connect_fn=lambda u: conn)
    res = src.fetch()
    assert conn.read_only is True
    assert conn.closed is True
    assert conn.seen == QUERY
    assert res.report.accepted == 1


def test_fetch_never_pulls_an_unbounded_table():
    conn = FakeConn([row(transaction_id=f"t{i}") for i in range(500)])
    cursor_limit = {}

    class Recording(FakeConn):
        def execute(self, sql):
            cur = FakeCursor(self.rows)
            cursor_limit["cur"] = cur
            return cur

    conn = Recording([row(transaction_id=f"t{i}") for i in range(500)])
    SqlSource(url=URL, query=QUERY, limit=25, connect_fn=lambda u: conn).fetch()
    assert cursor_limit["cur"].asked == 25


def test_an_explicit_limit_overrides_the_configured_one():
    conn = FakeConn([row(transaction_id=f"t{i}") for i in range(10)])
    res = SqlSource(url=URL, query=QUERY, limit=1000,
                    connect_fn=lambda u: conn).fetch(limit=3)
    assert res.report.fetched == 3


def test_result_converts_to_the_frame_the_ml_path_expects():
    conn = FakeConn([row(transaction_id="t1"), row(transaction_id="t2")])
    df = SqlSource(url=URL, query=QUERY, connect_fn=lambda u: conn).fetch().to_frame()
    assert list(df["transaction_id"]) == ["t1", "t2"]
    assert "amount_usd" in df.columns
    assert not set(OUTCOME_COLUMNS) & set(df.columns)


# ------------------------------------------------------------------ configuration

def test_from_env_reads_a_full_configuration():
    src = SqlSource.from_env({
        "RECOVERAI_SOURCE_URL": URL, "RECOVERAI_SOURCE_QUERY": QUERY,
        "RECOVERAI_SOURCE_PROVIDER": "Razorpay", "RECOVERAI_SOURCE_LIMIT": "42",
        "RECOVERAI_SOURCE_COLUMNS": '{"cust_ref": "customer_id"}'})
    assert src.provider == "razorpay" and src.limit == 42
    assert src.column_map == {"cust_ref": "customer_id"}


@pytest.mark.parametrize("env", [
    {}, {"RECOVERAI_SOURCE_URL": URL}, {"RECOVERAI_SOURCE_QUERY": QUERY}])
def test_from_env_refuses_a_half_configuration(env):
    with pytest.raises(SqlSourceError):
        SqlSource.from_env(env)


def test_a_malformed_column_map_is_a_startup_error_not_a_silent_identity_map():
    with pytest.raises(SqlSourceError, match="valid JSON"):
        SqlSource.from_env({"RECOVERAI_SOURCE_URL": URL, "RECOVERAI_SOURCE_QUERY": QUERY,
                            "RECOVERAI_SOURCE_COLUMNS": "{not json}"})


def test_the_report_reads_as_prose():
    res = source(provider="stripe").build([row(recovered=1, customer_email="a@b.com"),
                                           row(transaction_id="x", failure_code="nope")])
    text = res.report.describe()
    assert "fetched 2" in text and "accepted 1" in text and "rejected 1" in text
    assert "recovered" in text and "customer_email" in text


def test_bytes_columns_are_decoded_rather_than_rejected():
    """psycopg returns bytes for text columns on a SQL_ASCII connection, which older
    payment databases still use. Left alone every identifier becomes b'txn_1', every enum
    lookup misses, and the whole table lands in the reject pile for a reason that has
    nothing to do with the data. Found against a real Postgres, not a fake one."""
    res = source().build([{
        "customer_id": b"cus_1", "transaction_id": b"txn_1", "amount": 100.0,
        "currency": b"USD", "payment_method": b"card", "failure_code": b"insufficient_funds",
    }])
    assert res.report.accepted == 1
    t = res.transactions[0]
    assert t.transaction_id == "txn_1"
    assert t.payment_method is PaymentMethod.CARD
    assert t.failure_code is FailureCode.INSUFFICIENT_FUNDS


def test_undecodable_bytes_do_not_lose_the_payment():
    """latin-1 cannot fail. Mangling one accented name beats discarding a recoverable case."""
    res = source().build([dict(row(), customer_id=b"cus_\xff\xfe")])
    assert res.report.accepted == 1
    assert res.transactions[0].customer_id.startswith("cus_")


def test_a_connection_that_cannot_be_made_read_only_is_refused():
    """An alternative driver injected through connect_fn must not quietly lose the
    read-only guarantee that the psycopg path has."""
    class Immutable:
        __slots__ = ()

    with pytest.raises(SqlSourceError, match="read-only"):
        SqlSource(url=URL, query=QUERY, connect_fn=lambda u: Immutable()).fetch()
