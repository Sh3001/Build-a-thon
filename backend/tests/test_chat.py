"""Chat layer -- the parser must not answer a different question than the one asked.

Every bug these tests pin down shipped at some point, and each produced a confident,
wrong, plausible-looking answer:

* "total recovered by category" filtered to already-recovered cases, which dropped the
  RISK_COMPLIANCE bucket entirely and reported per-bucket counts for the wrong population.
* "fraud" bound to nothing, so a question about 45 cases was answered over all 1,842.
* the aggregate field was resolved before the group-by field, so a total of
  `amount_recovered` silently became a total of `amount_usd`.
* per-bucket shares were divided by the sum of the rows under LIMIT, so a truncated
  top-15 reported each customer as ~8% of a run they were 0.4% of.
"""
from __future__ import annotations

import pytest

from backend.app.chat import answer as render
from backend.app.chat.dsl import Agg, DataQuery, Op, run as run_query
from backend.app.chat.parse import parse
from backend.app.database.db import connect
from backend.app.database.migrations import migrate

#: txn, customer, amount_usd, code, category, status, recovered, amount_recovered
ROWS = [
    ("txn_1", "cust_a", 100.0, "expired_card", "CUSTOMER_ACTION", "recovered", 1, 100.0),
    ("txn_2", "cust_a", 200.0, "expired_card", "CUSTOMER_ACTION", "escalated", 0, 0.0),
    ("txn_3", "cust_b", 300.0, "suspected_fraud", "RISK_COMPLIANCE", "escalated", 0, 0.0),
    ("txn_4", "cust_b", 400.0, "bank_timeout", "TEMPORARY", "recovered", 1, 400.0),
    ("txn_5", "cust_c", 500.0, "bank_timeout", "TEMPORARY", "stopped", 0, 0.0),
    ("txn_6", "cust_c", 6000.0, "insufficient_funds", "CUSTOMER_ACTION", "recovered", 1, 3000.0),
]


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    c = connect(tmp_path_factory.mktemp("chat") / "chat.db")
    migrate(c)
    for r in ROWS:
        c.execute(
            "INSERT INTO cases (transaction_id, strategy, customer_id, amount_usd, "
            "failure_code, failure_category, status, recovered, amount_recovered, "
            "expected_recovery) VALUES (?, 'recoverai', ?, ?, ?, ?, ?, ?, ?, ?)",
            [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[2] * 0.5])
    c.commit()
    return c


def ask(conn, question):
    q = parse(question)
    assert q.validate_against_schema() == [], f"{question!r} produced an invalid query"
    r = run_query(conn, q)
    return q, r, render.render(q, r) + render.caveat(q)


# ---------------------------------------------------------------- parsing correctness
def test_recovered_as_a_verb_does_not_become_a_status_filter():
    q = parse("Total recovered by failure category")
    assert [f.field for f in q.filters] == [], "the verb 'recovered' was read as a status"
    assert q.group_by == "failure_category"
    assert q.field == "amount_recovered", "the aggregate took a dimension as its measure"
    assert q.agg is Agg.SUM


def test_recovered_qualifying_a_noun_is_still_a_status_filter():
    for phrasing in ("how many recovered cases", "how many cases were recovered",
                     "count of successful cases"):
        q = parse(phrasing)
        assert any(f.field == "status" and f.value == "recovered" for f in q.filters), \
            f"{phrasing!r} lost its status filter"


def test_fraud_binds_to_a_failure_code():
    q = parse("Why did we lose money on fraud cases?")
    assert any(f.field == "failure_code" and f.value == "suspected_fraud"
               for f in q.filters), "'fraud' was ignored, widening the population silently"
    assert q.field == "amount_lost"


def test_expired_card_is_not_mistaken_for_an_unstored_payment_method():
    q = parse("how many cases failed with an expired card")
    assert q.unresolved == [], f"spurious warning: {q.unresolved}"
    assert any(f.value == "expired_card" for f in q.filters)


def test_a_term_the_table_cannot_express_is_always_reported():
    q = parse("how many cases used whatsapp")
    assert q.unresolved, "an unbindable term vanished instead of being surfaced"
    assert "channel" in q.unresolved[0]


def test_grouped_question_does_not_filter_to_the_bucket_it_groups_by():
    q = parse("which failure cause loses the most money")
    assert q.group_by == "failure_code"
    assert [f.field for f in q.filters] == [], "grouping by a filtered column returns 1 row"


def test_rate_questions_produce_a_rate_not_a_count():
    for phrasing in ("what is the recovery rate", "what percentage of cases were recovered"):
        assert parse(phrasing).agg is Agg.RATE, f"{phrasing!r} did not become a rate"


def test_how_many_beats_a_rate_word():
    assert parse("how many cases were recovered").agg is Agg.COUNT


# ---------------------------------------------------------------- answers match SQL
def test_totals_by_category_cover_every_bucket(conn):
    q, r, text = ask(conn, "Total recovered by failure category")
    got = {row["bucket"]: float(row["value"]) for row in r.rows}
    assert got == {"CUSTOMER_ACTION": 3100.0, "TEMPORARY": 400.0, "RISK_COMPLIANCE": 0.0}
    assert "risk compliance" in text, "the all-zero bucket was dropped from the answer"
    assert "$3,100.00" in text


def test_recovery_rate_matches_the_data(conn):
    q, r, text = ask(conn, "what is the recovery rate")
    assert r.scalar == pytest.approx(0.5)          # 3 of 6
    assert "50.0%" in text and "3 of 6" in text


def test_loss_by_cause_matches_the_data(conn):
    q, r, text = ask(conn, "which failure cause loses the most money")
    got = {row["bucket"]: float(row["value"]) for row in r.rows}
    assert got["insufficient_funds"] == 3000.0     # 6000 at risk, 3000 recovered
    assert got["suspected_fraud"] == 300.0
    assert got["expired_card"] == 200.0
    assert r.rows[0]["bucket"] == "insufficient_funds", "buckets are not ranked"


def test_filtered_count_reports_its_share_of_the_run(conn):
    q, r, text = ask(conn, "how many cases over 500 dollars")
    assert int(r.scalar) == 1 and r.grand_total == 6
    assert "16.7%" in text, "a bare count was returned with nothing to compare it against"


def test_group_shares_are_divided_by_all_buckets_not_the_visible_ones(conn):
    """The denominator must survive LIMIT."""
    q = DataQuery(agg=Agg.SUM, field="amount_usd", group_by="failure_code", limit=1)
    r = run_query(conn, q)
    assert len(r.rows) == 1
    assert r.scalar == pytest.approx(7500.0), "denominator only covered the visible bucket"
    assert "80.0%" in render.render(q, r)          # 6000 of 7500, not 100%


def test_average_is_an_average_and_not_a_sum(conn):
    q, r, text = ask(conn, "what is the average amount of a recovered case")
    assert r.scalar == pytest.approx((100 + 400 + 6000) / 3)
    assert "Average" in text


# ---------------------------------------------------------------- readability
@pytest.mark.parametrize("question", [
    "how many cases are there",
    "total recovered by failure category",
    "which failure cause loses the most money",
    "what is the recovery rate",
    "top 3 cases over 100 dollars",
    "how many cases used whatsapp",
])
def test_answers_never_leak_raw_column_names(conn, question):
    _, _, text = ask(conn, question)
    for raw in ("amount_usd", "amount_recovered", "customer_id", "failure_code",
                "failure_category", "amount_lost", "transaction_id"):
        assert raw not in text, f"{question!r} showed the reader a column name: {raw}"


def test_identifiers_keep_their_underscores(conn):
    _, _, text = ask(conn, "show me the top 2 cases")
    assert "`txn_" in text, "transaction ids were mangled into unsearchable prose"


def test_an_unanswerable_narrowing_is_flagged_in_the_answer_text(conn):
    _, _, text = ask(conn, "how many enterprise customers failed")
    assert "Heads up" in text and "customer segment" in text
