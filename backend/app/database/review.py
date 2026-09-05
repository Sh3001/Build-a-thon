"""Human review: the queue an operator works, and the record of what they decided.

Two tables, and the split is deliberate.

`human_review` is task state -- pending, approved, rejected, expired. It changes.

`decision_overrides` is evidence. When a person overrules the system, that fact is
written once and never updated: who, when, from what verdict to what verdict, and why a
reason string was required rather than optional. A human override is the one path by
which an action reaches the executor without a policy APPROVE, so it is exactly the path
an auditor will want to reconstruct, and "the operator later edited their justification"
must not be a possible sentence.

Nothing here executes anything. Approving a task records an approval; the executor still
requires a `PolicyResult` whose decision permits execution, which is why `approve()`
returns the approving `PolicyResult` rather than performing the action itself.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from backend.app.config import DB_PATH, DB_URL, REVIEW_SLA_HOURS
from backend.app.database.db import Connection, connect
from backend.app.database.store import DEFAULT_TENANT
from backend.app.models.enums import PolicyDecision, ReviewStatus, Role
from backend.app.models.schemas import PolicyResult, ProposedAction


def _now() -> datetime:
    return datetime.now(UTC)


def _migrate(conn) -> None:
    from backend.app.database.migrations import migrate
    migrate(conn)


class NotAuthorised(PermissionError):
    """The actor's role does not permit this operation."""


class ReviewClosed(RuntimeError):
    """The task has already been resolved. Re-deciding a closed task would let two
    operators disagree and the last writer win, silently."""


#: Only these roles may resolve a review task. ANALYST and AUDITOR are read-only by
#: design: the person who evaluates the system is not the person who authorises money to
#: move, and collapsing the two removes the separation the audit log exists to record.
CAN_RESOLVE: frozenset[Role] = frozenset({Role.OPERATOR, Role.ADMIN})


class ReviewStore:
    """Tenant-scoped, like every other store. There is no unscoped read."""

    def __init__(self, path=DB_PATH, conn: Connection | None = None,
                 url: str | None = DB_URL, tenant_id: str = DEFAULT_TENANT,
                 sla_hours: int = REVIEW_SLA_HOURS):
        self.conn = conn or connect(path, url)
        self.tenant_id = tenant_id
        self.sla_hours = sla_hours
        _migrate(self.conn)

    # ------------------------------------------------------------------ write
    def open_task(self, transaction_id: str, policy: PolicyResult, *,
                  customer_id: str = "", run_id: str | None = None,
                  amount_usd: float | None = None,
                  expected_profit: float | None = None,
                  model_version: str | None = None,
                  policy_version: str | None = None) -> str:
        """Record a HUMAN_REVIEW verdict as a task. Idempotent per (case, rule).

        A case that re-plans and hits the same rule again must not create a second
        identical task -- an operator queue that grows one row per graph iteration is
        unusable, and the duplicates all describe the same decision.
        """
        if policy.decision is not PolicyDecision.HUMAN_REVIEW:
            raise ValueError(
                f"only a HUMAN_REVIEW verdict opens a review task, got {policy.decision.value}")

        rule_id = policy.rules_fired[-1] if policy.rules_fired else "unknown"
        existing = self.conn.execute(
            "SELECT review_id FROM human_review WHERE tenant_id = ? AND transaction_id = ? "
            "AND rule_id = ? AND status = ?",
            (self.tenant_id, transaction_id, rule_id, ReviewStatus.PENDING.value)).fetchone()
        if existing:
            return str(existing["review_id"])

        review_id = f"rev_{uuid.uuid4().hex[:16]}"
        now = _now()
        action = policy.effective_action
        self.conn.execute(
            "INSERT INTO human_review (review_id, tenant_id, run_id, transaction_id, "
            "customer_id, status, rule_id, reason, proposed_action, amount_usd, "
            "expected_profit, model_version, policy_version, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (review_id, self.tenant_id, run_id, transaction_id, customer_id,
             ReviewStatus.PENDING.value, rule_id, policy.reason,
             json.dumps(action.model_dump(mode="json")) if action else None,
             amount_usd, expected_profit, model_version, policy_version,
             now.isoformat(), (now + timedelta(hours=self.sla_hours)).isoformat()))
        self.conn.commit()
        return review_id

    def open_escalation(self, transaction_id: str, action: ProposedAction, reason: str, *,
                        customer_id: str = "", run_id: str | None = None,
                        amount_usd: float | None = None,
                        rule_id: str = "ESCALATE_CASE") -> str:
        """Record an agent-initiated escalation as a review task.

        `ESCALATE_CASE` is the agent deciding a person should own the case -- fraud, a
        compliance hold, a value above the ceiling. That is the same queue a HUMAN_REVIEW
        verdict feeds, and routing it anywhere else would give operators two inboxes for
        one job while leaving escalated cases with no record that anyone is expected to
        act on them.
        """
        return self.open_task(
            transaction_id,
            PolicyResult(decision=PolicyDecision.HUMAN_REVIEW, effective_action=action,
                         rules_fired=[rule_id], reason=reason),
            customer_id=customer_id, run_id=run_id, amount_usd=amount_usd)

    def approve(self, review_id: str, actor: str, role: Role, reason: str) -> PolicyResult:
        """Authorise the withheld action. Returns the `PolicyResult` that permits it.

        The returned verdict is APPROVE, which is what the executor requires -- so the
        override reaches execution through the same single gate as everything else,
        rather than through a side door that skips it.
        """
        row = self._resolve(review_id, actor, role, reason, ReviewStatus.APPROVED)
        action = ProposedAction(**json.loads(row["proposed_action"])) \
            if row["proposed_action"] else None
        return PolicyResult(
            decision=PolicyDecision.APPROVE, effective_action=action,
            rules_fired=[row["rule_id"] or "", "R-HUMAN-OVERRIDE"],
            reason=f"approved by {actor} ({role.value}): {reason}")

    def reject(self, review_id: str, actor: str, role: Role, reason: str) -> PolicyResult:
        """Decline the withheld action. The case takes no automated step on this route."""
        row = self._resolve(review_id, actor, role, reason, ReviewStatus.REJECTED)
        return PolicyResult(
            decision=PolicyDecision.REJECT, effective_action=None,
            rules_fired=[row["rule_id"] or "", "R-HUMAN-OVERRIDE"],
            reason=f"rejected by {actor} ({role.value}): {reason}")

    def _resolve(self, review_id: str, actor: str, role: Role, reason: str,
                 status: ReviewStatus) -> dict:
        if role not in CAN_RESOLVE:
            raise NotAuthorised(
                f"role {role.value} may not resolve review tasks; "
                f"one of {sorted(r.value for r in CAN_RESOLVE)} is required")
        if not reason or not reason.strip():
            raise ValueError(
                "an override requires a reason: an unexplained override is indistinguishable "
                "from a mistake when someone reads the log a year from now")

        row = self.get(review_id)
        if row is None:
            raise KeyError(f"unknown review task {review_id}")
        if row["status"] != ReviewStatus.PENDING.value:
            raise ReviewClosed(
                f"review {review_id} is already {row['status']}; reopening it would let a "
                f"second operator overwrite the first without either seeing the conflict")

        now = _now()
        self.conn.execute(
            "UPDATE human_review SET status = ?, resolved_at = ?, resolved_by = ?, "
            "resolution_note = ? WHERE tenant_id = ? AND review_id = ?",
            (status.value, now.isoformat(), actor, reason, self.tenant_id, review_id))
        self.conn.execute(
            "INSERT INTO decision_overrides (override_id, review_id, tenant_id, "
            "transaction_id, actor, actor_role, at, original_decision, new_decision, "
            "action, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"ovr_{uuid.uuid4().hex[:16]}", review_id, self.tenant_id,
             row["transaction_id"], actor, role.value, now.isoformat(),
             PolicyDecision.HUMAN_REVIEW.value,
             PolicyDecision.APPROVE.value if status is ReviewStatus.APPROVED
             else PolicyDecision.REJECT.value,
             (json.loads(row["proposed_action"]) or {}).get("action")
             if row["proposed_action"] else None,
             reason))
        self.conn.commit()
        return row

    def expire_overdue(self, now: datetime | None = None) -> int:
        """Close tasks past their SLA. An unbounded queue is not a safety mechanism; it
        is a place decisions go to be forgotten, and a task nobody looked at within the
        window is a decision that was made by not making it."""
        cutoff = (now or _now()).isoformat()
        rows = list(self.conn.execute(
            "SELECT review_id FROM human_review WHERE tenant_id = ? AND status = ? "
            "AND expires_at IS NOT NULL AND expires_at < ?",
            (self.tenant_id, ReviewStatus.PENDING.value, cutoff)))
        for r in rows:
            self.conn.execute(
                "UPDATE human_review SET status = ?, resolved_at = ?, resolved_by = ?, "
                "resolution_note = ? WHERE tenant_id = ? AND review_id = ?",
                (ReviewStatus.EXPIRED.value, cutoff, "system",
                 f"expired unreviewed after {self.sla_hours}h", self.tenant_id,
                 r["review_id"]))
        self.conn.commit()
        return len(rows)

    # ------------------------------------------------------------------ read
    def get(self, review_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM human_review WHERE tenant_id = ? AND review_id = ?",
            (self.tenant_id, review_id)).fetchone()
        return dict(row) if row else None

    def pending(self, limit: int = 200, offset: int = 0) -> list[dict]:
        """Highest value first: an operator's minute is the scarce resource, so the queue
        is ordered by what that minute is worth rather than by arrival time."""
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM human_review WHERE tenant_id = ? AND status = ? "
            "ORDER BY amount_usd DESC NULLS LAST, created_at ASC LIMIT ? OFFSET ?"
            if self.conn.is_pg else
            "SELECT * FROM human_review WHERE tenant_id = ? AND status = ? "
            "ORDER BY amount_usd IS NULL, amount_usd DESC, created_at ASC LIMIT ? OFFSET ?",
            (self.tenant_id, ReviewStatus.PENDING.value, limit, offset))]

    def by_transaction(self, transaction_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM human_review WHERE tenant_id = ? AND transaction_id = ? "
            "ORDER BY created_at", (self.tenant_id, transaction_id))]

    def overrides(self, limit: int = 200, transaction_id: str | None = None) -> list[dict]:
        sql = "SELECT * FROM decision_overrides WHERE tenant_id = ?"
        args: list[Any] = [self.tenant_id]
        if transaction_id:
            sql += " AND transaction_id = ?"
            args.append(transaction_id)
        sql += " ORDER BY at DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def stats(self) -> dict:
        rows = [dict(r) for r in self.conn.execute(
            "SELECT status, COUNT(*) c, COALESCE(SUM(amount_usd), 0) v "
            "FROM human_review WHERE tenant_id = ? GROUP BY status", (self.tenant_id,))]
        by_status = {r["status"]: int(r["c"]) for r in rows}
        pending_value = next((float(r["v"]) for r in rows
                              if r["status"] == ReviewStatus.PENDING.value), 0.0)
        return {
            "by_status": by_status,
            "pending": by_status.get(ReviewStatus.PENDING.value, 0),
            "pending_value_usd": round(pending_value, 2),
            "sla_hours": self.sla_hours,
            "total": sum(by_status.values()),
        }

    def close(self) -> None:
        self.conn.close()


class ConsentStore:
    """Customer contact consent. Read by the `R-OPT-OUT` policy rule.

    `channel='*'` is a blanket opt-out and beats any per-channel record: someone who has
    said "stop contacting me" has not agreed to be contacted on a channel they did not
    name.
    """

    def __init__(self, path=DB_PATH, conn: Connection | None = None,
                 url: str | None = DB_URL, tenant_id: str = DEFAULT_TENANT):
        self.conn = conn or connect(path, url)
        self.tenant_id = tenant_id
        _migrate(self.conn)

    def opt_out(self, customer_id: str, channel: str = "*", reason: str = "",
                source: str = "customer") -> None:
        self.conn.execute(
            "DELETE FROM customer_consent WHERE tenant_id = ? AND customer_id = ? AND channel = ?",
            (self.tenant_id, customer_id, channel))
        self.conn.execute(
            "INSERT INTO customer_consent (tenant_id, customer_id, opted_out, channel, "
            "reason, source, updated_at) VALUES (?, ?, 1, ?, ?, ?, ?)",
            (self.tenant_id, customer_id, channel, reason, source, _now().isoformat()))
        self.conn.commit()

    def opt_in(self, customer_id: str, channel: str = "*") -> None:
        self.conn.execute(
            "DELETE FROM customer_consent WHERE tenant_id = ? AND customer_id = ? AND channel = ?",
            (self.tenant_id, customer_id, channel))
        self.conn.commit()

    def is_opted_out(self, customer_id: str, channel: str | None = None) -> bool:
        rows = list(self.conn.execute(
            "SELECT channel, opted_out FROM customer_consent WHERE tenant_id = ? AND customer_id = ?",
            (self.tenant_id, customer_id)))
        for r in rows:
            if not int(r["opted_out"]):
                continue
            if r["channel"] == "*" or (channel is not None and r["channel"] == channel):
                return True
        return False

    def opted_out_customers(self) -> set[str]:
        """Bulk read for a batch run -- one query instead of one per case."""
        return {str(r["customer_id"]) for r in self.conn.execute(
            "SELECT DISTINCT customer_id FROM customer_consent "
            "WHERE tenant_id = ? AND opted_out = 1", (self.tenant_id,))}

    def close(self) -> None:
        self.conn.close()
