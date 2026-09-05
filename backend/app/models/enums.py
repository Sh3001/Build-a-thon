"""Phase 1 -- the shared vocabulary.

Every other module keys off these. The failure taxonomy is the important one: the
category a failure code belongs to determines what the policy engine will even
consider, so `FAILURE_CATEGORY` is the single source of truth for that mapping.
"""
from __future__ import annotations

from enum import Enum


class FailureCategory(str, Enum):
    TEMPORARY = "TEMPORARY"
    CUSTOMER_ACTION = "CUSTOMER_ACTION"
    PERSISTENT = "PERSISTENT"
    RISK_COMPLIANCE = "RISK_COMPLIANCE"


class FailureCode(str, Enum):
    # TEMPORARY -- the rail wobbled; the instrument is fine.
    BANK_TIMEOUT = "bank_timeout"
    NETWORK_ERROR = "network_error"
    PROCESSOR_UNAVAILABLE = "processor_unavailable"
    TEMPORARY_DECLINE = "temporary_decline"
    # CUSTOMER_ACTION -- needs the customer to do something (or to have money).
    INSUFFICIENT_FUNDS = "insufficient_funds"
    EXPIRED_CARD = "expired_card"
    INVALID_PAYMENT_METHOD = "invalid_payment_method"
    PAYMENT_LIMIT_EXCEEDED = "payment_limit_exceeded"
    # PERSISTENT -- retrying the same instrument cannot work.
    MULTIPLE_DECLINES = "multiple_declines"
    INVALID_ACCOUNT = "invalid_account"
    CLOSED_ACCOUNT = "closed_account"
    # RISK_COMPLIANCE -- never automated, always human.
    SUSPECTED_FRAUD = "suspected_fraud"
    HIGH_RISK_TRANSACTION = "high_risk_transaction"
    COMPLIANCE_HOLD = "compliance_hold"


FAILURE_CATEGORY: dict[FailureCode, FailureCategory] = {
    FailureCode.BANK_TIMEOUT: FailureCategory.TEMPORARY,
    FailureCode.NETWORK_ERROR: FailureCategory.TEMPORARY,
    FailureCode.PROCESSOR_UNAVAILABLE: FailureCategory.TEMPORARY,
    FailureCode.TEMPORARY_DECLINE: FailureCategory.TEMPORARY,
    FailureCode.INSUFFICIENT_FUNDS: FailureCategory.CUSTOMER_ACTION,
    FailureCode.EXPIRED_CARD: FailureCategory.CUSTOMER_ACTION,
    FailureCode.INVALID_PAYMENT_METHOD: FailureCategory.CUSTOMER_ACTION,
    FailureCode.PAYMENT_LIMIT_EXCEEDED: FailureCategory.CUSTOMER_ACTION,
    FailureCode.MULTIPLE_DECLINES: FailureCategory.PERSISTENT,
    FailureCode.INVALID_ACCOUNT: FailureCategory.PERSISTENT,
    FailureCode.CLOSED_ACCOUNT: FailureCategory.PERSISTENT,
    FailureCode.SUSPECTED_FRAUD: FailureCategory.RISK_COMPLIANCE,
    FailureCode.HIGH_RISK_TRANSACTION: FailureCategory.RISK_COMPLIANCE,
    FailureCode.COMPLIANCE_HOLD: FailureCategory.RISK_COMPLIANCE,
}


def category_of(code: FailureCode | str) -> FailureCategory:
    return FAILURE_CATEGORY[FailureCode(code)]


class PaymentMethod(str, Enum):
    CARD = "card"
    UPI = "upi"
    UPI_AUTOPAY = "upi_autopay"
    NETBANKING = "netbanking"
    ACH = "ach"
    WALLET = "wallet"


class CustomerSegment(str, Enum):
    CONSUMER = "consumer"
    SMB = "smb"
    ENTERPRISE = "enterprise"


class Channel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    IN_APP = "in_app"


class InterventionType(str, Enum):
    """The action allowlist. Nothing outside this enum can ever be executed."""
    RETRY_PAYMENT = "retry_payment"
    SEND_PAYMENT_LINK = "send_payment_link"
    SEND_REMINDER = "send_reminder"
    REQUEST_PAYMENT_METHOD_UPDATE = "request_payment_method_update"
    ESCALATE_CASE = "escalate_case"
    WAIT = "wait"
    STOP = "stop"


#: Actions that move money or contact a customer. `WAIT`/`STOP` are control flow.
EXECUTABLE_ACTIONS = frozenset({
    InterventionType.RETRY_PAYMENT,
    InterventionType.SEND_PAYMENT_LINK,
    InterventionType.SEND_REMINDER,
    InterventionType.REQUEST_PAYMENT_METHOD_UPDATE,
    InterventionType.ESCALATE_CASE,
})


#: Actions that actually reach the customer. Escalation is an internal handoff to a
#: human and must NOT consume the customer's contact budget.
CONTACT_ACTIONS = frozenset({
    InterventionType.SEND_REMINDER,
    InterventionType.SEND_PAYMENT_LINK,
    InterventionType.REQUEST_PAYMENT_METHOD_UPDATE,
})


class PolicyDecision(str, Enum):
    APPROVE = "approve"
    MODIFY = "modify"
    REJECT = "reject"
    #: Not refused, but not automatable either -- an authorised human must approve the
    #: action before it can run. Distinct from REJECT on purpose: rejecting a $50,000
    #: recovery silently drops the most valuable case in the queue, which is a business
    #: failure dressed up as a safety win. Nothing executes on this verdict, so the
    #: system still fails closed.
    HUMAN_REVIEW = "human_review"


#: Verdicts under which the executor may run something. HUMAN_REVIEW is deliberately
#: absent: it is a *pending* decision, and an unapproved action is an unsafe one.
EXECUTABLE_DECISIONS = frozenset({PolicyDecision.APPROVE, PolicyDecision.MODIFY})


class ReviewStatus(str, Enum):
    """Lifecycle of one human-review task."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Role(str, Enum):
    """Who may do what. Enforced in the backend; the frontend only hides buttons."""
    ADMIN = "admin"          # configure policy, manage keys
    OPERATOR = "operator"    # approve/reject human-review tasks, release DLQ entries
    ANALYST = "analyst"      # read experiments, metrics, queue
    AUDITOR = "auditor"      # read audit log and overrides, nothing else
    SYSTEM = "system"        # execute approved actions; no interactive powers


class CaseStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RECOVERED = "recovered"
    ESCALATED = "escalated"
    STOPPED = "stopped"
    EXHAUSTED = "exhausted"


TERMINAL_STATUSES = frozenset({
    CaseStatus.RECOVERED, CaseStatus.ESCALATED,
    CaseStatus.STOPPED, CaseStatus.EXHAUSTED,
})


class ActionOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    BLOCKED = "blocked"
    DELIVERED = "delivered"
    ESCALATED = "escalated"
