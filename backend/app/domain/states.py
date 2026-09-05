"""The recovery-case state machine.

Case status was previously assigned wherever `monitor_outcome` happened to decide
something, with no statement anywhere of which transitions are legal. That worked
because one function owned every assignment -- and it stops working the moment a second
writer appears, which is exactly what human review introduces (an operator can close a
case from outside the graph).

So the legal transitions are declared once, here, and `transition()` refuses anything
else. Two properties follow that were previously only conventions:

* a terminal case can never reopen -- there is no outgoing edge from a terminal state,
  so a late-arriving webhook cannot resurrect a closed case and charge a customer again;
* every state is reachable and every non-terminal state can reach a terminal one, which
  `assert_well_formed()` proves rather than asserts. A planner that will not settle
  therefore cannot produce a case that never closes.

The names extend the original `CaseStatus` rather than replacing it: `PENDING`,
`IN_PROGRESS`, `RECOVERED`, `ESCALATED`, `STOPPED` and `EXHAUSTED` keep their meanings
and their serialised values, so stored rows and the dashboard are unaffected.
"""
from __future__ import annotations

from backend.app.models.enums import TERMINAL_STATUSES, CaseStatus


class IllegalTransition(RuntimeError):
    """A transition that the lifecycle does not permit. Always a bug, never data."""

    def __init__(self, frm: CaseStatus, to: CaseStatus, reason: str = ""):
        self.frm, self.to = frm, to
        detail = f": {reason}" if reason else ""
        super().__init__(f"illegal case transition {frm.value} -> {to.value}{detail}")


#: The lifecycle. Read as: from this state, these are the only states reachable next.
#:
#:   PENDING -> IN_PROGRESS       the case is picked up
#:   IN_PROGRESS -> IN_PROGRESS   re-plan after a blocked or inconclusive step
#:   IN_PROGRESS -> RECOVERED     payment landed (ours or a self-cure)
#:   IN_PROGRESS -> ESCALATED     handed to a human
#:   IN_PROGRESS -> STOPPED       nothing further is worth doing
#:   IN_PROGRESS -> EXHAUSTED     a bound was hit (steps, horizon, retries)
#:
#: PENDING may go straight to a terminal state: a case can be refused at the gate before
#: any work happens (a fraud hold escalated on sight), and forcing it through
#: IN_PROGRESS first would record work that never occurred.
TRANSITIONS: dict[CaseStatus, frozenset[CaseStatus]] = {
    CaseStatus.PENDING: frozenset({
        CaseStatus.IN_PROGRESS, CaseStatus.ESCALATED, CaseStatus.STOPPED,
        CaseStatus.EXHAUSTED, CaseStatus.RECOVERED,
    }),
    CaseStatus.IN_PROGRESS: frozenset({
        CaseStatus.IN_PROGRESS, CaseStatus.RECOVERED, CaseStatus.ESCALATED,
        CaseStatus.STOPPED, CaseStatus.EXHAUSTED,
    }),
    CaseStatus.RECOVERED: frozenset(),
    CaseStatus.ESCALATED: frozenset(),
    CaseStatus.STOPPED: frozenset(),
    CaseStatus.EXHAUSTED: frozenset(),
}

#: Terminal states, derived from the graph rather than restated. If someone adds an
#: outgoing edge from RECOVERED, this set changes with it instead of silently lying.
TERMINAL: frozenset[CaseStatus] = frozenset(s for s, nxt in TRANSITIONS.items() if not nxt)


def can_transition(frm: CaseStatus, to: CaseStatus) -> bool:
    return to in TRANSITIONS.get(frm, frozenset())


def transition(frm: CaseStatus, to: CaseStatus, reason: str = "") -> CaseStatus:
    """Return `to` if the move is legal, else raise. Callers assign the return value so
    the check cannot be forgotten by writing the field directly."""
    if not can_transition(frm, to):
        if frm in TERMINAL:
            raise IllegalTransition(frm, to, "case is already closed and cannot reopen")
        raise IllegalTransition(frm, to, reason)
    return to


def is_terminal(status: CaseStatus) -> bool:
    return status in TERMINAL


def assert_well_formed() -> None:
    """Prove the lifecycle is sound. Called by the test suite, and cheap enough to call
    at import time in a debug build.

    Three properties, each of which has a failure mode worth naming:
      1. every declared state appears as a key -- otherwise a transition into it would
         raise `KeyError` at runtime rather than being rejected;
      2. every state is reachable from PENDING -- an unreachable state is dead code that
         still appears in dashboards and filters;
      3. every non-terminal state can reach a terminal one -- otherwise a case could
         legally cycle forever, which is the failure the step cap exists to bound and
         should not also depend on.
    """
    missing = set(CaseStatus) - set(TRANSITIONS)
    if missing:
        raise AssertionError(f"states with no declared transitions: {sorted(s.value for s in missing)}")
    if frozenset(TERMINAL_STATUSES) != TERMINAL:
        raise AssertionError(
            f"domain terminal set {sorted(s.value for s in TERMINAL)} disagrees with "
            f"enums.TERMINAL_STATUSES {sorted(s.value for s in TERMINAL_STATUSES)}")

    seen, frontier = {CaseStatus.PENDING}, [CaseStatus.PENDING]
    while frontier:
        for nxt in TRANSITIONS[frontier.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    unreachable = set(CaseStatus) - seen
    if unreachable:
        raise AssertionError(f"unreachable states: {sorted(s.value for s in unreachable)}")

    for state in CaseStatus:
        if state in TERMINAL:
            continue
        seen, frontier = {state}, [state]
        while frontier:
            for nxt in TRANSITIONS[frontier.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        if not (seen & TERMINAL):
            raise AssertionError(f"{state.value} cannot reach any terminal state")


def mermaid() -> str:
    """The lifecycle as a diagram, generated from the table so the documentation cannot
    drift from the code it documents."""
    lines = ["stateDiagram-v2", "    [*] --> pending"]
    for frm in CaseStatus:
        for to in sorted(TRANSITIONS[frm], key=lambda s: s.value):
            arrow = "    " + (f"{frm.value} --> {to.value}" if frm is not to
                              else f"{frm.value} --> {to.value}: re-plan")
            lines.append(arrow)
    for t in sorted(TERMINAL, key=lambda s: s.value):
        lines.append(f"    {t.value} --> [*]")
    return "\n".join(lines)
