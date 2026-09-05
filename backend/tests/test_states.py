"""The recovery-case lifecycle.

The property that matters most is that a closed case cannot reopen: a late webhook must
not be able to resurrect a settled case and charge the customer again.
"""
from __future__ import annotations

import pytest

from backend.app.domain.states import (
    TERMINAL,
    TRANSITIONS,
    IllegalTransition,
    assert_well_formed,
    can_transition,
    is_terminal,
    mermaid,
    transition,
)
from backend.app.models.enums import CaseStatus


def test_lifecycle_is_well_formed():
    """Every state declared, every state reachable, every state able to terminate."""
    assert_well_formed()


def test_a_closed_case_cannot_reopen():
    for terminal in TERMINAL:
        for target in CaseStatus:
            assert not can_transition(terminal, target), \
                f"{terminal.value} -> {target.value} would resurrect a closed case"
            with pytest.raises(IllegalTransition, match="already closed"):
                transition(terminal, target)


def test_the_normal_path_is_permitted():
    s = transition(CaseStatus.PENDING, CaseStatus.IN_PROGRESS)
    s = transition(s, CaseStatus.IN_PROGRESS, "re-plan after a block")
    assert transition(s, CaseStatus.RECOVERED) is CaseStatus.RECOVERED


def test_a_case_may_close_before_any_work_happens():
    """A fraud hold is escalated on sight; forcing it through IN_PROGRESS first would
    record work that never occurred."""
    assert transition(CaseStatus.PENDING, CaseStatus.ESCALATED) is CaseStatus.ESCALATED


def test_terminal_set_is_derived_not_restated():
    assert frozenset(s for s, nxt in TRANSITIONS.items() if not nxt) == TERMINAL
    assert all(is_terminal(s) for s in TERMINAL)


def test_diagram_is_generated_from_the_table():
    """Documentation that cannot drift from the code it documents."""
    d = mermaid()
    assert d.startswith("stateDiagram-v2")
    for s in CaseStatus:
        assert s.value in d
