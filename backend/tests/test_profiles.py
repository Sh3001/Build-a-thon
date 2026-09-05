"""Deployment profiles must fail at boot rather than look healthy while unsafe."""
from __future__ import annotations

import pytest

from backend.app.profiles import (
    PROFILES,
    ProfileError,
    current,
    describe,
    validate,
    validate_or_raise,
)


def test_development_needs_nothing():
    assert validate({"RECOVERAI_PROFILE": "development"}) == []


def test_production_demands_a_signing_secret_and_a_real_database():
    problems = validate({"RECOVERAI_PROFILE": "production"})
    assert any("RECOVERAI_JWT_SECRET" in p for p in problems)
    assert any("RECOVERAI_DB_URL" in p for p in problems)
    assert any("WEBHOOK_SECRET" in p for p in problems)


def test_production_refuses_a_mock_rail():
    """A deployment that believes it is moving money and is running a simulator would
    report simulated recoveries as real ones."""
    problems = validate({
        "RECOVERAI_PROFILE": "production", "RECOVERAI_JWT_SECRET": "x" * 40,
        "RECOVERAI_DB_URL": "postgresql://u:p@h/db",
        "RECOVERAI_WEBHOOK_SECRET": "whsec_" + "y" * 32,
    })
    assert any("mock" in p for p in problems)


def test_a_fully_configured_production_environment_is_satisfied():
    env = {
        "RECOVERAI_PROFILE": "production", "RECOVERAI_JWT_SECRET": "x" * 40,
        "RECOVERAI_DB_URL": "postgresql://u:p@h/db",
        "RECOVERAI_WEBHOOK_SECRET": "whsec_" + "y" * 32,
        "RECOVERAI_PAYMENT_RAIL": "stripe_sandbox",
    }
    assert validate(env) == []
    assert validate_or_raise(env).name == "production"


def test_turning_auth_off_in_production_is_refused():
    env = {
        "RECOVERAI_PROFILE": "production", "RECOVERAI_JWT_SECRET": "x" * 40,
        "RECOVERAI_DB_URL": "postgresql://u:p@h/db",
        "RECOVERAI_WEBHOOK_SECRET": "whsec_" + "y" * 32,
        "RECOVERAI_PAYMENT_RAIL": "stripe_sandbox",
        "RECOVERAI_AUTH_REQUIRED": "false",
    }
    assert any("not optional" in p for p in validate(env))


def test_a_misconfigured_profile_raises_rather_than_warns():
    with pytest.raises(ProfileError, match="Refusing to start"):
        validate_or_raise({"RECOVERAI_PROFILE": "production"})


def test_a_typo_in_the_profile_name_is_refused():
    """Guessing 'development' for a typo'd 'prodcution' is the failure this prevents."""
    with pytest.raises(ProfileError, match="unknown profile"):
        current({"RECOVERAI_PROFILE": "prodcution"})


def test_testing_forbids_calling_out():
    assert not PROFILES["testing"].allows_llm


def test_describe_reports_what_is_unmet():
    d = describe({"RECOVERAI_PROFILE": "staging"})
    assert d["satisfied"] is False and d["problems"]
