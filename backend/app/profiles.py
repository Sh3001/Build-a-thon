"""Deployment profiles.

Five environments with genuinely different requirements, and one place that says what
each one demands. Without this the difference between "development" and "production" is
whichever environment variables someone remembered to set, which is how a system ends up
in production with authentication off and mock rails wired in.

`validate_profile()` is called at startup and *raises* rather than warning. A process that
starts in a misconfigured production profile is a process that will look healthy while
being unsafe -- and the whole value of a profile is that it fails at boot instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    #: Environment variables that must be set and non-trivial.
    requires: tuple[str, ...] = ()
    #: Settings that must be false in this profile.
    forbids_mock_rail: bool = False
    requires_auth: bool = False
    requires_persistent_db: bool = False
    requires_webhook_secret: bool = False
    allows_anonymous_reads: bool = True
    #: Whether the LLM may be given a case at all. Off in testing so a suite never calls out.
    allows_llm: bool = True


PROFILES: dict[str, Profile] = {
    "development": Profile(
        "development",
        "Local work. Mock rails, SQLite, anonymous reads, no secrets required.",
        allows_anonymous_reads=True),
    "simulation": Profile(
        "simulation",
        "Experiment runs. Identical to development except that it is never pointed at "
        "a real rail -- a sweep that accidentally talks to a sandbox produces results "
        "nobody can interpret.",
        allows_llm=True),
    "testing": Profile(
        "testing",
        "CI. No network, no API keys, no LLM: a suite that can call out is a suite whose "
        "results depend on someone else's uptime.",
        allows_llm=False),
    "staging": Profile(
        "staging",
        "Production shape, sandbox rails. Everything production requires except real "
        "money.",
        requires=("RECOVERAI_JWT_SECRET",),
        requires_auth=True, requires_persistent_db=True, requires_webhook_secret=True,
        allows_anonymous_reads=False),
    "production": Profile(
        "production",
        "Real money. Authentication mandatory, mock rails refused, persistent database "
        "and webhook verification required.",
        requires=("RECOVERAI_JWT_SECRET", "RECOVERAI_DB_URL"),
        forbids_mock_rail=True, requires_auth=True, requires_persistent_db=True,
        requires_webhook_secret=True, allows_anonymous_reads=False, allows_llm=True),
}


class ProfileError(RuntimeError):
    """The environment does not satisfy the profile it claims to be."""


def current(env: dict[str, str] | None = None) -> Profile:
    e = env if env is not None else dict(os.environ)
    name = e.get("RECOVERAI_PROFILE", "development").lower()
    if name not in PROFILES:
        raise ProfileError(
            f"unknown profile {name!r}; one of {sorted(PROFILES)}. There is no default "
            f"for an unrecognised name -- guessing 'development' for a typo'd "
            f"'prodcution' is exactly the failure this check exists to prevent.")
    return PROFILES[name]


def validate(env: dict[str, str] | None = None) -> list[str]:
    """Return the profile's unmet requirements. Empty means the environment is coherent."""
    e = env if env is not None else dict(os.environ)
    profile = current(e)
    problems: list[str] = []

    for key in profile.requires:
        value = e.get(key, "")
        if not value or len(value) < 8:
            problems.append(f"{key} is required in the {profile.name} profile")

    if profile.requires_auth:
        disabled = e.get("RECOVERAI_AUTH_REQUIRED", "").lower() in ("0", "false", "no", "off")
        if disabled:
            problems.append(
                f"RECOVERAI_AUTH_REQUIRED is off in the {profile.name} profile; "
                f"authentication is not optional here")

    if profile.requires_persistent_db and not e.get("RECOVERAI_DB_URL"):
        problems.append(
            f"the {profile.name} profile requires RECOVERAI_DB_URL: a SQLite file in a "
            f"container is state that disappears on the next deploy")

    if profile.requires_webhook_secret and not e.get("RECOVERAI_WEBHOOK_SECRET"):
        problems.append(
            "RECOVERAI_WEBHOOK_SECRET is unset; unverified payment events would be the "
            "only unauthenticated write path into this system")

    if profile.forbids_mock_rail and e.get("RECOVERAI_PAYMENT_RAIL", "mock") == "mock":
        problems.append(
            "RECOVERAI_PAYMENT_RAIL is 'mock' in the production profile; a deployment "
            "that believes it is moving money and is running a simulator would report "
            "simulated recoveries as real ones")

    return problems


def validate_or_raise(env: dict[str, str] | None = None) -> Profile:
    problems = validate(env)
    if problems:
        raise ProfileError(
            f"the {current(env).name} profile is not satisfied:\n  - "
            + "\n  - ".join(problems)
            + "\n\nRefusing to start. A process that boots misconfigured looks healthy "
              "while being unsafe.")
    return current(env)


def describe(env: dict[str, str] | None = None) -> dict:
    profile = current(env)
    problems = validate(env)
    return {
        "profile": profile.name,
        "description": profile.description,
        "requires_auth": profile.requires_auth,
        "requires_persistent_db": profile.requires_persistent_db,
        "forbids_mock_rail": profile.forbids_mock_rail,
        "allows_anonymous_reads": profile.allows_anonymous_reads,
        "allows_llm": profile.allows_llm,
        "satisfied": not problems,
        "problems": problems,
    }
