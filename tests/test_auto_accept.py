"""Low-risk facts should not queue up waiting for approval nobody gives.

Every proposed fact currently waits for an explicit human accept. In practice that
means the store sits empty and the assistant works from conversation history
instead of memory — the gate is safe but it is also unused.

The fix is a risk split, not an unlocked gate:

* **auto-acceptable** — a plain preference or a small environment fact, with no
  conflict flagged and no near-duplicate of something already stored. Getting one
  of these wrong costs a correction, not trust.
* **must be reviewed** — anything that supersedes an accepted fact, contradicts
  one, is a decision or a constraint, mentions a credential/account/payment, or is
  a correction of the user's own stated position. Those change what the assistant
  believes about the world, so a human decides.

The dangerous direction is auto-accepting something that should have been reviewed,
so every test here is written around that: when in doubt, the answer is review.

Run: python -m pytest tests/test_auto_accept.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.review import (AUTO_ACCEPT, REVIEW_REQUIRED, classify_risk,
                          is_auto_acceptable)


def test_plain_preference_is_auto_acceptable():
    verdict, reason = classify_risk(
        {"kind": "preference", "claim": "The user prefers terse status updates"},
        conflict=False, duplicate=False)
    assert verdict == AUTO_ACCEPT, reason


def test_environment_fact_is_auto_acceptable():
    verdict, reason = classify_risk(
        {"kind": "environment", "claim": "The development machine has 8 GB of RAM"},
        conflict=False, duplicate=False)
    assert verdict == AUTO_ACCEPT, reason


def test_a_flagged_conflict_is_never_auto_accepted():
    """A contradiction changes what the assistant believes — a human decides."""
    verdict, reason = classify_risk(
        {"kind": "preference", "claim": "The user prefers verbose output"},
        conflict=True, duplicate=False)
    assert verdict == REVIEW_REQUIRED
    assert "conflict" in reason.lower()


def test_proximity_to_a_stored_fact_is_never_auto_accepted():
    """The slot-value trap.

    The conflict detector is negation-only, so "prefers dark mode" vs "prefers light
    mode" is flagged as neither conflict nor duplicate. If proximity did not block
    auto-acceptance, the second claim would silently replace the first — a belief the
    user approved, overwritten with no approval. Proximity is the only signal that
    catches it, so it must block.
    """
    verdict, reason = classify_risk(
        {"kind": "preference", "claim": "The user prefers light mode"},
        conflict=False, duplicate=False, related=1)
    assert verdict == REVIEW_REQUIRED, (
        "a claim adjacent to a stored fact was auto-accepted")
    assert "close to" in reason.lower()


def test_a_near_duplicate_is_never_auto_accepted():
    """Replacing an existing fact silently would lose the old one's nuance."""
    verdict, reason = classify_risk(
        {"kind": "preference", "claim": "The user prefers terse updates"},
        conflict=False, duplicate=True)
    assert verdict == REVIEW_REQUIRED
    assert "duplicate" in reason.lower()


@pytest.mark.parametrize("kind", ["decision", "constraint", "correction"])
def test_consequential_kinds_require_review(kind):
    """Decisions and constraints steer future behaviour; they are not trivia."""
    verdict, reason = classify_risk(
        {"kind": kind, "claim": "Never deploy on Fridays"}, conflict=False,
        duplicate=False)
    assert verdict == REVIEW_REQUIRED, f"{kind} was auto-accepted: {reason}"


@pytest.mark.parametrize("claim", [
    "The user's API key for the billing service starts with sk-",
    "The user's card number ends in 4242",
    "The account login for the portal is under sakib@example.com",
    "The password hint is the street they grew up on",
])
def test_anything_touching_secrets_or_money_requires_review(claim):
    verdict, reason = classify_risk(
        {"kind": "preference", "claim": claim}, conflict=False, duplicate=False)
    assert verdict == REVIEW_REQUIRED, f"auto-accepted sensitive claim: {claim}"


def test_unknown_kinds_default_to_review():
    """An unrecognised kind is a reason to ask, not a reason to assume."""
    verdict, reason = classify_risk(
        {"kind": "speculation", "claim": "The user might like Rust"},
        conflict=False, duplicate=False)
    assert verdict == REVIEW_REQUIRED
    assert "kind" in reason.lower()


def test_missing_kind_defaults_to_review():
    verdict, reason = classify_risk({"claim": "Something vague"}, conflict=False,
                                    duplicate=False)
    assert verdict == REVIEW_REQUIRED


def test_empty_claim_defaults_to_review():
    verdict, _reason = classify_risk({"kind": "preference", "claim": "   "},
                                     conflict=False, duplicate=False)
    assert verdict == REVIEW_REQUIRED


def test_the_gate_is_off_unless_configured():
    """Default behaviour must not change: no config, no auto-acceptance."""
    assert is_auto_acceptable({}, {"kind": "preference", "claim": "x"}) is False
    assert is_auto_acceptable({"auto_accept": False},
                              {"kind": "preference", "claim": "x"}) is False
    assert is_auto_acceptable({"auto_accept": True},
                              {"kind": "preference", "claim": "x"}) is True
