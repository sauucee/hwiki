"""End-to-end: does the risk split actually change what lands in the database?

The unit tests next door prove ``classify_risk`` returns the right verdict. That is
not the same as the store acting on it — a gate wired to the wrong variable would
pass those tests and still auto-accept a contradiction. These tests drive a real
``ProposalStore`` and read the resulting row back.

Run: python -m pytest tests/test_auto_accept_store.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.facts import ProposalStore


def _store(tmp_path: Path, **config) -> ProposalStore:
    return ProposalStore(str(tmp_path / "facts.sqlite"), config=config)


def _propose(store, claim, *, kind="preference"):
    return store.propose(
        {"kind": kind, "claim": claim, "evidence": "said so in chat"},
        user_text=claim, session_id="s1")


def _status(store, proposal_id) -> str:
    return store.get(proposal_id)["status"]


def test_opted_in_store_accepts_a_plain_preference(tmp_path: Path):
    """The whole point: a trivial fact stops waiting for an approval nobody gives."""
    store = _store(tmp_path, auto_accept=True)
    proposal = _propose(store, "The user prefers terse status updates")
    assert _status(store, proposal["id"]) == "accepted"


def test_the_default_store_still_queues_everything(tmp_path: Path):
    """Upgrading must not silently start writing memories on its own."""
    store = _store(tmp_path)  # no auto_accept
    proposal = _propose(store, "The user prefers terse status updates")
    assert _status(store, proposal["id"]) == "proposed"


def test_opted_in_store_still_queues_a_decision(tmp_path: Path):
    """A decision steers future behaviour, so it waits even when opted in."""
    store = _store(tmp_path, auto_accept=True)
    proposal = _propose(store, "Never deploy on Fridays", kind="decision")
    assert _status(store, proposal["id"]) == "proposed"


def test_opted_in_store_still_queues_a_credential(tmp_path: Path):
    store = _store(tmp_path, auto_accept=True)
    proposal = _propose(
        store, "The user's API key for billing starts with sk-live")
    assert _status(store, proposal["id"]) == "proposed"


def test_a_contradiction_is_queued_even_when_opted_in(tmp_path: Path):
    """The dangerous case: a new claim that replaces an accepted one.

    "dark mode" vs "light mode" is a slot-value contradiction. The conflict detector
    is negation-only and does NOT flag it (documented, measured — see
    benchmarks/phase4-floor-shape.md), so if the gate trusted that flag alone the
    second claim would be silently accepted and the assistant would believe the
    opposite of what the user last approved.

    What must hold: the claim is queued for review. Do not assert the flag fires —
    it will not, and pretending otherwise is how a real gap gets hidden.
    """
    store = _store(tmp_path, auto_accept=True)
    first = _propose(store, "The user prefers dark mode in editors")
    assert _status(store, first["id"]) == "accepted"

    second = _propose(store, "The user prefers light mode in editors")
    assert _status(store, second["id"]) == "proposed", (
        "a replacing claim was auto-accepted over an accepted one")
    # The engine does not claim to have spotted the contradiction, and the record
    # should not say it did. Proximity is what routed it to review.
    assert not store.get(second["id"])["possible_conflict"], (
        "the fixture now trips the negation rule, so this no longer tests the gap")


def test_an_exact_duplicate_is_not_accepted_twice(tmp_path: Path):
    """Proposing the same fact again must not create a second accepted row."""
    store = _store(tmp_path, auto_accept=True)
    first = _propose(store, "The user prefers terse status updates")
    again = _propose(store, "The user prefers terse status updates")
    assert again["id"] == first["id"], "a duplicate created a second proposal"
    accepted = [p for p in store.list_proposals(status="accepted", limit=10)]
    assert len(accepted) == 1, f"{len(accepted)} accepted rows for one fact"


def test_auto_accepted_fact_is_readable_as_memory(tmp_path: Path):
    """An auto-accepted fact must be visible to recall, not just marked accepted."""
    store = _store(tmp_path, auto_accept=True)
    _propose(store, "The user prefers terse status updates")
    facts = store.list_proposals(status="accepted", limit=10)
    assert any("terse" in p["claim"] for p in facts), (
        "the accepted fact is not readable back as memory")
