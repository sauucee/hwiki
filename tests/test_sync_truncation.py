"""A merged claim must not be silently shortened by a hand-edit sync.

``compile`` writes a claim into its owner page as a ``###`` section, and the
heading is truncated to ``_MAX_CLAIM`` (160) characters. ``sync_from_vault``
treats the vault as authoritative and reads the claim *back* from the page — so a
long merged claim was rewritten in the database as its own truncated heading.

The loss was permanent: the page no longer holds the full text, the DB has been
overwritten with the short copy, and nothing records that anything was cut. These
tests pin the property that a sync which did not change the *fact* leaves the
stored claim alone.

Run: python -m pytest tests/test_sync_truncation.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.compile import _MAX_CLAIM
from hwiki.facts import Embedder, ProposalStore, write_fact_page

MEM = Path("Hermes/Memory/Auto")

LONG_CLAIM = (
    "The deployment pipeline for the customer-facing billing service requires "
    "manual approval from the platform team lead before any production rollout "
    "proceeds, per the change advisory board decision of October."
)


def _env(tmp: Path):
    vault = tmp / "vault"
    (vault / MEM).mkdir(parents=True)
    store = ProposalStore(tmp / "facts.sqlite", embedder=Embedder(mode="hash"))
    return vault, store


def _accept(store: ProposalStore, vault: Path, claim: str):
    proposal = store.propose({"kind": "preference", "claim": claim,
                              "evidence": "stated"}, user_text="stated",
                             session_id="s1")
    store.accept(proposal["id"])
    page = write_fact_page(vault, MEM, store.get(proposal["id"]))
    store.set_page_path(proposal["id"], page)
    return store.get(proposal["id"])


def test_long_claim_is_longer_than_the_heading_limit():
    """Guard the fixture: the claim must actually exceed the truncation point."""
    assert len(LONG_CLAIM) > _MAX_CLAIM


def test_sync_keeps_a_stored_claim_that_the_page_heading_shortened():
    """The page heading is a truncation of the claim, not a new claim.

    A sync that finds no real edit must not rewrite the stored claim with the
    shortened heading; that silently destroys the tail of the fact.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        vault, store = _env(tmp)
        fact = _accept(store, vault, LONG_CLAIM)
        assert len(fact["claim"]) > _MAX_CLAIM

        # Simulate what compile does: the page heading carries the truncated form.
        page = Path(fact["page_path"])
        text = page.read_text(encoding="utf-8")
        heading = next(l for l in text.splitlines() if l.startswith("# "))
        page.write_text(text.replace(heading, f"# {LONG_CLAIM[:_MAX_CLAIM]}", 1),
                        encoding="utf-8")

        store.sync_from_vault()

        after = store.get(fact["id"])
        assert after["claim"] == LONG_CLAIM, (
            f"sync truncated the stored claim to {len(after['claim'])} chars: "
            f"{after['claim']!r}")
        assert after["claim"].endswith("decision of October."), after["claim"]


def test_sync_still_honours_a_genuine_hand_edit():
    """The fix must not freeze the claim against real edits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        vault, store = _env(tmp)
        fact = _accept(store, vault, "The user prefers dark mode in editors")

        page = Path(fact["page_path"])
        text = page.read_text(encoding="utf-8")
        heading = next(l for l in text.splitlines() if l.startswith("# "))
        edited = "The user prefers a light theme in editors"
        page.write_text(text.replace(heading, f"# {edited}", 1), encoding="utf-8")

        store.sync_from_vault()
        assert store.get(fact["id"])["claim"] == edited
