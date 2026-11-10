"""Switching embedding tiers must not silently blind the memory system.

The two tiers produce vectors of different lengths (ONNX 1536 floats, hashed n-grams
8192). ``_cosine_bytes`` returns 0.0 on a length mismatch, so after a mode switch every
stored fact scores 0 against every query: recall does not fail, it just stops finding
anything. That is the failure mode these tests exist to prevent, because nothing in the
UI would ever show it.

Run: python -m pytest tests/test_embedding_migration.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.facts import Embedder, ProposalStore, _cosine_bytes


class _FixedEmbedder(Embedder):
    """An embedder with a controllable vector length, to stand in for each tier."""

    def __init__(self, dim: int, label: str = "onnx"):
        self._dim = dim
        self.mode = label
        self._onnx_failed = False

    def embed(self, text: str) -> bytes:
        import struct
        seed = float(sum(bytearray(str(text or "").encode())) or 1)
        vec = [seed / 1000.0] * self._dim
        return struct.pack(f"<{self._dim}f", *vec)


def _store(tmp_path: Path, embedder) -> ProposalStore:
    return ProposalStore(str(tmp_path / "facts.sqlite"), embedder=embedder)


def test_vectors_of_different_lengths_score_zero():
    """The reason migration is needed: a mismatch fails quietly, not loudly."""
    import struct
    a = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    b = struct.pack("<8f", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert _cosine_bytes(a, b) == 0.0


def test_switching_tier_reembeds_stored_facts(tmp_path: Path):
    """After a mode change, old facts must still be matchable in the new space."""
    first = _store(tmp_path, _FixedEmbedder(8192, "hash"))
    first.propose({"kind": "preference", "claim": "The user prefers terse updates",
                   "evidence": "chat"}, user_text="x", session_id="s1")

    # Re-open with the other tier, as a restart after a config change would.
    second = _store(tmp_path, _FixedEmbedder(1536, "onnx"))
    with sqlite3.connect(str(tmp_path / "facts.sqlite")) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT claim, embedding FROM proposals").fetchall()
    assert rows, "the stored fact vanished"
    for row in rows:
        assert len(row["embedding"]) == 1536 * 4, (
            "a stored vector was left in the old tier's shape, so it now scores 0 "
            "against every query")

    # And matching actually works in the new space.
    new_claim = "The user prefers terse updates"
    assert _cosine_bytes(second.embedder.embed(new_claim),
                         rows[0]["embedding"]) > 0.9


def test_no_reembed_when_the_tier_is_unchanged(tmp_path: Path):
    """A normal restart must not rewrite the table."""
    _store(tmp_path, _FixedEmbedder(1536, "onnx")).propose(
        {"kind": "preference", "claim": "The user likes chai", "evidence": "chat"},
        user_text="x", session_id="s1")

    same = _store(tmp_path, _FixedEmbedder(1536, "onnx"))
    with sqlite3.connect(str(tmp_path / "facts.sqlite")) as conn:
        conn.row_factory = sqlite3.Row
        sig = conn.execute(
            "SELECT value FROM meta WHERE key='embedding_sig'").fetchone()["value"]
    assert sig.endswith(":1536"), f"signature records the wrong width: {sig}"

    # Re-opening again keeps the same signature and does not churn.
    _store(tmp_path, _FixedEmbedder(1536, "onnx"))
    with sqlite3.connect(str(tmp_path / "facts.sqlite")) as conn:
        again = conn.execute(
            "SELECT value FROM meta WHERE key='embedding_sig'").fetchone()[0]
    assert again == sig


def test_first_run_stamps_the_signature_without_rewriting(tmp_path: Path):
    """A brand-new database has nothing to migrate; it just records the tier."""
    _store(tmp_path, _FixedEmbedder(1536, "onnx"))
    with sqlite3.connect(str(tmp_path / "facts.sqlite")) as conn:
        sig = conn.execute(
            "SELECT value FROM meta WHERE key='embedding_sig'").fetchone()[0]
    assert sig == "onnx:1536", "the signature must record dimensions, not byte count"
