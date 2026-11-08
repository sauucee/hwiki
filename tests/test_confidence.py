"""Advisory recall confidence (Phase 4, model-free half).

There is deliberately no "does this answer the question?" verdict here. Two
deterministic attempts to derive one were measured and failed, and Honcho needs a
reasoning model for it. What this exposes is the raw signal a caller *with more
context* can judge, plus a caveat that must travel with it.

The tests pin three things: the numbers are honest, confidence never changes what
recall returns, and the caveat cannot be lost.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _hwiki_path  # noqa: F401,E402

from hwiki.facts import Embedder, ProposalStore  # noqa: E402

MEMORY_REL = Path("Hermes/Memory/Auto")


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "vault" / MEMORY_REL).mkdir(parents=True)
        self.store = ProposalStore(self.root / "facts.sqlite",
                                   embedder=Embedder(mode="hash"))
        for claim in ("User prefers dark mode in every editor",
                      "User prefers terse status updates with no pings"):
            p = self.store.propose({"kind": "preference", "claim": claim,
                                    "evidence": "stated"}, user_text="stated",
                                   session_id="s")
            self.store.accept(p["id"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_confidence_reports_the_top_hit(self):
        info = self.store.recall_confidence("dark mode editor preference")
        self.assertGreater(info["best"], 0.0)
        self.assertGreaterEqual(info["candidates"], 1)

    def test_empty_result_is_not_confident(self):
        info = self.store.recall_confidence("what is the airspeed of a swallow")
        self.assertEqual(info["best"], 0.0)
        self.assertEqual(info["candidates"], 0)
        self.assertFalse(info["confident"])

    def test_caveat_always_travels_with_the_signal(self):
        for query in ("dark mode editor preference",
                      "what is the airspeed of a swallow"):
            info = self.store.recall_confidence(query)
            self.assertEqual(info["caveat"], "relatedness is not answerability")

    def test_confidence_never_changes_what_recall_returns(self):
        query = "dark mode editor preference"
        before = self.store.recall(query, limit=8)
        self.store.recall_confidence(query)
        after = self.store.recall(query, limit=8)
        self.assertEqual([h["id"] for h in before], [h["id"] for h in after],
                         "confidence must be a pure observation")

    def test_margin_is_best_minus_runner_up(self):
        info = self.store.recall_confidence("user prefers dark mode")
        hits = self.store.recall("user prefers dark mode", limit=8)
        expected = round(hits[0]["score"] - (hits[1]["score"] if len(hits) > 1 else 0.0), 3)
        self.assertAlmostEqual(info["margin"], expected, places=3)

    def test_overlap_is_a_ratio(self):
        info = self.store.recall_confidence("dark mode editor preference")
        self.assertGreaterEqual(info["overlap"], 0.0)
        self.assertLessEqual(info["overlap"], 1.0)

    def test_floor_matches_the_tier(self):
        info = self.store.recall_confidence("dark mode")
        self.assertEqual(info["floor"], round(float(self.store.embedder.noise_floor), 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
