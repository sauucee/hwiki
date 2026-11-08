"""Query citations.

The LLM Wiki pattern asks for answers that can be traced back to the page they
came from. Every recall hit therefore carries a wikilink to its vault page and the
date the fact was accepted. Two things must hold: the citation is usable, and it
never leaks the absolute filesystem path (a home directory) into caller output.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _hwiki_path  # noqa: F401,E402

from hwiki.facts import Embedder, ProposalStore, write_fact_page  # noqa: E402

MEMORY_REL = Path("Hermes/Memory/Auto")


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name) / "vault"
        (self.vault / MEMORY_REL).mkdir(parents=True)
        self.store = ProposalStore(Path(self.tmp.name) / "facts.sqlite",
                                   embedder=Embedder(mode="hash"))

    def tearDown(self):
        self.tmp.cleanup()

    def accept(self, claim):
        p = self.store.propose({"kind": "preference", "claim": claim,
                                "evidence": "stated"}, user_text="stated",
                               session_id="s")
        self.store.accept(p["id"])
        page = write_fact_page(self.vault, MEMORY_REL, self.store.get(p["id"]))
        self.store.set_page_path(p["id"], page)
        return p["id"], Path(page)

    def test_hit_carries_a_wikilink_to_its_page(self):
        _pid, page = self.accept("User prefers dark mode in every tool")
        hits = self.store.recall("dark mode preference", limit=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["page"], page.stem)
        self.assertEqual(hits[0]["cite"], f"[[{page.stem}]]")

    def test_hit_carries_the_accepted_date(self):
        self.accept("User prefers dark mode in every tool")
        hit = self.store.recall("dark mode preference", limit=5)[0]
        self.assertTrue(hit.get("accepted"), "a citation needs an acceptance date")
        self.assertEqual(len(hit["accepted"]), len("YYYY-MM-DD"))

    def test_citation_never_exposes_an_absolute_path(self):
        self.accept("User prefers dark mode in every tool")
        hit = self.store.recall("dark mode preference", limit=5)[0]
        self.assertNotIn(str(self.tmp.name), hit["cite"] + hit["page"])
        self.assertFalse(hit["page"].startswith("/"))
        self.assertNotIn(".md", hit["cite"], "the wikilink carries the slug, not a filename")

    def test_fact_without_a_page_has_no_citation(self):
        p = self.store.propose({"kind": "preference", "claim": "User likes terse output",
                                "evidence": "stated"}, user_text="stated",
                               session_id="s")
        self.store.accept(p["id"])
        hit = self.store.recall("terse output", limit=5)[0]
        self.assertEqual(hit["cite"], "")
        self.assertEqual(hit["page"], "")

    def test_every_kept_hit_is_citable(self):
        # Query each fact on its own terms: a combined "what are the user
        # preferences" query honestly scores below the noise floor, which is the
        # correct behaviour and not a citation problem.
        for claim, query in (
            ("User prefers dark mode in every editor", "dark mode editor"),
            ("User prefers terse status updates", "terse status updates"),
            ("Deploys are frozen on Fridays", "deploy freeze friday"),
        ):
            self.accept(claim)
            hits = self.store.recall(query, limit=5)
            self.assertTrue(hits, f"nothing retrieved for {query!r}")
            for hit in hits:
                self.assertTrue(hit["cite"].startswith("[["), hit["cite"])
                self.assertNotIn("/", hit["cite"])

    def test_citation_still_names_the_superseded_page_in_lineage(self):
        old, old_page = self.accept("Staging server is moving to Hetzner VPS")
        new, _new_page = self.accept("Staging server stays on Fly.io")
        self.store.supersede(new, old)
        hit = self.store.recall("where does staging run", limit=5)[0]
        self.assertIn("fly.io", hit["claim"].lower())
        self.assertEqual(hit["replaced"], "Staging server is moving to Hetzner VPS")
        self.assertNotEqual(hit["page"], old_page.stem,
                            "the citation points at the current page, not the tombstone")


if __name__ == "__main__":
    unittest.main(verbosity=2)
