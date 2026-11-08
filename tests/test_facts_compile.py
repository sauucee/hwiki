"""Integration: compile runs inside the real promote path.

These exercise ``FactsEngine._promote`` end-to-end against a throwaway vault,
proving that an accepted fact that refines an existing one ends up as a
section of the owning page — and that every failure mode falls back to the
pre-compile behaviour (a standalone page) without losing the fact.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _hwiki_path  # noqa: F401,E402

from hwiki.facts import read_fact_pages  # noqa: E402
from hwiki.facts_provider import FactsEngine  # noqa: E402

MEMORY_REL = Path("Hermes/Memory/Auto")


class CompilePromoteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.home = self.root / "hermes_home"
        self.vault = self.root / "vault"
        (self.vault / MEMORY_REL).mkdir(parents=True)
        self.home.mkdir()
        self._engines = []

    def tearDown(self):
        for engine in self._engines:
            try:
                store = engine._store
                if store is not None:
                    store.close()
            except Exception:
                pass
        self.tmp.cleanup()

    def engine(self, **config):
        facts_cfg = {"embeddings": {"mode": "hash"}, "compile_enabled": True}
        facts_cfg.update(config)
        eng = FactsEngine(str(self.home), self.vault, MEMORY_REL, config=facts_cfg)
        self._engines.append(eng)
        eng.store()
        return eng

    def accept(self, engine, claim, evidence="stated by the user", kind="preference"):
        proposal = engine.store().propose(
            {"kind": kind, "claim": claim, "evidence": evidence},
            user_text=evidence, session_id="s1",
        )
        engine.store().accept(proposal["id"])
        return engine.store().get(proposal["id"])

    # -- happy path ------------------------------------------------------

    def test_refinement_merges_into_the_owning_page(self):
        eng = self.engine()
        base = self.accept(eng, "The user prefers dark mode in editors")
        first_page = eng._promote(base["id"])
        self.assertIsNotNone(first_page)
        self.assertTrue(Path(first_page).exists())

        # A near-duplicate refinement (same topic -> above the merge bar).
        refined = self.accept(eng, "The user prefers dark mode in editors and terminals")
        merged_page = eng._promote(refined["id"])
        self.assertIsNotNone(merged_page)

        owner = Path(merged_page)
        text = owner.read_text(encoding="utf-8")
        # This MUST merge, not fall back: if it ever takes the fallback branch
        # the test would silently prove nothing.
        self.assertIn("<!-- hwiki:compiled:start -->", text,
                      "a 0.88-similarity refinement must merge into its owner")
        self.assertIn("### The user prefers dark mode in editors and terminals", text)
        self.assertIn("# The user prefers dark mode in editors", text)
        self.assertIn("compiled_sections: 1", text)

        # Lineage points at the page that actually holds the claim.
        row = eng.store().get(refined["id"])
        self.assertEqual(Path(row["page_path"]), owner)

        # One page for the topic, not two near-duplicates.
        pages = [p.name for p in (self.vault / MEMORY_REL).glob("*.md")
                 if p.name != "audit-log.md"]
        self.assertEqual(len(pages), 1, pages)
        self.assertEqual(pages[0], Path(first_page).name)

    def test_merged_claim_stays_recallable(self):
        """Guarantee 1: a merged fact must never vanish from recall."""
        eng = self.engine()
        base = self.accept(eng, "The user prefers dark mode in editors")
        eng._promote(base["id"])
        refined = self.accept(eng, "The user prefers dark mode in editors and terminals")
        eng._promote(refined["id"])

        claims = [f["claim"] for f in
                  read_fact_pages(self.vault, MEMORY_REL, max_facts=50)]
        self.assertIn("The user prefers dark mode in editors", claims)
        self.assertIn("The user prefers dark mode in editors and terminals", claims,
                      "merged claim must remain visible to recall")

    # -- guarantees ------------------------------------------------------

    def test_conflict_is_never_merged(self):
        """Guarantee 2: contradictions keep their own page."""
        eng = self.engine()
        base = self.accept(eng, "The user prefers dark mode in editors")
        eng._promote(base["id"])

        negated = self.accept(eng, "The user does not prefer dark mode in editors")
        row = eng.store().get(negated["id"])
        self.assertEqual(row["possible_conflict"], 1)
        page = eng._promote(negated["id"])
        self.assertTrue(Path(page).exists())
        text = Path(page).read_text(encoding="utf-8")
        self.assertNotIn("<!-- hwiki:compiled:start -->", text,
                         "a conflict must not be folded into the claim it disputes")

    def test_read_only_engine_never_compiles(self):
        eng = self.engine()
        eng.read_only = True
        base = self.accept(eng, "The user prefers dark mode in editors")
        eng._promote(base["id"])
        refined = self.accept(eng, "The user prefers dark mode in editors and terminals")
        eng._promote(refined["id"])
        for page in (self.vault / MEMORY_REL).glob("*.md"):
            self.assertNotIn("<!-- hwiki:compiled:start -->",
                             page.read_text(encoding="utf-8"))

    def test_compile_enabled_false_reproduces_pre_compile_behaviour(self):
        eng = self.engine(compile_enabled=False)
        base = self.accept(eng, "The user prefers dark mode in editors")
        eng._promote(base["id"])
        refined = self.accept(eng, "The user prefers dark mode in editors and terminals")
        eng._promote(refined["id"])
        pages = [p for p in (self.vault / MEMORY_REL).glob("*.md")
                 if p.name != "audit-log.md"]
        self.assertEqual(len(pages), 2, [p.name for p in pages])
        for page in pages:
            self.assertNotIn("<!-- hwiki:compiled:start -->",
                             page.read_text(encoding="utf-8"))

    def test_compile_failure_keeps_the_standalone_page(self):
        """Guarantee 4: a broken compile must not lose the accepted fact."""
        eng = self.engine()
        base = self.accept(eng, "The user prefers dark mode in editors")
        eng._promote(base["id"])
        refined = self.accept(eng, "The user prefers dark mode in editors and terminals")

        original = eng._candidate_owners

        def boom(proposal):
            raise RuntimeError("simulated compile failure")

        eng._candidate_owners = boom  # type: ignore[assignment]
        try:
            page = eng._promote(refined["id"])
        finally:
            eng._candidate_owners = original  # type: ignore[assignment]

        self.assertIsNotNone(page)
        self.assertTrue(Path(page).exists(), "the fact must survive a compile crash")
        self.assertIn("The user prefers dark mode in editors and terminals",
                      Path(page).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
