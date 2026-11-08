"""Wiki lint for the memory tree.

Lint is the LLM Wiki step that keeps a wiki from rotting silently: a page can lose
its provenance, its supersession link or its catalog entry and still be recalled
as if nothing happened. These tests pin the two properties that make it usable —
it finds real rot, and it never invents what it cannot read — plus the fact that
it is read-only.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _hwiki_path  # noqa: F401,E402

from hwiki.facts import Embedder, ProposalStore, write_fact_page  # noqa: E402
from hwiki.lint import lint_memory, format_report, to_dict, ERROR, WARN  # noqa: E402
from hwiki.store import WikiStore  # noqa: E402

MEMORY_REL = Path("Hermes/Memory/Auto")


class LintTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vault"
        self.mem = self.root / MEMORY_REL
        self.mem.mkdir(parents=True)
        (self.root / "index.md").write_text("# Index\n", encoding="utf-8")
        self.store = WikiStore(root=self.root, memory_dir=MEMORY_REL.as_posix())
        self.facts = ProposalStore(self.root / "facts.sqlite",
                                   embedder=Embedder(mode="hash"))

    def tearDown(self):
        self.tmp.cleanup()

    # -- helpers ---------------------------------------------------------

    def fact_page(self, claim, *, project="", peer="user"):
        p = self.facts.propose({"kind": "preference", "claim": claim,
                                "evidence": "stated"}, user_text="stated",
                               session_id="s", project=project, peer=peer)
        self.facts.accept(p["id"])
        page = write_fact_page(self.root, MEMORY_REL, self.facts.get(p["id"]))
        self.facts.set_page_path(p["id"], page)
        return Path(page)

    def checks(self, report):
        return [(f.check, f.severity) for f in report.findings]

    # -- clean vault -----------------------------------------------------

    def test_clean_vault_has_no_errors(self):
        self.fact_page("User prefers terse status updates")
        report = lint_memory(self.store)
        self.assertEqual(report.errors, [], format_report(report))
        self.assertTrue(report.ok)
        self.assertGreaterEqual(report.pages_checked, 1)

    def test_lint_is_deterministic(self):
        self.fact_page("User prefers terse status updates")
        self.fact_page("Deploys use the staging key")
        first = format_report(lint_memory(self.store), show_info=True)
        second = format_report(lint_memory(self.store), show_info=True)
        self.assertEqual(first, second, "two runs must produce identical output")

    def test_lint_never_writes(self):
        page = self.fact_page("User prefers terse status updates")
        before = {p: p.read_bytes() for p in self.mem.rglob("*.md")}
        before[page] = page.read_bytes()
        lint_memory(self.store)
        after = {p: p.read_bytes() for p in self.mem.rglob("*.md")}
        self.assertEqual(before, after, "lint must not modify the vault")

    # -- provenance rot --------------------------------------------------

    def test_missing_provenance_is_reported_not_invented(self):
        (self.mem / "broken.md").write_text(
            "---\nextractor: hwiki-extractor\n---\n\n# A claim with no provenance\n",
            encoding="utf-8")
        report = lint_memory(self.store)
        detail = " ".join(f.detail for f in report.findings)
        self.assertIn("`kind` missing", detail)
        self.assertIn("`status` missing", detail)
        self.assertIn("`accepted_at` missing", detail)
        # The point: it reports unknown rather than filling in a value.
        self.assertNotIn("kind=fact", detail)

    def test_page_without_a_claim_heading_is_an_error(self):
        (self.mem / "nohead.md").write_text(
            "---\nextractor: hwiki-extractor\nkind: fact\nstatus: accepted\n"
            "accepted_at: 2026-01-01\n---\n\nJust prose, no heading.\n",
            encoding="utf-8")
        report = lint_memory(self.store)
        self.assertIn(("claim.missing", ERROR), self.checks(report))

    def test_secret_material_in_a_handwritten_page_is_an_error(self):
        (self.mem / "leaky.md").write_text(
            "---\nextractor: hwiki-extractor\nkind: fact\nstatus: accepted\n---\n\n"
            "# Set the key with token=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\n"
            "> **Evidence:** pasted\n", encoding="utf-8")
        report = lint_memory(self.store)
        self.assertIn(("secret.content", ERROR), self.checks(report))

    def test_secret_material_in_a_handwritten_note_is_still_reported(self):
        # Pages without the extractor marker skip schema checks, but secrets are
        # never skipped — a hand-written note is the likeliest place one lands.
        (self.mem / "note.md").write_text(
            "# My note\n\npassword=hunter2024\n", encoding="utf-8")
        report = lint_memory(self.store)
        self.assertIn(("secret.content", ERROR), self.checks(report))

    # -- supersession ----------------------------------------------------

    def test_superseded_page_pointing_at_a_missing_page(self):
        page = self.fact_page("Staging server is moving to Hetzner VPS")
        text = page.read_text(encoding="utf-8").replace(
            "status: auto",
            'status: superseded\nsuperseded_by: "[[fact-does-not-exist]]"')
        page.write_text(text, encoding="utf-8")
        report = lint_memory(self.store)
        self.assertIn(("supersede.orphan", WARN), self.checks(report))

    def test_superseded_without_a_target_is_incomplete(self):
        page = self.fact_page("Staging server is moving to Hetzner VPS")
        page.write_text(page.read_text(encoding="utf-8").replace(
            "status: auto", "status: superseded"), encoding="utf-8")
        report = lint_memory(self.store)
        self.assertIn(("supersede.incomplete", WARN), self.checks(report))

    # -- duplicates and links -------------------------------------------

    def test_two_pages_with_the_same_accepted_claim_are_flagged(self):
        first = self.fact_page("Deploys are frozen on Fridays")
        second = self.mem / "manual-copy.md"
        second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
        report = lint_memory(self.store)
        self.assertIn(("duplicate.claim", WARN), self.checks(report))

    def test_dangling_wikilink_is_an_error(self):
        page = self.fact_page("User prefers terse status updates")
        page.write_text(page.read_text(encoding="utf-8")
                        + "\nSee [[no-such-page-anywhere]].\n", encoding="utf-8")
        report = lint_memory(self.store)
        self.assertIn(("link.dangling", ERROR), self.checks(report))

    # -- catalog ---------------------------------------------------------

    def test_catalog_missing_an_entry(self):
        self.fact_page("User prefers terse status updates")
        (self.root / "index.md").write_text(
            "# Index\n\n- [[some-other-page]]\n", encoding="utf-8")
        report = lint_memory(self.store)
        checks = self.checks(report)
        self.assertIn(("catalog.entry_missing", WARN), checks)
        self.assertIn(("catalog.stale", WARN), checks)

    # -- reporting -------------------------------------------------------

    def test_report_dict_is_machine_readable(self):
        (self.mem / "nohead.md").write_text(
            "---\nextractor: hwiki-extractor\nkind: fact\nstatus: accepted\n---\n"
            "\nno heading\n", encoding="utf-8")
        payload = to_dict(lint_memory(self.store))
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["counts"][ERROR], 1)
        self.assertEqual(payload["findings"][0]["severity"], ERROR)

    def test_ok_is_true_when_only_warnings_remain(self):
        (self.mem / "thin.md").write_text(
            "---\nextractor: hwiki-extractor\nkind: fact\nstatus: accepted\n---\n"
            "\n# A claim\n", encoding="utf-8")
        report = lint_memory(self.store)
        self.assertTrue(any(f.severity == WARN for f in report.findings))
        self.assertFalse(report.errors)
        self.assertTrue(report.ok)

    def test_empty_memory_zone_is_not_an_error(self):
        for page in self.mem.rglob("*.md"):
            page.unlink()
        report = lint_memory(self.store)
        self.assertEqual(report.pages_checked, 0)
        self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class SemanticSuggestionTests(unittest.TestCase):
    """The semantic pass may only *suggest*. It is advisory, read-only, and it
    must never declare a contradiction or resolve one.

    The 0.62-0.90 band is exercised by stubbing the scorer: the hashed tier has no
    semantic blur (near-identical claims score ~1.0, related ones ~0.15-0.23), so a
    real hash-tier pair never lands in the band. That is a property of the tier, not
    of the pass — see test_hash_tier_yields_no_suggestions.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vault"
        self.mem = self.root / MEMORY_REL
        self.mem.mkdir(parents=True)
        self.store = WikiStore(root=self.root, memory_dir=MEMORY_REL.as_posix())
        self.facts = ProposalStore(self.root / "facts.sqlite",
                                   embedder=Embedder(mode="hash"))
        for claim in ("Staging server stays on Fly.io",
                      "Every deploy needs a green CI run",
                      "The user prefers terse status updates"):
            p = self.facts.propose({"kind": "environment", "claim": claim,
                                    "evidence": "stated"}, user_text="stated",
                                   session_id="s")
            self.facts.accept(p["id"])

    def tearDown(self):
        self.tmp.cleanup()

    def _stub_scores(self, matrix):
        """Patch the pairwise scorer so chosen similarity values are used."""
        import hwiki.facts as facts_mod
        rows = list(matrix)

        def fake(_vec, others):
            # The last row has an empty comparison slice; still gets called.
            if not others:
                return []
            return (rows.pop(0) if rows else [])[:len(others)]

        original = facts_mod._cosine_batch
        facts_mod._cosine_batch = fake
        self.addCleanup(setattr, facts_mod, "_cosine_batch", original)

    def test_pair_inside_the_band_is_suggested(self):
        from hwiki.lint import suggest_review_pairs
        # rows[0] -> similarity to facts 1 and 2 ; rows[1] -> similarity to fact 2
        self._stub_scores([[0.75, 0.10], [0.05]])
        report = suggest_review_pairs(self.facts)
        self.assertEqual([f.check for f in report.findings], ["semantic.review_pair"])

    def test_near_duplicates_above_the_band_are_left_to_the_write_path(self):
        from hwiki.lint import suggest_review_pairs
        self._stub_scores([[0.95, 0.10], [0.05]])
        self.assertEqual(suggest_review_pairs(self.facts).findings, [])

    def test_unrelated_pairs_below_the_band_are_ignored(self):
        from hwiki.lint import suggest_review_pairs
        self._stub_scores([[0.20, 0.10], [0.05]])
        self.assertEqual(suggest_review_pairs(self.facts).findings, [])

    def test_suggestions_are_never_errors(self):
        from hwiki.lint import suggest_review_pairs, INFO
        self._stub_scores([[0.75, 0.10], [0.05]])
        report = suggest_review_pairs(self.facts)
        self.assertTrue(all(f.severity == INFO for f in report.findings))
        self.assertTrue(report.ok, "similarity must never fail lint")

    def test_suggestion_says_compare_not_resolve(self):
        from hwiki.lint import suggest_review_pairs
        self._stub_scores([[0.75, 0.10], [0.05]])
        for finding in suggest_review_pairs(self.facts).findings:
            self.assertIn("do not auto-resolve", finding.detail)

    def test_semantic_pass_writes_nothing(self):
        from hwiki.lint import suggest_review_pairs
        self._stub_scores([[0.75, 0.10], [0.05]])
        db = self.root / "facts.sqlite"
        before = db.read_bytes()
        suggest_review_pairs(self.facts)
        self.assertEqual(db.read_bytes(), before,
                         "a suggestion pass must not write, not even a status change")

    def test_hash_tier_yields_no_suggestions(self):
        """Documented tier limit: no mid-band pairs exist without semantic blur."""
        from hwiki.lint import suggest_review_pairs
        self.assertEqual(suggest_review_pairs(self.facts).findings, [])

    def test_single_fact_is_not_suggested(self):
        from hwiki.lint import suggest_review_pairs
        alone = ProposalStore(self.root / "alone.sqlite",
                              embedder=Embedder(mode="hash"))
        p = alone.propose({"kind": "fact", "claim": "Only one fact here",
                           "evidence": "stated"}, user_text="stated",
                          session_id="s")
        alone.accept(p["id"])
        self.assertEqual(suggest_review_pairs(alone).findings, [])
