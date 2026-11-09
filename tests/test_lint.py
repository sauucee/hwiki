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
from hwiki import lint  # noqa: E402
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

    def test_a_folder_qualified_catalog_entry_counts_as_navigation(self):
        """A memory note whose stem is duplicated elsewhere is still catalogued.

        The catalog names the folder for duplicate stems, so counting a catalog
        entry as an inbound route has to resolve the same way.
        """
        curated = self.root / "Hermes" / "fact.md"
        curated.parent.mkdir(parents=True, exist_ok=True)
        curated.write_text("# fact\n\nA curated page that shares the stem.\n",
                           encoding="utf-8")
        (self.root / "index.md").write_text(
            "# Index\n\n- [[Hermes/fact|fact]]\n- [[Hermes/Decisions|Decisions]]\n"
            "- [[Projects/Alpha/Decisions|Decisions]]\n"
            f"- [[{MEMORY_REL.as_posix()}/fact|fact]]\n", encoding="utf-8")
        report = lint_memory(self.store)
        orphans = [f.line for f in report.findings if f.check == "link.orphan"]
        self.assertEqual(orphans, [])

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

    def curated_page(self, rel="Projects/Alpha/Current State.md",
                     body="The deploy is on hold.\n"):
        """A hand-written page outside the memory zone."""
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path

    def generated_index(self):
        """Write the index exactly as the catalog renders it (one ``[[stem]]``/page)."""
        from hwiki.catalog import build_index
        text = build_index(self.store, memory_dir=MEMORY_REL.as_posix())
        (self.root / "index.md").write_text(text, encoding="utf-8")
        return text

    def test_catalog_covers_the_whole_vault_not_only_the_memory_zone(self):
        """The catalog lists every page, so no curated page is 'stale'.

        Judging index.md against the memory zone alone reported each curated page as
        gone on every run while the page sat on disk — the false positive that trains
        a reader to ignore lint.
        """
        self.fact_page("User prefers terse status updates")
        self.curated_page("Projects/Alpha/Current State.md",
                          "---\ntitle: Current State\n---\n\nThe deploy is on hold.\n")
        self.curated_page("Projects/Alpha/Tasks.md", "# Tasks\n\nNothing yet.\n")
        self.generated_index()

        report = lint_memory(self.store)
        catalog_checks = [(f.check, f.page, f.detail) for f in report.findings
                          if f.check.startswith("catalog.")]
        self.assertEqual(catalog_checks, [], catalog_checks)

    def test_catalog_still_flags_a_page_that_is_gone(self):
        """Real rot survives the fix: one missing page, one finding naming it."""
        self.fact_page("User prefers terse status updates")
        self.curated_page()
        index = self.generated_index()
        (self.root / "index.md").write_text(
            index + "\n- [[ghost-page]]\n", encoding="utf-8")

        report = lint_memory(self.store)
        stale = [f for f in report.findings if f.check == "catalog.stale"]
        self.assertEqual(len(stale), 1, [f.line for f in report.findings])
        self.assertIn("ghost-page", stale[0].detail)

    def test_catalog_missing_a_curated_page_is_flagged(self):
        """A page that exists but is not in the catalog is rot too."""
        page = self.fact_page("User prefers terse status updates")
        self.curated_page()
        (self.root / "index.md").write_text(
            f"# Index\n\n- [[{page.stem}]]\n", encoding="utf-8")

        report = lint_memory(self.store)
        missing = [f for f in report.findings if f.check == "catalog.entry_missing"]
        self.assertEqual(len(missing), 1, [f.line for f in report.findings])
        self.assertIn("current state", missing[0].detail.lower())
        self.assertEqual([f for f in report.findings if f.check == "catalog.stale"], [])

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



class OrphanTests(LintTests):
    """Nothing navigates to it: no inbound link, no catalog entry."""

    def _note(self, name, body, *, generated=False):
        meta = ("---\ntitle: Note\nstatus: generated\ntype: maintenance\n---\n\n"
                if generated else "---\ntitle: Note\n---\n\n")
        path = self.mem / f"{name}.md"
        path.write_text(meta + body, encoding="utf-8")
        return path

    def test_an_unlinked_page_is_an_orphan(self):
        self._note("lonely", "Nobody points here.\n")
        report = lint_memory(self.store)
        self.assertIn("link.orphan", [c for c, _s in self.checks(report)])

    def test_a_catalog_entry_counts_as_navigation(self):
        page = self._note("listed", "Catalogued, so it is reachable.\n")
        (self.root / "index.md").write_text(
            f"# Index\n\n- [[{page.stem}]] — a note\n", encoding="utf-8")
        report = lint_memory(self.store)
        self.assertNotIn("link.orphan", [c for c, _s in self.checks(report)])

    def test_an_inbound_link_clears_it(self):
        target = self._note("target", "Linked from a fact page.\n")
        self.fact_page(f"See [[{target.stem}]] for context")
        report = lint_memory(self.store)
        orphans = [f for f in report.findings if f.check == "link.orphan"]
        self.assertNotIn(str(target.relative_to(self.root)), [f.page for f in orphans])

    def test_generated_artifacts_are_not_orphans(self):
        self._note("maintenance", "Generated every tick.\n", generated=True)
        report = lint_memory(self.store)
        orphans = [f for f in report.findings if f.check == "link.orphan"]
        self.assertEqual([o.page for o in orphans if "maintenance" in o.page], [])

    def test_a_curated_page_linking_a_note_counts_as_inbound(self):
        """Navigation crosses zones: only memory-zone pages were read as sources.

        A hand-written page under ``Hermes/`` that links a memory note is a real
        inbound route; the note was still reported as an orphan because the inbound
        pass read only the memory zone it was reporting on.
        """
        target = self._note("target", "Linked from a curated page.\n")
        curated = self.root / "Hermes" / "Home.md"
        curated.parent.mkdir(parents=True, exist_ok=True)
        curated.write_text(f"# Home\n\nSee [[{target.stem}]].\n", encoding="utf-8")
        report = lint_memory(self.store)
        orphans = [f for f in report.findings if f.check == "link.orphan"]
        self.assertNotIn(str(target.relative_to(self.root)), [f.page for f in orphans],
                         [f.line for f in orphans])

    def test_it_is_info_not_a_warning(self):
        self._note("quiet", "Nothing links here.\n")
        report = lint_memory(self.store)
        orphans = [f for f in report.findings if f.check == "link.orphan"]
        self.assertTrue(orphans)
        self.assertTrue(all(f.severity == "info" for f in orphans))

    def test_a_self_link_does_not_count_as_inbound(self):
        page = self._note("navel", "Gazing at [[navel]].\n")
        report = lint_memory(self.store)
        self.assertIn(str(page.stem), " ".join(
            f.page for f in report.findings if f.check == "link.orphan"))


class HandwrittenLinkTests(LintTests):
    """A hand-written memory note is where a broken link most often lands.

    The schema checks `continue` on a page without the extractor marker; the link
    check used to sit after that `continue`, so it only ever ran on provider-written
    fact pages and every broken link in a hand-written note was invisible.
    """

    def test_a_handwritten_note_reports_its_dangling_link(self):
        (self.mem / "note.md").write_text(
            "---\ntitle: Note\n---\n\n# Note\n\nSee [[nowhere-page]].\n",
            encoding="utf-8")
        report = lint_memory(self.store)
        lines = [f.line for f in report.findings if f.check == "link.dangling"]
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("nowhere-page", lines[0])

    def test_a_handwritten_overview_page_gets_no_schema_findings(self):
        (self.mem / "overview.md").write_text(
            "# Overview\n\nProse without frontmatter.\n", encoding="utf-8")
        report = lint_memory(self.store)
        checks = [c for c, _s in self.checks(report)]
        self.assertNotIn("frontmatter.incomplete", checks)
        self.assertNotIn("claim.missing", checks)


class UnlinkableNameTests(LintTests):
    """A page whose path a wikilink cannot carry must not be linted as missing.

    The catalog now lists such pages as plain paths, so treating them as uncatalogued
    (or as orphans) would be a warning the user can never clear.
    """

    def _hostile(self):
        # a memory page keeps the lint pass from returning early on an empty
        # memory zone, and gives the catalog a second entry to compare against
        (self.mem / "note.md").write_text("---\ntitle: Note\n---\n\n# Note\n",
                                          encoding="utf-8")
        (self.root / "a").mkdir(exist_ok=True)
        path = self.root / "a" / "we|ird.md"
        path.write_text("# Weird\n", encoding="utf-8")
        return path

    def test_it_is_not_demanded_as_a_catalog_link(self):
        from hwiki.catalog import build_index
        self._hostile()
        (self.root / "index.md").write_text(build_index(self.store), encoding="utf-8")
        report = lint_memory(self.store)
        checks = [c for c, _s in self.checks(report)]
        self.assertNotIn("catalog.entry_missing", checks)
        self.assertNotIn("catalog.stale", checks)
        self.assertNotIn("link.orphan", checks)

    def test_a_linkable_page_is_still_demanded(self):
        from hwiki.catalog import build_index
        self._hostile()
        (self.root / "a" / "plain.md").write_text("# Plain\n", encoding="utf-8")
        (self.root / "index.md").write_text(build_index(self.store), encoding="utf-8")
        report = lint_memory(self.store)
        self.assertNotIn("catalog.entry_missing", [c for c, _s in self.checks(report)])


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


class PathQualifiedLinkTests(unittest.TestCase):
    """Obsidian path-qualified links (``[[Folder/Page]]``) are first-class links.

    The catalog emits them once duplicate stems exist, and a person writing in
    Obsidian produces them by hand. Lint must resolve them against the page it
    names — not call them stale, and not call the page they point at missing.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vault"
        self.mem = self.root / MEMORY_REL
        self.mem.mkdir(parents=True)
        for rel in ("Hermes/Decisions.md", "Projects/Alpha/Decisions.md"):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# Decisions\n\n{rel}\n", encoding="utf-8")
        (self.mem / "fact.md").write_text(
            "---\ntitle: Prefers dark mode\nkind: preference\nstatus: accepted\n---\n\n"
            "# Prefers dark mode\n", encoding="utf-8")
        self.store = WikiStore(root=self.root, memory_dir=MEMORY_REL.as_posix())

    def tearDown(self):
        self.tmp.cleanup()

    def _catalog_findings(self, text):
        (self.root / "index.md").write_text(text, encoding="utf-8")
        report = lint_memory(self.store)
        return [f.line for f in report.findings if f.check.startswith("catalog.")]

    def test_a_path_qualified_catalog_is_not_stale(self):
        findings = self._catalog_findings(
            "# Index\n\n"
            "- [[Hermes/Decisions|Decisions]]\n"
            "- [[Projects/Alpha/Decisions|Decisions]]\n"
            "- [[fact]]\n")
        self.assertEqual(findings, [])

    def test_a_path_qualified_link_in_a_page_is_not_dangling(self):
        (self.mem / "note.md").write_text(
            "---\ntitle: Note\nextractor: hwiki-extractor\nkind: note\n"
            "status: accepted\n---\n\n# Note\n\nSee [[Projects/Alpha/Decisions]].\n",
            encoding="utf-8")
        report = lint_memory(self.store)
        self.assertNotIn(("link.dangling", ERROR), self.checks(report))

    def test_a_bare_shared_stem_covers_only_the_page_it_resolves_to(self):
        """Coverage means what ``resolve_page`` means, even for a duplicate stem.

        A bare ``[[Decisions]]`` has always resolved to the first match, so it really
        does catalogue that page; claiming it covered *both* overstated coverage, and
        claiming neither understated it. The twin it does not name is reported missing.
        """
        findings = self._catalog_findings(
            "# Index\n\n- [[Decisions]]\n- [[fact]]\n")
        resolved = self.store.resolve_page("Decisions")
        self.assertIsNotNone(resolved)
        missing = [f for f in findings if "catalog.entry_missing" in f]
        self.assertEqual(len(missing), 1, findings)
        self.assertNotIn(resolved.path, " ".join(missing))
        self.assertEqual([f for f in findings if "catalog.ambiguous" in f], [])

    def test_one_missing_twin_is_still_reported(self):
        findings = self._catalog_findings(
            "# Index\n\n- [[Hermes/Decisions]]\n- [[fact]]\n")
        self.assertTrue(any("projects/alpha/decisions" in f.lower() for f in findings),
                        findings)

    def test_a_folder_qualified_catalog_entry_counts_as_navigation(self):
        """A memory note whose stem is duplicated elsewhere is still catalogued.

        The catalog names the folder for duplicate stems, so counting a catalog
        entry as an inbound route has to resolve the same way.
        """
        curated = self.root / "Hermes" / "fact.md"
        curated.parent.mkdir(parents=True, exist_ok=True)
        curated.write_text("# fact\n\nA curated page that shares the stem.\n",
                           encoding="utf-8")
        (self.root / "index.md").write_text(
            "# Index\n\n- [[Hermes/fact|fact]]\n- [[Hermes/Decisions|Decisions]]\n"
            "- [[Projects/Alpha/Decisions|Decisions]]\n"
            f"- [[{MEMORY_REL.as_posix()}/fact|fact]]\n", encoding="utf-8")
        report = lint_memory(self.store)
        orphans = [f.line for f in report.findings if f.check == "link.orphan"]
        self.assertEqual(orphans, [])

    def checks(self, report):
        return [(f.check, f.severity) for f in report.findings]


class LinkIdentityTests(unittest.TestCase):
    """Catalog and orphan checks must compare page identity, not file stems.

    A path-qualified link names one page; collapsing it back to its stem makes two
    pages look equally covered, which hides a page nobody navigates to and claims
    coverage for a page that is not in the catalog. A bare link to a stem that two
    pages share likewise cannot say which page it means, so it must not count as an
    entry for either one.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vault"
        mem = self.root / MEMORY_REL
        (mem / "a").mkdir(parents=True)
        (mem / "b").mkdir(parents=True)
        (mem / "a" / "note.md").write_text("---\ntitle: Note A\n---\n\n# Note A\n",
                                           encoding="utf-8")
        (mem / "b" / "note.md").write_text("---\ntitle: Note B\n---\n\n# Note B\n",
                                           encoding="utf-8")
        (mem / "fact.md").write_text(
            "---\ntitle: Prefers dark mode\nkind: preference\nstatus: accepted\n---\n\n"
            "# Prefers dark mode\n", encoding="utf-8")
        for rel in ("Hermes/Decisions.md", "Projects/Alpha/Decisions.md",
                    "Docs/Plan.md", "docs/plan.md"):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {Path(rel).stem}\n\nbody\n", encoding="utf-8")
        self.store = WikiStore(root=self.root, memory_dir=MEMORY_REL.as_posix())
        self.mem_rel = MEMORY_REL.as_posix()

    def tearDown(self):
        self.tmp.cleanup()

    def findings(self, text):
        (self.root / "index.md").write_text(text, encoding="utf-8")
        report = lint_memory(self.store)
        return report.findings

    def catalogue_all(self):
        return (
            f"# Index\n\n"
            f"- [[{self.mem_rel}/a/note|note]]\n"
            f"- [[{self.mem_rel}/b/note|note]]\n"
            f"- [[{self.mem_rel}/fact|fact]]\n"
            f"- [[Hermes/Decisions|Decisions]]\n"
            f"- [[Projects/Alpha/Decisions|Decisions]]\n"
            f"- [[Docs/Plan]]\n"
            f"- [[docs/plan]]\n")

    def test_catalogue_of_every_page_reports_nothing(self):
        findings = self.findings(self.catalogue_all())
        self.assertEqual(findings, [], [f.line for f in findings])

    def test_a_missing_orphan_twin_is_not_hidden_by_a_path_qualified_entry(self):
        text = self.catalogue_all().replace(
            f"- [[{self.mem_rel}/b/note|note]]\n", "")
        findings = self.findings(text)
        orphans = [f.page for f in findings if f.check == "link.orphan"]
        self.assertIn(f"{self.mem_rel}/b/note.md", orphans)

    def test_a_bare_shared_stem_covers_the_page_the_resolver_picks(self):
        findings = self.findings(f"# Index\n\n- [[Decisions]]\n- [[{self.mem_rel}/fact|fact]]\n")
        checks = [f.check for f in findings]
        resolved = self.store.resolve_page("Decisions")
        missing = [f.detail.lower() for f in findings if f.check == "catalog.entry_missing"]
        twin = ("hermes/decisions.md" if resolved.path.lower().endswith("projects/alpha/decisions.md")
                else "projects/alpha/decisions.md")
        self.assertTrue(any(twin in m for m in missing), missing)
        self.assertFalse(any(resolved.path.lower() in m for m in missing), missing)
        self.assertNotIn("catalog.ambiguous", checks)
        self.assertNotIn("catalog.stale", checks)

    def test_a_case_colliding_entry_covers_only_the_page_it_names(self):
        text = self.catalogue_all().replace("- [[docs/plan]]\n", "")
        findings = self.findings(text)
        self.assertEqual([f.line for f in findings if f.check == "catalog.ambiguous"], [])
        missing = [f.detail.lower() for f in findings if f.check == "catalog.entry_missing"]
        self.assertEqual(len(missing), 1, [f.line for f in findings])
        self.assertIn("docs/plan.md", missing[0])



class ArtifactScopeTests(LintTests):
    """Generated artifacts are not pages the catalog must carry.

    ``maintenance.md`` is written after the artifacts job, so the index regenerated in
    the same tick can never list it. Requiring it made every later tick rewrite the
    index for no change.
    """

    def test_the_maintenance_report_is_not_demanded_in_the_index(self):
        from hwiki import maintain
        from hwiki.catalog import build_index
        self.write = (self.mem / maintain.MAINTENANCE_FILENAME)
        self.write.write_text("---\ntitle: Maintenance\ntype: maintenance\n"
                              "status: generated\n---\n\n# Maintenance\n", encoding="utf-8")
        (self.mem / "fact.md").write_text(
            "---\ntitle: Fact\n---\n\n# Fact\n", encoding="utf-8")
        (self.root / "index.md").write_text(build_index(self.store), encoding="utf-8")
        report = lint_memory(self.store)
        missing = [f.line for f in report.findings if f.check == "catalog.entry_missing"]
        self.assertEqual(missing, [])

    def test_artifact_names_agree_with_the_generated_files(self):
        from hwiki import catalog, maintain
        names = lint._catalog_artifact_names()
        self.assertIn(maintain.MAINTENANCE_FILENAME.lower(), names)
        self.assertIn(catalog.INDEX_FILENAME.lower(), names)
        self.assertIn(catalog.LOG_FILENAME.lower(), names)
        self.assertEqual(maintain.MAINTENANCE_FILENAME, catalog.MAINTENANCE_FILENAME)
        zone = "Hermes/Memory/Auto"
        self.assertTrue(catalog.is_catalog_artifact(f"{zone}/{maintain.MAINTENANCE_FILENAME}",
                                                    memory_dir=zone))
        self.assertFalse(catalog.is_catalog_artifact("Projects/X/maintenance.md",
                                                     memory_dir=zone))
        self.assertFalse(catalog.is_catalog_artifact("Projects/X/index.md",
                                                     memory_dir=zone))
    def test_only_the_generated_report_path_is_an_artifact(self):
        """A curated page that happens to be called maintenance.md is a page.

        Matching the basename anywhere meant such a page was never demanded in the
        catalog — it could be missing from the index forever without a warning.
        """
        from hwiki import maintain
        from hwiki.catalog import build_index
        (self.root / "Projects" / "X").mkdir(parents=True)
        (self.root / "Projects" / "X" / "maintenance.md").write_text(
            "# Maintenance window\n\nEvery Sunday.\n", encoding="utf-8")
        (self.mem / maintain.MAINTENANCE_FILENAME).write_text(
            "---\ntitle: Report\ntype: maintenance\nstatus: generated\n---\n\n# R\n",
            encoding="utf-8")
        (self.mem / "fact.md").write_text("---\ntitle: F\n---\n\n# F\n",
                                          encoding="utf-8")
        generated = build_index(self.store)
        # folder-qualified: the bare stem also names the generated report page
        self.assertIn("[[Projects/X/maintenance|maintenance]]", generated)
        # simulate an index that forgot the curated page
        (self.root / "index.md").write_text(
            "\n".join(line for line in generated.splitlines()
                      if "[[Projects/X/maintenance" not in line) + "\n",
            encoding="utf-8")
        report = lint_memory(self.store)
        missing = [f.line for f in report.findings if f.check == "catalog.entry_missing"]
        self.assertTrue(any("Projects/X/maintenance.md" in line for line in missing), missing)
        self.assertFalse(any(f"Auto/{maintain.MAINTENANCE_FILENAME}" in line
                             for line in missing), missing)

    def test_an_ambiguous_catalog_entry_covers_neither_page(self):
        """Lint credited every candidate; the resolver refuses to pick one.

        ``[[DOCS/PLAN]]`` matched two pages case-insensitively, so both counted as
        listed although the reader can open neither. Coverage must mean what
        ``resolve_page`` means.
        """
        from hwiki.catalog import build_index
        (self.root / "Docs").mkdir()
        (self.root / "docs").mkdir(exist_ok=True)
        (self.root / "Docs" / "Plan.md").write_text("# upper\n", encoding="utf-8")
        (self.root / "docs" / "plan.md").write_text("# lower\n", encoding="utf-8")
        (self.mem / "note.md").write_text("---\ntitle: Note\n---\n\n# Note\n",
                                          encoding="utf-8")
        index = (self.root / "index.md")
        index.write_text(
            "---\ntitle: Wiki Index\nsource: hwiki-catalog\n---\n\n"
            "*2 curated page(s) across 2 section(s), 1 memory note(s) — 3 page(s) catalogued.*\n\n"
            "## Docs\n\n- [[docs/plan]]\n\n## Docs\n\n- [[DOCS/PLAN]]\n\n"
            "## Note\n\n- [[note]]\n", encoding="utf-8")
        report = lint_memory(self.store)
        missing = [f.line for f in report.findings if f.check == "catalog.entry_missing"]
        self.assertTrue(any("Docs/Plan.md" in line for line in missing), missing)
        self.assertFalse(any("docs/plan.md" in line for line in missing), missing)
        # and the catalog the generator writes is always fully resolved
        index.write_text(build_index(self.store), encoding="utf-8")
        report = lint_memory(self.store)
        self.assertEqual([f.line for f in report.findings
                          if f.check == "catalog.entry_missing"], [])

    def test_candidates_never_credit_a_page_the_resolver_refuses(self):
        """Parity lock: catalog candidates and ``resolve_page`` must agree.

        The catalog used to credit a case-different path (``readme.md`` for
        ``README.md``, ``a/TARGET.MD``) that the resolver returns ``None`` for, and to
        treat a unique-stem ``.md`` miss as a case-folded hit. Every target is checked
        against the store: one candidate means that page, several mean the resolver
        refuses to choose, none means nothing matches.
        """
        from hwiki.lint import _Identity

        for rel in ("README.md", "docs/plan.md", "Docs/Plan.md",
                    "a/target.md", "b/target.md", "Hermes/Decisions.md"):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {rel}\n", encoding="utf-8")
        pages = list(self.store.iter_pages())
        identity = _Identity()
        for page in pages:
            identity.add(page)
        identity.resolver = lambda ref: self.store._resolve_in(ref, pages)

        targets = ["readme.md", "README.md", "a/TARGET.MD", "a/target.md",
                   "b/target.MD", "DOCS/PLAN", "docs/plan", "Docs/Plan",
                   "decisions", "Hermes/DECISIONS.MD", "target", "plan",
                   "a/target", "nope.md", "nope"]
        for target in targets:
            candidates = identity.candidates(target)
            resolved = identity.resolve(target)
            with self.subTest(target=target):
                if resolved is not None:
                    # bare slug/title tiers keep first-match, so the resolver's pick is
                    # one of the candidates; the folded tier never picks between two
                    self.assertIn(resolved, candidates)
                elif len(candidates) == 1:
                    self.fail(f"{target!r}: credited {candidates} but the resolver refused")

    def test_slugified_filename_twins_resolve_like_the_store(self):
        """``a/Foo Bar.md`` and ``b/foo-bar.md`` share a slugified stem, not a stem.

        The catalog must not emit a bare link whose meaning depends on directory
        iteration order, and lint's candidate set must be exactly the page the store
        resolves — otherwise the two disagree about which page an entry covers.
        """
        from hwiki.catalog import build_index
        from hwiki.lint import _Identity
        (self.root / "a").mkdir()
        (self.root / "b").mkdir()
        (self.root / "a" / "Foo Bar.md").write_text("# first\n", encoding="utf-8")
        (self.root / "b" / "foo-bar.md").write_text("# second\n", encoding="utf-8")
        pages = list(self.store.iter_pages())
        identity = _Identity()
        for page in pages:
            identity.add(page)
        identity.resolver = lambda ref: self.store._resolve_in(ref, pages)
        for target in ("foo-bar", "Foo Bar", "a/Foo Bar", "b/foo-bar"):
            resolved = identity.resolve(target)
            with self.subTest(target=target):
                self.assertEqual(identity.candidates(target),
                                 {resolved} if resolved else set())
        index = build_index(self.store)
        # both pages must be folder-qualified: their slugified stems collide, and the
        # store picks the first glob match, which is not a stable answer
        self.assertIn("[[a/Foo Bar|Foo Bar]]", index)
        self.assertIn("[[b/foo-bar|foo-bar]]", index)
        self.assertNotIn("- [[foo-bar]]", index)
        for line in index.splitlines():
            if not line.startswith("- [["):
                continue
            target = line.split("[[", 1)[1].split("]]", 1)[0].split("|", 1)[0]
            page = self.store.resolve_page(target)
            self.assertIsNotNone(page, target)
            if "Foo Bar" in line:
                self.assertEqual(page.path, "a/Foo Bar.md")
            else:
                self.assertEqual(page.path, "b/foo-bar.md")


if __name__ == "__main__":
    unittest.main(verbosity=2)
