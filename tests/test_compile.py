"""Tests for ``compile`` — merging an accepted fact into its owning page.

The four guarantees under test (see compile.py's docstring):
1. never lose a fact, 2. never silently contradict, 3. never clobber hand
edits, 4. never raise.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _hwiki_path  # noqa: F401,E402

from hwiki import compile as compile_mod  # noqa: E402
from hwiki.compile import (  # noqa: E402
    MERGE_SIMILARITY,
    SECTION_END,
    SECTION_START,
    append_compiled_section,
    choose_owner,
    compiled_section_block,
    existing_sections,
    merge_into_page,
)

PAGE = """---
title: Prefers dark mode
kind: preference
updated: 2026-11-01
---

# Prefers dark mode

> **Evidence:** user said so.

Hand-written paragraph the user added in Obsidian.
"""


class BlockFormatTests(unittest.TestCase):
    def test_block_has_delimiters_and_heading(self):
        block = compiled_section_block(
            "Prefers dark mode in editors too",
            evidence="user said so", kind="preference",
            accepted_iso="2026-11-08", proposal_id="abc123", peer="user")
        self.assertTrue(block.startswith(SECTION_START))
        self.assertTrue(block.rstrip().endswith(SECTION_END))
        self.assertIn("### Prefers dark mode in editors too", block)
        self.assertIn("**Evidence:** user said so", block)
        self.assertIn("proposal: abc123", block)

    def test_claim_is_one_line_and_bounded(self):
        block = compiled_section_block("a" * 500 + "\n\nsecond line")
        heading = [l for l in block.splitlines() if l.startswith("### ")][0]
        self.assertLessEqual(len(heading), 4 + 160)
        self.assertEqual(heading[4:].count("a"), heading.count("a"))

    def test_existing_sections_parses_multiple_blocks(self):
        text = append_compiled_section(
            append_compiled_section(PAGE, compiled_section_block("First merged")),
            compiled_section_block("Second merged"))
        self.assertEqual(existing_sections(text), ["First merged", "Second merged"])


class ChooseOwnerTests(unittest.TestCase):
    def _c(self, **kw):
        base = {"path": "/p/a.md", "claim": "x", "score": 0.9, "conflict": False,
                "is_self": False, "merged_count": 0}
        base.update(kw)
        return base

    def test_picks_highest_score(self):
        owner = choose_owner([self._c(path="/p/low.md", score=0.70),
                              self._c(path="/p/high.md", score=0.95)])
        self.assertEqual(owner["path"], "/p/high.md")

    def test_rejects_below_threshold(self):
        self.assertIsNone(choose_owner([self._c(score=MERGE_SIMILARITY - 0.01)]))

    def test_never_merges_a_conflict(self):
        """Guarantee 2: a contradiction keeps its own page."""
        self.assertIsNone(choose_owner([self._c(score=0.99, conflict=True)]))

    def test_rejects_self_and_pathless(self):
        self.assertIsNone(choose_owner([self._c(score=0.99, is_self=True)]))
        self.assertIsNone(choose_owner([self._c(score=0.99, path="")]))

    def test_tie_breaks_toward_fewer_merged_sections(self):
        owner = choose_owner([self._c(path="/p/busy.md", score=0.90, merged_count=5),
                              self._c(path="/p/fresh.md", score=0.90, merged_count=0)])
        self.assertEqual(owner["path"], "/p/fresh.md")

    def test_empty_candidates(self):
        self.assertIsNone(choose_owner([]))
        self.assertIsNone(choose_owner(None))


class MergeIntoPageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.page = self.dir / "prefers-dark-mode.md"
        self.page.write_text(PAGE, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_merge_preserves_hand_written_text(self):
        """Guarantee 3: nothing outside the delimiters is touched.

        The frontmatter legitimately gains ``compiled_sections`` and a fresh
        ``updated`` date, so the comparison is body-only — every byte the user
        (or hwiki's original writer) put there must survive.
        """
        from hwiki.compile import _FRONTMATTER_RE

        before_body = _FRONTMATTER_RE.sub("", self.page.read_text(encoding="utf-8"))
        result = merge_into_page(self.page, "Also prefers light themes at night")
        self.assertTrue(result.merged)
        after = self.page.read_text(encoding="utf-8")
        after_body = _FRONTMATTER_RE.sub("", after)
        self.assertTrue(after_body.startswith(before_body.rstrip("\n")),
                        "existing body must be preserved byte for byte")
        self.assertIn("Also prefers light themes at night", after)
        self.assertIn("Hand-written paragraph", after)
        # Only the two expected keys change.
        self.assertIn("compiled_sections: 1", after)
        self.assertIn("title: Prefers dark mode", after)
        self.assertIn("kind: preference", after)

    def test_merge_is_idempotent(self):
        merge_into_page(self.page, "Same claim twice")
        second = merge_into_page(self.page, "Same claim twice")
        self.assertFalse(second.merged)
        self.assertEqual(second.reason, "already compiled")
        self.assertEqual(existing_sections(self.page.read_text()), ["Same claim twice"])

    def test_already_argument_prevents_duplicates(self):
        result = merge_into_page(self.page, "Claim carried elsewhere",
                                 already=["claim carried elsewhere"])
        self.assertFalse(result.merged)

    def test_frontmatter_tracks_compiled_count(self):
        merge_into_page(self.page, "First")
        merge_into_page(self.page, "Second")
        text = self.page.read_text(encoding="utf-8")
        self.assertIn("compiled_sections: 2", text)
        self.assertIn("updated:", text)
        self.assertEqual(len(existing_sections(text)), 2)

    def test_page_without_frontmatter_is_not_restructured(self):
        bare = self.dir / "bare.md"
        bare.write_text("# Bare\n\nJust prose.\n", encoding="utf-8")
        merge_into_page(bare, "Merged claim")
        text = bare.read_text(encoding="utf-8")
        self.assertFalse(text.startswith("---"))
        self.assertIn("Merged claim", text)

    def test_missing_page_does_not_raise(self):
        result = merge_into_page(self.dir / "nope.md", "claim")
        self.assertFalse(result.merged)
        self.assertIn("unreadable", result.reason)

    def test_directory_target_does_not_raise(self):
        result = merge_into_page(self.dir, "claim")
        self.assertFalse(result.merged)
        self.assertTrue(result.reason)

    def test_no_temp_file_left_behind(self):
        merge_into_page(self.page, "Claim")
        self.assertEqual([p.name for p in self.dir.glob("*.tmp")], [])

    def test_unicode_claim(self):
        result = merge_into_page(self.page, "Préfère le thème sombre ☕")
        self.assertTrue(result.merged)
        self.assertIn("Préfère le thème sombre ☕",
                      self.page.read_text(encoding="utf-8"))



class MergeRefusalTests(unittest.TestCase):
    """Compile must not add to a page that is no longer current, or to a derived file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.mem = self.root / "mem"
        self.mem.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _page(self, name, frontmatter, body="# Page\n\nclaim text\n"):
        path = self.mem / name
        path.write_text(f"---\n{frontmatter}\n---\n\n{body}", encoding="utf-8")
        return path

    def test_a_superseded_page_is_not_merged_into(self):
        page = self._page("old.md", "title: Old\ntype: memory\nstatus: superseded")
        before = page.read_text(encoding="utf-8")
        result = compile_mod.merge_into_page(page, "a newer claim", proposal_id="p1")
        self.assertFalse(result.merged)
        self.assertIn("superseded", result.reason)
        self.assertEqual(page.read_text(encoding="utf-8"), before)

    def test_the_maintenance_report_is_not_merged_into(self):
        page = self._page("maintenance.md",
                          "title: Memory maintenance\ntype: maintenance\nstatus: generated")
        before = page.read_text(encoding="utf-8")
        result = compile_mod.merge_into_page(page, "a claim", proposal_id="p1")
        self.assertFalse(result.merged)
        self.assertIn("generated", result.reason)
        self.assertEqual(page.read_text(encoding="utf-8"), before)

    def test_a_filed_answer_is_not_merged_into(self):
        answers = self.mem / "answers"
        answers.mkdir()
        page = answers / "answer.md"
        page.write_text("---\ntitle: Answer\n---\n\n# Answer\n\ntext\n", encoding="utf-8")
        result = compile_mod.merge_into_page(page, "a claim", proposal_id="p1")
        self.assertFalse(result.merged)
        self.assertIn("filed answer", result.reason)

    def test_an_ordinary_fact_page_still_merges(self):
        page = self._page("fact.md", "title: Fact\ntype: memory\nstatus: auto")
        result = compile_mod.merge_into_page(page, "a second claim", proposal_id="p1")
        self.assertTrue(result.merged, result.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
