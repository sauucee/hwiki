"""Tests for the LLM-wiki navigation artifacts (``index.md`` + ``log.md``).

The catalogs are DERIVED: they must never raise into a write path, must never
be written into the memory zone, and must never outrank curated pages in
recall. Each of those rules has a test below.
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.catalog import (  # noqa: E402
    INDEX_FILENAME,
    LOG_FILENAME,
    append_log,
    build_index,
    read_log_tail,
    summarize_page,
    write_index,
)
from hwiki.store import WikiStore  # noqa: E402


def _make_wiki(root: Path) -> WikiStore:
    (root / "concepts").mkdir(parents=True, exist_ok=True)
    (root / "entities").mkdir(parents=True, exist_ok=True)
    (root / "memory" / "auto").mkdir(parents=True, exist_ok=True)
    (root / "concepts" / "retrieval.md").write_text(
        "---\ntitle: Retrieval\nupdated: 2026-11-01\n---\n\n"
        "# Retrieval\n\n"
        "Hybrid search blends lexical and vector recall. It beats either alone.\n\n"
        "Second paragraph must not appear.\n",
        encoding="utf-8",
    )
    (root / "entities" / "honcho.md").write_text(
        "---\ntitle: Honcho\n---\n\n# Honcho\n\n"
        "A memory service that derives facts from conversation transcripts.\n",
        encoding="utf-8",
    )
    (root / "memory" / "auto" / "fact-one.md").write_text(
        "---\ntitle: Prefers dark mode\nkind: preference\nupdated: 2026-11-08\n---\n\n"
        "# Prefers dark mode\n\n> **Evidence:** user said so.\n\n"
        "The user prefers dark mode in editors.\n",
        encoding="utf-8",
    )
    return WikiStore(root, include=["**/*.md"], memory_dir="memory/auto")


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = _make_wiki(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_prose_sentence_only(self):
        page = self.store.load_rel("concepts/retrieval.md")
        summary = summarize_page(page)
        self.assertIn("Hybrid search blends", summary)
        self.assertNotIn("Second paragraph", summary)

    def test_frontmatter_summary_wins(self):
        page = self.store.load_rel("entities/honcho.md")
        page.frontmatter["summary"] = "  The derived-memory reference.  "
        self.assertEqual(summarize_page(page), "The derived-memory reference.")

    def test_description_fallback(self):
        page = self.store.load_rel("entities/honcho.md")
        page.frontmatter["description"] = "Desc text here."
        self.assertEqual(summarize_page(page), "Desc text here.")

    def test_markup_stripped(self):
        page = self.store.load_rel("entities/honcho.md")
        page.body = (
            "# T\n\n"
            "See [[honcho|Honcho]] and [docs](https://x.test) for `detail` "
            "and <b>bold</b> text.\n"
        )
        summary = summarize_page(page)
        for raw in ("[[", "]]", "](", "<b>", "`"):
            self.assertNotIn(raw, summary)
        self.assertIn("docs", summary)

    def test_skips_fenced_code_and_headings(self):
        page = self.store.load_rel("entities/honcho.md")
        page.body = "# Heading\n\n```\ncode line\n```\n\nReal prose sentence here.\n"
        self.assertEqual(summarize_page(page), "Real prose sentence here.")

    def test_empty_body_is_empty_summary(self):
        page = self.store.load_rel("entities/honcho.md")
        page.body = ""
        page.frontmatter.pop("summary", None)
        self.assertEqual(summarize_page(page), "")

    def test_generated_callout_header_is_not_the_summary(self):
        """Real memory notes open with a two-line generated callout.

        Regression: the summary used to be the SECOND callout line
        ("conclusions into the curated wiki ..."), which describes the
        generator rather than the page.
        """
        page = self.store.load_rel("memory/auto/fact-one.md")
        page.body = (
            "# Hermes Agent\n\n"
            "> Auto-maintained by the `hwiki` memory provider. Promote durable\n"
            "> conclusions into the curated wiki rather than editing in place.\n\n"
            "## Log\n\n"
            "- **2026-11-08** — The Hermes Agent is an open-source AI agent "
            "framework. #hermes #cli _(source: docs)_\n"
        )
        summary = summarize_page(page)
        self.assertIn("open-source AI agent framework", summary)
        self.assertNotIn("curated wiki", summary)
        self.assertNotIn("Auto-maintained", summary)
        self.assertNotIn("#hermes", summary)
        self.assertNotIn("source:", summary)

    def test_date_and_hashtail_stripped_from_bullets(self):
        page = self.store.load_rel("memory/auto/fact-one.md")
        page.body = (
            "# Prefs\n\n## Log\n\n"
            "- **2026-11-08** — Prefers dark mode in editors. #pref _(source: chat)_\n"
        )
        self.assertEqual(summarize_page(page), "Prefers dark mode in editors.")

    def test_deterministic(self):
        page = self.store.load_rel("concepts/retrieval.md")
        self.assertEqual(summarize_page(page), summarize_page(page))

    def test_truncation_is_bounded(self):
        page = self.store.load_rel("entities/honcho.md")
        page.body = "word " * 200
        self.assertLessEqual(len(summarize_page(page, max_chars=80)), 80)


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = _make_wiki(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_groups_by_folder(self):
        text = build_index(self.store, memory_dir="memory/auto", today="2026-11-08")
        self.assertIn("## concepts", text)
        self.assertIn("## entities", text)
        self.assertIn("[[retrieval]]", text)
        self.assertIn("[[honcho]]", text)
        self.assertIn("updated: 2026-11-08", text)
        self.assertIn("type: index", text)

    def test_memory_zone_listed_separately(self):
        text = build_index(self.store, memory_dir="memory/auto", today="2026-11-08")
        self.assertIn("## Memory notes", text)
        self.assertIn("[[fact-one]]", text)
        # The memory note must appear only under "Memory notes", not as a
        # folder section of its own.
        head, _, tail = text.partition("## Memory notes")
        self.assertNotIn("[[fact-one]]", head)
        self.assertIn("[[fact-one]]", tail)

    def test_index_and_log_list_themselves_not(self):
        (self.root / INDEX_FILENAME).write_text("# old index\n", encoding="utf-8")
        (self.root / LOG_FILENAME).write_text("# old log\n", encoding="utf-8")
        text = build_index(self.store, memory_dir="memory/auto", today="2026-11-08")
        self.assertNotIn(f"[[{Path(INDEX_FILENAME).stem}]]", text)
        self.assertNotIn(f"[[{Path(LOG_FILENAME).stem}]]", text)

    def test_each_page_listed_once(self):
        text = build_index(self.store, memory_dir="memory/auto", today="2026-11-08")
        self.assertEqual(text.count("[[retrieval]]"), 1)
        self.assertEqual(text.count("[[honcho]]"), 1)

    def test_write_index_roundtrip_and_overwrite(self):
        path = write_index(self.store, memory_dir="memory/auto")
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        first = path.read_text(encoding="utf-8")
        second_dir = write_index(self.store, memory_dir="memory/auto")
        self.assertEqual(second_dir, path)
        self.assertEqual(path.read_text(encoding="utf-8"), first)
        # No temp files left behind.
        self.assertEqual([p.name for p in self.root.glob(".hwiki-*")], [])

    def test_write_index_never_raises_on_unwritable_root(self):
        target = self.root / "sub" / INDEX_FILENAME
        # Point the target into a path that cannot be created (parent is a file).
        blocker = self.root / "blocker"
        blocker.write_text("x", encoding="utf-8")
        self.assertIsNone(write_index(self.store, path=blocker / INDEX_FILENAME,
                                      memory_dir="memory/auto"))

    def test_index_not_written_into_memory_zone(self):
        write_index(self.store, memory_dir="memory/auto")
        self.assertFalse((self.root / "memory" / "auto" / INDEX_FILENAME).exists())
        self.assertTrue((self.root / INDEX_FILENAME).exists())


class LogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = _make_wiki(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_header_then_append(self):
        when = datetime(2026, 11, 8, tzinfo=timezone.utc)
        path = append_log(self.store, "write", "Prefers dark mode", when=when)
        self.assertIsNotNone(path)
        text = path.read_text(encoding="utf-8")
        self.assertIn("type: log", text)
        self.assertIn("## [2026-11-08] write | Prefers dark mode", text)

        append_log(self.store, "compile", "Merged retrieval pages", when=when)
        entries = read_log_tail(self.store, limit=10)
        self.assertEqual([e["kind"] for e in entries], ["write", "compile"])
        self.assertEqual(entries[-1]["subject"], "Merged retrieval pages")
        self.assertEqual(entries[0]["date"], "2026-11-08")

    def test_detail_preserved_on_one_line(self):
        append_log(self.store, "lint", "orphan sweep",
                   detail="3 orphans\n2 dangling")
        text = (self.root / LOG_FILENAME).read_text(encoding="utf-8")
        self.assertIn("3 orphans 2 dangling", text)
        self.assertEqual(len(read_log_tail(self.store)), 1)

    def test_parseable_with_grep_prefix(self):
        append_log(self.store, "write", "a")
        append_log(self.store, "write", "b")
        lines = [ln for ln in (self.root / LOG_FILENAME)
                 .read_text(encoding="utf-8").splitlines() if ln.startswith("## [")]
        self.assertEqual(len(lines), 2)

    def test_unicode_and_empty_subject_safe(self):
        self.assertIsNotNone(append_log(self.store, "write", "café — ünïcode ✅"))
        self.assertIsNotNone(append_log(self.store, "", ""))
        entries = read_log_tail(self.store)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[1]["kind"], "event")
        self.assertEqual(entries[1]["subject"], "(no subject)")

    def test_tail_limit_returns_newest(self):
        for i in range(6):
            append_log(self.store, "write", f"entry-{i}")
        entries = read_log_tail(self.store, limit=2)
        self.assertEqual([e["subject"] for e in entries], ["entry-4", "entry-5"])

    def test_missing_log_returns_empty(self):
        self.assertEqual(read_log_tail(self.store), [])

    def test_append_never_raises_on_unwritable_path(self):
        blocker = self.root / "blocker2"
        blocker.write_text("x", encoding="utf-8")
        self.assertIsNone(append_log(self.store, "write", "x",
                                     path=blocker / LOG_FILENAME))


if __name__ == "__main__":
    unittest.main(verbosity=2)
