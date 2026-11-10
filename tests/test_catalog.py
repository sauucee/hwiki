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

from hwiki import catalog, maintain  # noqa: E402
from hwiki.facts import Embedder, ProposalStore  # noqa: E402
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



class OwnershipGuardTests(unittest.TestCase):
    """A generated artifact must never overwrite something a person wrote."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = WikiStore(self.root, memory_dir="mem")
        (self.root / "mem").mkdir()
        (self.root / "mem" / "a-fact.md").write_text(
            "---\ntitle: A Fact\n---\n\n# A Fact\n\nStaging stays on Fly.io.\n",
            encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_hand_written_index_is_left_alone(self):
        index = self.root / "index.md"
        mine = "# My Vault\n\n- [[Home]]\n- [[Project X]]\n"
        index.write_text(mine, encoding="utf-8")
        self.assertIsNone(catalog.write_index(self.store, memory_dir="mem"))
        self.assertEqual(index.read_text(encoding="utf-8"), mine)

    def test_a_hand_written_index_is_replaced_only_on_explicit_force(self):
        index = self.root / "index.md"
        index.write_text("# My Vault\n", encoding="utf-8")
        written = catalog.write_index(self.store, memory_dir="mem", force=True)
        self.assertIsNotNone(written)
        self.assertIn("hwiki-catalog", index.read_text(encoding="utf-8"))

    def test_a_generated_index_is_regenerated_normally(self):
        index = self.root / "index.md"
        catalog.write_index(self.store, memory_dir="mem")
        index.write_text(index.read_text(encoding="utf-8") + "\nstale line\n")
        catalog.write_index(self.store, memory_dir="mem")
        self.assertNotIn("stale line", index.read_text(encoding="utf-8"))

    def test_a_missing_index_is_created(self):
        self.assertIsNotNone(catalog.write_index(self.store, memory_dir="mem"))
        self.assertTrue((self.root / "index.md").exists())

    def test_a_hand_written_log_is_not_appended_to(self):
        log = self.root / "log.md"
        mine = "# My Log\n\n- 2026: something I wrote\n"
        log.write_text(mine, encoding="utf-8")
        self.assertIsNone(catalog.append_log(self.store, "accept", "x"))
        self.assertEqual(log.read_text(encoding="utf-8"), mine)

    def test_the_guard_reads_frontmatter_only(self):
        """A marker quoted inside the body is not a claim of ownership."""
        index = self.root / "index.md"
        index.write_text("# Notes\n\nThe generated file says `source: hwiki-catalog`.\n",
                         encoding="utf-8")
        self.assertFalse(catalog.owns_artifact(index))

    def test_the_tick_regenerates_a_stale_generated_index(self):
        """The artifacts job must actually fire — reading the wrong Finding field once
        made it report "catalogs current" forever."""
        catalog.write_index(self.store, memory_dir="mem")
        index = self.root / "index.md"
        index.write_text(
            "---\ntitle: Wiki Index\nsource: hwiki-catalog\n---\n\n# Wiki Index\n\n"
            "- [[gone-page]]\n", encoding="utf-8")
        store = ProposalStore(self.root / "facts.sqlite", embedder=Embedder(mode="hash"))
        report = maintain.run_maintenance(store, self.root, "mem", write=True)
        note = " ".join(n for job in report.jobs for n in job.notes)
        self.assertIn("regenerated index.md", note)
        self.assertNotIn("gone-page", index.read_text(encoding="utf-8"))

    def test_the_tick_leaves_a_hand_written_index_alone(self):
        index = self.root / "index.md"
        index.write_text("# My Vault\n", encoding="utf-8")
        store = ProposalStore(self.root / "facts.sqlite",
                              embedder=Embedder(mode="hash"))
        # a catalog that lies about the vault is what makes the tick want to regenerate
        (self.root / "index.md").write_text("# My Vault\n\n- [[gone-page]]\n",
                                            encoding="utf-8")
        report = maintain.run_maintenance(store, self.root, "mem", write=True)
        note = " ".join(n for job in report.jobs for n in job.notes)
        self.assertIn("My Vault", (self.root / "index.md").read_text(encoding="utf-8"))
        self.assertIn("left alone", note)


class DuplicateStemLinkTests(unittest.TestCase):
    """Two pages with the same filename: the catalog link must name the folder.

    A bare ``[[Decisions]]`` is ambiguous in Obsidian and unresolvable by
    ``WikiStore.resolve_page`` (the resolver refuses to guess between twins), so the
    catalog emitted navigation that its own reader could not follow. Unique stems
    keep the bare form: existing vaults, hand-written links and hand-edited indexes
    must keep working unchanged.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vault"
        (self.root / "Hermes").mkdir(parents=True)
        (self.root / "Projects" / "Alpha").mkdir(parents=True)
        (self.root / "memory" / "auto").mkdir(parents=True)
        (self.root / "Hermes" / "Decisions.md").write_text(
            "# Decisions\n\nHermes-wide decisions.\n", encoding="utf-8")
        (self.root / "Projects" / "Alpha" / "Decisions.md").write_text(
            "# Decisions\n\nProject decisions.\n", encoding="utf-8")
        (self.root / "Hermes" / "Work Log.md").write_text(
            "# Work Log\n\nOne log.\n", encoding="utf-8")
        (self.root / "memory" / "auto" / "fact-one.md").write_text(
            "---\ntitle: Prefers dark mode\nkind: preference\n---\n\n# Prefers dark mode\n",
            encoding="utf-8")
        self.store = WikiStore(self.root, include=["**/*.md"], memory_dir="memory/auto")

    def tearDown(self):
        self.tmp.cleanup()

    def index_targets(self) -> list:
        text = build_index(self.store, memory_dir="memory/auto")
        return [t.strip() for t in _wikilink_targets(text)]

    def test_duplicate_stems_get_a_folder_qualified_link(self):
        text = build_index(self.store, memory_dir="memory/auto")
        self.assertIn("[[Hermes/Decisions|Decisions]]", text)
        self.assertIn("[[Projects/Alpha/Decisions|Decisions]]", text)

    def test_unique_stems_keep_the_bare_link(self):
        text = build_index(self.store, memory_dir="memory/auto")
        self.assertIn("- [[Work Log]]", text)
        self.assertIn("- [[fact-one]]", text)

    def test_every_catalogued_page_resolves_to_its_own_link(self):
        targets = self.index_targets()
        resolved = {self.store.resolve_page(t).path for t in targets}
        catalogued = {p.path for p in self.store.iter_pages()
                      if Path(p.path).name not in (INDEX_FILENAME, LOG_FILENAME)}
        self.assertEqual(resolved, catalogued)
        self.assertNotIn(None, resolved)

    def test_root_page_sharing_a_stem_with_a_nested_page_stays_linkable(self):
        """A root page is reachable by its bare stem even when a twin exists.

        ``[[notes]]`` resolves to the root ``notes.md`` through the exact tier, so
        listing it as "not linkable from an index" was wrong. The catalog used to
        gate the bare-stem option on the stem *count*, which skipped it whenever any
        twin existed and left a perfectly linkable page as a plain path.
        """
        (self.root / "notes.md").write_text("# Root notes\n", encoding="utf-8")
        (self.root / "wiki").mkdir(parents=True, exist_ok=True)
        (self.root / "wiki" / "notes.md").write_text("# Wiki notes\n", encoding="utf-8")

        text = build_index(self.store, memory_dir="memory/auto")
        self.assertNotIn("Not linkable from an index", text)
        # Both pages are linked, and each link opens its own page.
        for target in ("notes", "wiki/notes"):
            page = self.store.resolve_page(target)
            self.assertIsNotNone(page, f"[[{target}]] in the index does not resolve")
        self.assertEqual(self.store.resolve_page("notes").path, "notes.md")
        self.assertEqual(self.store.resolve_page("wiki/notes").path, "wiki/notes.md")


class CatalogOutputContractTests(unittest.TestCase):
    """The index for an ordinary vault must stay byte-identical to the old builder.

    ``tests/data/index_v1_golden.md`` was generated by the previous ``build_index`` on
    this fixture. New behaviour (path-only section, changed counters) must appear only
    for vaults that need it; everything else is a contract existing indexes and diff
    reviews depend on.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vault"
        for folder in ("Hermes", "Projects/Alpha", "concepts", "memory/auto"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        (self.root / "Hermes" / "Decisions.md").write_text(
            "---\ntitle: Decisions\nupdated: 2026-01-02\n---\n\n# Decisions\n\n"
            "Ship small.\n", encoding="utf-8")
        (self.root / "Projects" / "Alpha" / "Decisions.md").write_text(
            "---\ntitle: Decisions\n---\n\n# Decisions\n\nAlpha scope.\n",
            encoding="utf-8")
        (self.root / "concepts" / "retrieval.md").write_text(
            "---\ntitle: Retrieval\nupdated: 2026-01-03\n---\n\n# Retrieval\n\n"
            "Hybrid search.\n", encoding="utf-8")
        (self.root / "memory" / "auto" / "fact-one.md").write_text(
            "---\ntitle: Prefers dark mode\nkind: preference\nupdated: 2026-01-04\n"
            "---\n\n# Prefers dark mode\n", encoding="utf-8")
        self.store = WikiStore(self.root, include=["**/*.md"], memory_dir="memory/auto")

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_output_matches_the_previous_builder(self):
        golden = (Path(__file__).parent / "data" / "index_v1_golden.md").read_text(
            encoding="utf-8")
        self.assertEqual(build_index(self.store, memory_dir="memory/auto",
                                     today="2026-01-01"), golden)


class HostileNameTests(unittest.TestCase):
    """Wikilinks cannot address every filename, so the catalog must not pretend.

    ``[[...]]`` splits on ``|`` (alias) and ``#`` (heading), and ``]]`` ends the link.
    A page whose own path contains one of those characters cannot be named by any
    wikilink — the resolver strips it and lands on a different page or nothing. The
    catalog used to emit a folder-qualified link for such twin anyway, producing a
    link that silently pointed nowhere.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vault"
        for folder in ("a", "b", "memory/auto"):
            (self.root / folder).mkdir(parents=True)
        (self.root / "a" / "we|ird.md").write_text("# A\n", encoding="utf-8")
        (self.root / "b" / "we|ird.md").write_text("# B\n", encoding="utf-8")
        (self.root / "a" / "hash#note.md").write_text("# H\n", encoding="utf-8")
        (self.root / "a" / "bracket]].md").write_text("# C\n", encoding="utf-8")
        (self.root / "a" / "plain.md").write_text("# P\n", encoding="utf-8")
        self.store = WikiStore(self.root, include=["**/*.md"], memory_dir="memory/auto")

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_emitted_link_is_truncated_by_a_hostile_name(self):
        text = build_index(self.store, memory_dir="memory/auto")
        for target in _wikilink_targets(text):
            page = self.store.resolve_page(target)
            self.assertIsNotNone(page, target)
            self.assertNotIn("|", target)
            self.assertNotIn("#", target)
            self.assertNotIn("]", target)

    def test_hostile_pages_are_listed_as_plain_paths_not_links(self):
        text = build_index(self.store, memory_dir="memory/auto")
        for name in ("a/we|ird.md", "a/hash#note.md", "a/bracket]].md", "b/we|ird.md"):
            self.assertIn(f"`{name}`", text, text)
        self.assertIn("[[plain]]", text)

    def test_a_linkable_catalog_still_covers_every_linkable_page(self):
        text = build_index(self.store, memory_dir="memory/auto")
        resolved = {self.store.resolve_page(t).path for t in _wikilink_targets(text)}
        expected = {p.path for p in self.store.iter_pages()
                    if catalog.linkable_target(p.path)}
        self.assertEqual(resolved, expected)


def _wikilink_targets(text: str) -> list:
    import re
    return [m.group(1) for m in re.finditer(r"\[\[([^\]|#]+)", text)]


class CatalogLinkIdentityTests(unittest.TestCase):
    """Every emitted link must open the page the catalog lists it for."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "memory" / "Auto").mkdir(parents=True)
        self.store = WikiStore(self.root, memory_dir="memory/Auto")

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_emitted_link_resolves_to_a_page_the_catalog_meant(self):
        """The catalog's one job: a listed link opens the page it is listed for.

        Stems are qualified only on collision, which is not enough when a page's stem
        also names a root artifact (``Projects/index.md`` beside ``index.md``). Every
        emitted target is checked against the store, and the set of pages they resolve
        to must be exactly the pages the catalog is supposed to list.
        """
        import re
        from hwiki.catalog import is_catalog_artifact, linkable_target
        for rel, text in {
            "README.md": "# README\n",
            "index.md": "# Root index\n",
            "log.md": "# Root log\n",
            "Projects/index.md": "# Project index\n",
            "Projects/log.md": "# Project log\n",
            "Projects/maintenance.md": "# Maintenance window\n",
            "a/notes.md": "# a notes\n",
            "b/notes.md": "# b notes\n",
            "a/ha#sh.md": "# Hash\n",
            "a/we|ird.md": "# Pipe\n",
            "memory/Auto/maintenance.md": "---\ntype: maintenance\n---\n\n# Report\n",
        }.items():
            path = self.store.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        index = build_index(self.store)
        targets = re.findall(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]", index)
        self.assertTrue(targets)
        resolved = {}
        for target in targets:
            page = self.store.resolve_page(target)
            self.assertIsNotNone(page, f"emitted target {target!r} does not resolve")
            resolved[page.path] = target
        expected = {
            page.path for page in self.store.iter_pages()
            if linkable_target(page.path)
            and not is_catalog_artifact(page.path, memory_dir="memory/Auto")
        }
        self.assertEqual(set(resolved), expected)


class LogResilienceTests(unittest.TestCase):
    """``log.md`` is an audit trail: appends must not be lost or merged.

    It is also the one artifact written on every interesting event, so it is written
    from turns that can crash, from two turns at once, and after the file has been
    truncated by a full disk. None of those may corrupt or lose history, and none may
    raise into the caller.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = _make_wiki(self.root)
        self.log = self.root / LOG_FILENAME

    def tearDown(self):
        self.tmp.cleanup()

    # -- crash and truncation ------------------------------------------

    def test_append_after_a_truncated_final_line_does_not_merge_entries(self):
        append_log(self.store, "write", "first")
        self.log.write_text(
            self.log.read_text(encoding="utf-8").rstrip("\n") + "\n## [2026-01-02] wri",
            encoding="utf-8")
        broken = self.log.read_text(encoding="utf-8")
        append_log(self.store, "write", "second")
        after = self.log.read_text(encoding="utf-8")
        self.assertTrue(after.startswith(broken), "append must only add bytes")
        entries = read_log_tail(self.store)
        self.assertEqual([e["subject"] for e in entries], ["first", "second"], entries)

    def test_an_empty_log_is_adopted_instead_of_blocking_appends(self):
        self.log.write_bytes(b"")
        self.assertIsNotNone(append_log(self.store, "write", "after crash"))
        entries = read_log_tail(self.store)
        self.assertEqual([e["subject"] for e in entries], ["after crash"])

    def test_a_non_empty_hand_written_log_is_still_left_alone(self):
        self.log.write_text("# My log\n\nnotes go here\n", encoding="utf-8")
        self.assertIsNone(append_log(self.store, "write", "not mine"))
        self.assertEqual(self.log.read_text(encoding="utf-8"),
                         "# My log\n\nnotes go here\n")

    # -- concurrent appends --------------------------------------------

    def test_concurrent_appends_on_a_fresh_log_lose_nothing(self):
        import threading

        def writer(index):
            append_log(self.store, "write", f"entry-{index}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        entries = read_log_tail(self.store, limit=0)
        self.assertEqual(sorted(e["subject"] for e in entries),
                         sorted(f"entry-{i}" for i in range(16)))

    def test_concurrent_appends_on_an_existing_log_lose_nothing(self):
        import threading

        append_log(self.store, "write", "seed")
        threads = [threading.Thread(target=append_log,
                                    args=(self.store, "write", f"entry-{i}"))
                   for i in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        entries = read_log_tail(self.store, limit=0)
        self.assertEqual(sorted(e["subject"] for e in entries),
                         sorted(["seed"] + [f"entry-{i}" for i in range(12)]))
        text = self.log.read_text(encoding="utf-8")
        self.assertEqual(text.count("type: log"), 1)

    # -- malformed inputs and timestamps -------------------------------

    def test_append_never_raises_on_a_bad_timestamp(self):
        self.assertIsNone(append_log(self.store, "write", "x", when="2026-01-01"))
        self.assertFalse(self.log.exists())

    def test_append_never_raises_on_non_string_fields(self):
        self.assertIsNotNone(append_log(self.store, None, None, detail=None))
        entries = read_log_tail(self.store)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["kind"], "event")

    def test_a_naive_datetime_is_accepted(self):
        self.assertIsNotNone(append_log(self.store, "write", "naive",
                                        when=datetime(2026, 1, 2)))
        self.assertEqual(read_log_tail(self.store)[0]["date"], "2026-01-02")

    def test_a_subject_containing_the_delimiter_round_trips(self):
        subject = "a | b ## [2026-01-01] fake | entry"
        append_log(self.store, "write", subject)
        entries = read_log_tail(self.store)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["subject"], subject)

    def test_a_control_character_in_the_subject_is_removed(self):
        append_log(self.store, "write", "before\x00after\x1b[31mred")
        entries = read_log_tail(self.store)
        self.assertEqual(len(entries), 1)
        self.assertNotIn("\x00", entries[0]["subject"])
        self.assertNotIn("\x1b", entries[0]["subject"])

    def test_limit_zero_and_negative_are_documented_as_all(self):
        for i in range(3):
            append_log(self.store, "write", f"e-{i}")
        self.assertEqual(len(read_log_tail(self.store, limit=0)), 3)
        self.assertEqual(len(read_log_tail(self.store, limit=-1)), 3)

    def test_a_partial_final_line_after_a_crash_is_ignored(self):
        append_log(self.store, "write", "complete")
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write("## [2026-01-02] wri")
        self.assertEqual([e["subject"] for e in read_log_tail(self.store)], ["complete"])

    def test_crlf_and_noise_lines_do_not_hide_entries(self):
        append_log(self.store, "write", "first")
        text = self.log.read_text(encoding="utf-8").replace("\n", "\r\n")
        text += "not an entry\r\n## [not-a-date] write | ignored\r\n"
        self.log.write_bytes(text.encode("utf-8"))
        self.assertEqual([e["subject"] for e in read_log_tail(self.store)], ["first"])

    def test_non_string_fields_include_numbers(self):
        self.assertIsNotNone(append_log(self.store, 42, 99, detail=7))
        entry = read_log_tail(self.store)[0]
        self.assertEqual(entry["kind"], "42")
        self.assertEqual(entry["subject"], "99")


class LogTailCostTests(unittest.TestCase):
    """``read_log_tail`` is called by every status read; a long log must stay cheap.

    The log is append-only and never truncated, so it grows for the life of the vault.
    Reading it whole to return five entries turned status into a multi-megabyte read
    (measured: 7 MB -> 148 ms, ~96 MB peak with the loader) on every call.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = _make_wiki(self.root)
        lines = [catalog._LOG_HEADER]
        for i in range(4000):
            lines.append(f"## [2026-01-01] write | entry {i} — unicode ışık ✅\n\n")
        (self.root / LOG_FILENAME).write_text("".join(lines), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_returning_a_short_tail_does_not_allocate_the_whole_log(self):
        import tracemalloc

        tracemalloc.start()
        try:
            entries = read_log_tail(self.store, limit=5)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(len(entries), 5)
        self.assertEqual(entries[-1]["subject"], "entry 3999 — unicode ışık ✅")
        self.assertLess(peak, 512 * 1024, f"peak {peak} bytes")

    def test_tail_matches_full_read_for_every_limit(self):
        full = read_log_tail(self.store, limit=0)
        self.assertEqual(len(full), 4000)
        for limit in (1, 2, 5, 10, 50, 4000, 5000):
            self.assertEqual(read_log_tail(self.store, limit=limit), full[-limit:],
                             limit)

    def test_appending_does_not_read_the_whole_log(self):
        """Ownership lives in the frontmatter; a grown log must not be re-read per write."""
        import tracemalloc

        lines = [catalog._LOG_HEADER]
        lines += [f"## [2026-01-01] write | entry {i} with padding text\n\n"
                  for i in range(20000)]
        log = self.root / LOG_FILENAME
        log.write_text("".join(lines), encoding="utf-8")
        self.assertGreater(log.stat().st_size, 1_000_000)
        tracemalloc.start()
        try:
            self.assertIsNotNone(append_log(self.store, "write", "after"))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 512 * 1024, f"peak {peak} bytes")
        self.assertEqual(read_log_tail(self.store, limit=1)[0]["subject"], "after")


if __name__ == "__main__":
    unittest.main(verbosity=2)
