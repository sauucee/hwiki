"""resolve_page: a wrong-folder path falls back to its unique slug.

Wiki links are written without a folder (``[[release-process]]``), so an agent
that guesses the folder (``wiki/concepts/`` vs ``wiki/guides/``) asks for a path
that does not exist. A bare slug always resolved; a ``.md`` path that missed
returned None without trying the slug. Now it falls back to the filename slug,
but only when exactly one page has it.

Run: python -m pytest tests/test_resolve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.store import WikiStore  # noqa: E402


def _page(root: Path, rel: str, title: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: {title}\n---\n\n# {title}\n\nbody of {rel}\n", encoding="utf-8")


@pytest.fixture()
def store(tmp_path: Path) -> WikiStore:
    _page(tmp_path, "wiki/guides/release-process.md", "Release process")
    _page(tmp_path, "wiki/concepts/glossary.md", "Glossary")
    _page(tmp_path, "wiki/concepts/dup.md", "Dup in concepts")
    _page(tmp_path, "wiki/it/dup.md", "Dup in it")
    _page(tmp_path, "raw/release-process-notes.md", "Raw import")
    return WikiStore(tmp_path, untrusted_dirs=("raw",), memory_dir="wiki/memory")


def test_wrong_folder_path_falls_back_to_unique_slug(store: WikiStore) -> None:
    page = store.resolve_page("wiki/concepts/release-process.md")
    assert page is not None
    assert page.path == "wiki/guides/release-process.md"


def test_exact_path_still_wins(store: WikiStore) -> None:
    assert store.resolve_page("wiki/concepts/glossary.md").path == "wiki/concepts/glossary.md"


def test_bare_slug_and_wikilink_still_resolve(store: WikiStore) -> None:
    assert store.resolve_page("release-process").path == "wiki/guides/release-process.md"
    assert store.resolve_page("[[release-process]]").path == "wiki/guides/release-process.md"


def test_ambiguous_slug_is_not_guessed(store: WikiStore) -> None:
    """Two pages named dup.md: a wrong-folder path must not silently pick one."""
    assert store.resolve_page("wiki/guides/dup.md") is None


def test_unknown_page_still_none(store: WikiStore) -> None:
    assert store.resolve_page("wiki/concepts/no-such-page.md") is None


# -- Obsidian path-qualified links (``[[Folder/Page]]``) --------------------
#
# Obsidian writes a folder-qualified link when a bare filename would be ambiguous,
# and the catalog now emits exactly that form for duplicate stems. The resolver has
# to accept the extension-less path, otherwise the catalog's own navigation cannot
# be followed — and a page that links ``[[Folder/Page]]`` is not broken.


def test_folder_qualified_link_resolves_without_the_extension(store: WikiStore) -> None:
    page = store.resolve_page("wiki/guides/release-process")
    assert page is not None
    assert page.path == "wiki/guides/release-process.md"


def test_folder_qualified_link_picks_the_named_twin(store: WikiStore) -> None:
    assert store.resolve_page("wiki/it/dup").path == "wiki/it/dup.md"
    assert store.resolve_page("wiki/concepts/dup").path == "wiki/concepts/dup.md"


def test_bare_ambiguous_stem_keeps_its_old_behaviour(store: WikiStore) -> None:
    """Backward compatibility: a bare twin link still resolves, never to None."""
    assert store.resolve_page("dup") is not None


def test_folder_qualified_link_is_not_dangling(store: WikiStore) -> None:
    (store.root / "wiki" / "notes.md").write_text(
        "# Notes\n\nSee [[wiki/it/dup]] and [[wiki/guides/release-process|release]].\n",
        encoding="utf-8",
    )
    assert store.dangling_links().get("wiki/notes.md", []) == []


def test_backlinks_follow_a_folder_qualified_link(store: WikiStore) -> None:
    (store.root / "wiki" / "notes.md").write_text(
        "# Notes\n\nSee [[wiki/it/dup]].\n", encoding="utf-8")
    _out, backlinks = store.link_graph()
    assert "notes" in backlinks.get("dup", [])


# -- exact case, generated link forms, traversal ----------------------------
#
# Three separate contracts around ``resolve_page``:
#   1. exact-case paths win, and case folding must never pick one of two colliding
#      paths (``Docs/Plan.md`` vs ``docs/plan.md``);
#   2. the link form the catalog itself emits (``[[Folder/Page|Page]]``) resolves;
#   3. a traversal or absolute reference is refused outright — it must not fall
#      through to a basename/title match and quietly resolve to some other page.


@pytest.fixture()
def case_store(tmp_path: Path) -> WikiStore:
    _page(tmp_path, "Docs/Plan.md", "Upper plan")
    _page(tmp_path, "docs/plan.md", "Lower plan")
    return WikiStore(tmp_path, memory_dir="wiki/memory")


def test_exact_case_path_wins_when_two_paths_differ_only_by_case(case_store: WikiStore) -> None:
    assert case_store.resolve_page("Docs/Plan").path == "Docs/Plan.md"
    assert case_store.resolve_page("docs/plan").path == "docs/plan.md"


def test_case_folded_path_is_refused_when_two_paths_collide(case_store: WikiStore) -> None:
    assert case_store.resolve_page("DOCS/PLAN") is None


def test_case_only_mismatch_resolves_when_the_path_is_unique(store: WikiStore) -> None:
    assert store.resolve_page("WIKI/GUIDES/RELEASE-PROCESS").path == \
        "wiki/guides/release-process.md"


def test_generated_catalog_link_form_resolves(store: WikiStore) -> None:
    assert store.resolve_page("[[wiki/guides/release-process|release]]").path == \
        "wiki/guides/release-process.md"
    assert store.resolve_page("wiki/guides/release-process#steps").path == \
        "wiki/guides/release-process.md"
    assert store.resolve_page("[[wiki/it/dup#intro|the it copy]]").path == "wiki/it/dup.md"


def test_traversal_reference_is_refused_not_basename_resolved(store: WikiStore) -> None:
    assert store.resolve_page("../../wiki/guides/release-process") is None
    assert store.resolve_page("../release-process") is None
    assert store.resolve_page("/release-process") is None


def test_link_graph_keeps_exact_case_when_paths_collide(case_store: WikiStore) -> None:
    (case_store.root / "Docs" / "upper-note.md").write_text(
        "# upper note\n\nSee [[Docs/Plan]].\n", encoding="utf-8")
    (case_store.root / "docs" / "lower-note.md").write_text(
        "# lower note\n\nSee [[docs/plan]].\n", encoding="utf-8")
    _out, backlinks = case_store.link_graph()
    assert "upper-note" in backlinks.get("Plan", []), backlinks
    assert "lower-note" in backlinks.get("plan", []), backlinks
    assert "lower-note" not in backlinks.get("Plan", []), backlinks
    assert "upper-note" not in backlinks.get("plan", []), backlinks


def test_link_graph_documents_that_identical_filenames_share_a_key(store: WikiStore) -> None:
    """Known limit, locked so it is a decision rather than an accident.

    The graph is keyed by slug (a public shape: prompt injection and ``hwiki status``
    read it that way). Two pages named ``dup.md`` therefore share one key and their
    backlinks merge; only a path-keyed graph could separate them, which would be a
    breaking API change and is deliberately not done here.
    """
    (store.root / "wiki" / "it-note.md").write_text(
        "# it note\n\nSee [[wiki/it/dup]].\n", encoding="utf-8")
    (store.root / "wiki" / "concepts-note.md").write_text(
        "# concepts note\n\nSee [[wiki/concepts/dup]].\n", encoding="utf-8")
    _out, backlinks = store.link_graph()
    assert "it-note" in backlinks.get("dup", [])
    assert "concepts-note" in backlinks.get("dup", [])


# -- graph identity ---------------------------------------------------------
#
# ``link_graph()`` is public: ``wiki_memory_read`` returns ``backlinks`` from it,
# ``wiki_memory_status(lint=True)`` and ``hermes hwiki lint`` derive their orphan
# lists from it. Keys are slugs, so two pages that share a filename share one key
# and every consumer misattributes the other page's links. Folding paths to slugs
# is what did that, so the graph needs an identity-preserving view — without
# changing the shape existing callers read.


@pytest.fixture()
def twin_store(tmp_path: Path) -> WikiStore:
    _page(tmp_path, "a/decisions.md", "Decisions A")
    _page(tmp_path, "b/decisions.md", "Decisions B")
    (tmp_path / "a" / "note.md").write_text(
        "# note a\n\nSee [[a/decisions]].\n", encoding="utf-8")
    (tmp_path / "b" / "note.md").write_text(
        "# note b\n\nSee [[b/decisions]].\n", encoding="utf-8")
    (tmp_path / "both.md").write_text(
        "# both\n\nSee [[a/decisions]] and [[b/decisions]].\n", encoding="utf-8")
    return WikiStore(tmp_path, memory_dir="wiki/memory")


def test_slug_graph_merges_backlinks_of_two_pages_that_share_a_filename(
        twin_store: WikiStore) -> None:
    """Documented limit of the slug-keyed view, locked so it stays deliberate."""
    _out, backlinks = twin_store.link_graph()
    assert set(backlinks.get("decisions", [])) == {"note", "both"}


def test_path_graph_attributes_each_backlink_to_the_page_it_names(
        twin_store: WikiStore) -> None:
    _out, backlinks = twin_store.link_graph(key="path")
    # each twin gets only the sources that link to it, and those sources are named
    # by their own identity: a/note.md and b/note.md no longer merge under "note"
    assert set(backlinks["a/decisions.md"]) == {"a/note.md", "both.md"}
    assert set(backlinks["b/decisions.md"]) == {"b/note.md", "both.md"}


def test_slug_graph_outlinks_do_not_repeat_a_collapsed_target(
        twin_store: WikiStore) -> None:
    outlinks, _backlinks = twin_store.link_graph()
    assert outlinks["both"] == ["decisions"]


def test_path_graph_outlinks_keep_both_targets_distinct(twin_store: WikiStore) -> None:
    outlinks, _backlinks = twin_store.link_graph(key="path")
    assert outlinks["both.md"] == ["a/decisions.md", "b/decisions.md"]


def test_graph_rejects_an_unknown_key_mode(twin_store: WikiStore) -> None:
    with pytest.raises(ValueError):
        twin_store.link_graph(key="nonsense")


# -- literal link forms and hostile references ------------------------------
#
# ``[[target|alias]]`` and ``[[target#heading]]`` are ordinary Obsidian syntax and
# the catalog emits the first form. A heading-only link (``[[#Heading]]``) points at
# the page it lives on, so it is neither a page reference nor a dangling link.


def test_heading_and_alias_together_resolve(tmp_path: Path) -> None:
    _page(tmp_path, "Hermes/Decisions.md", "Decisions")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    assert store.resolve_page("[[Hermes/Decisions#Open questions|the log]]") is not None
    assert store.resolve_page("Hermes/Decisions#Open questions") is not None


def test_alias_may_itself_contain_a_pipe_or_brackets(tmp_path: Path) -> None:
    _page(tmp_path, "Hermes/Decisions.md", "Decisions")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    assert store.resolve_page("[[Hermes/Decisions|a | b ]]") is not None


def test_heading_only_link_is_not_dangling_and_names_no_page(tmp_path: Path) -> None:
    _page(tmp_path, "Hermes/Decisions.md", "Decisions")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    page = store.resolve_page("Hermes/Decisions")
    assert page is not None
    assert store.dangling_links() == {}
    assert store.resolve_page("[[#Open questions]]") is None


def test_unsafe_references_are_refused_without_raising(tmp_path: Path) -> None:
    _page(tmp_path, "Hermes/Decisions.md", "Decisions")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    for hostile in ("../../etc/passwd", "..\\..\\etc\\passwd", "/etc/passwd",
                    "\\etc\\passwd", "Hermes/../../Decisions", "a/../../b",
                    "Hermes/Decisions.md\0", "\0", "[[../Decisions]]"):
        assert store.resolve_page(hostile) is None, hostile


def test_unsafe_reference_in_a_page_text_is_reported_not_resolved(tmp_path: Path) -> None:
    _page(tmp_path, "Hermes/Decisions.md", "Decisions")
    (tmp_path / "Hermes" / "note.md").write_text(
        "# note\n\n[[../Decisions]] and [[/etc/passwd]]\n", encoding="utf-8")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    dangling = store.dangling_links(key="path")
    assert dangling.get("Hermes/note.md") == ["../Decisions", "/etc/passwd"], dangling
    # the default view is unchanged: slug keys
    assert store.dangling_links() == {"note": ["../Decisions", "/etc/passwd"]}


def test_url_like_target_is_not_a_page(tmp_path: Path) -> None:
    _page(tmp_path, "Hermes/Decisions.md", "Decisions")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    assert store.resolve_page("[[https://example.com/x]]") is None
    assert store.resolve_page("[[file:///etc/passwd]]") is None


def test_public_read_shows_each_twins_own_backlinks(tmp_path: Path) -> None:
    """``wiki_memory_read`` is the surface an agent actually reads."""
    import json

    from hwiki import WikiMemoryProvider

    _page(tmp_path, "wiki/a/decisions.md", "Decisions A")
    _page(tmp_path, "wiki/b/decisions.md", "Decisions B")
    (tmp_path / "wiki" / "a" / "note.md").write_text(
        "# note a\n\nSee [[wiki/a/decisions]].\n", encoding="utf-8")
    (tmp_path / "wiki" / "b" / "note.md").write_text(
        "# note b\n\nSee [[wiki/b/decisions]].\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    provider = WikiMemoryProvider(config={
        "wiki_path": str(tmp_path), "backend": "fts5", "memory_dir": "wiki/memory",
        "index_path": str(home / "index.sqlite"), "reindex_interval_s": 0,
        "_hermes_home": str(home),
    })
    provider.initialize("s", hermes_home=str(home), platform="cli")

    def read(page: str):
        return json.loads(provider.handle_tool_call("wiki_memory_read", {"page": page}))

    a, b = read("wiki/a/decisions"), read("wiki/b/decisions")
    assert a["path"] == "wiki/a/decisions.md" and b["path"] == "wiki/b/decisions.md"
    # index.md legitimately links both; the twin's own note must not leak across
    assert "wiki/a/note.md" in a["backlink_paths"], a["backlink_paths"]
    assert "wiki/b/note.md" not in a["backlink_paths"], a["backlink_paths"]
    assert "wiki/b/note.md" in b["backlink_paths"], b["backlink_paths"]
    assert "wiki/a/note.md" not in b["backlink_paths"], b["backlink_paths"]
    # the historical slug field still exists and still merges the twins
    assert a["backlinks"] == b["backlinks"] == ["index", "note"]


def test_public_status_does_not_hide_an_orphan_twin(tmp_path: Path) -> None:
    """The status lint block used the slug graph, so a twin's backlinks covered both."""
    import json

    from hwiki import WikiMemoryProvider

    _page(tmp_path, "wiki/a/decisions.md", "Decisions A")
    _page(tmp_path, "wiki/b/decisions.md", "Decisions B")
    (tmp_path / "wiki" / "a" / "note.md").write_text(
        "# note a\n\nSee [[wiki/a/decisions]].\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    provider = WikiMemoryProvider(config={
        "wiki_path": str(tmp_path), "backend": "fts5", "memory_dir": "wiki/memory",
        "index_path": str(home / "index.sqlite"), "reindex_interval_s": 0,
        "catalog_enabled": False,  # no generated index: b/decisions stays orphaned
        "_hermes_home": str(home),
    })
    provider.initialize("s", hermes_home=str(home), platform="cli")
    status = json.loads(provider.handle_tool_call("wiki_memory_status", {"lint": True}))
    lint = status["lint"]
    orphans = lint["orphan_page_paths"]
    assert "wiki/b/decisions.md" in orphans, orphans      # linked by nobody
    assert "wiki/a/decisions.md" not in orphans, orphans  # linked by a/note.md
    # slug-keyed orphans keep the old (merged) semantics, beside the exact view
    assert "decisions" not in lint["orphan_pages"], lint["orphan_pages"]


# -- one resolver, three consumers ------------------------------------------
#
# ``resolve_page`` accepts a ``.md`` path whose folder is wrong when the filename is
# unique (an agent guessing ``concepts/`` for a page under ``guides/``). The graph and
# the dangling report used their own path tables, so the same link resolved for the
# reader but produced a phantom backlink key and a dangling report.


def test_md_reference_with_a_wrong_folder_is_not_a_phantom_backlink(tmp_path: Path) -> None:
    _page(tmp_path, "a/target.md", "Target")
    (tmp_path / "a" / "note.md").write_text(
        "# note\n\nSee [[b/target.md]].\n", encoding="utf-8")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    outlinks, backlinks = store.link_graph(key="path")
    assert backlinks["a/target.md"] == ["a/note.md"], backlinks
    assert "b/target.md" not in backlinks, backlinks
    assert outlinks["a/note.md"] == ["a/target.md"], outlinks


def test_md_reference_with_a_wrong_folder_is_not_dangling(tmp_path: Path) -> None:
    _page(tmp_path, "a/target.md", "Target")
    (tmp_path / "a" / "note.md").write_text(
        "# note\n\nSee [[b/target.md]].\n", encoding="utf-8")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    assert store.dangling_links() == {}


def test_lint_agrees_with_the_store_on_what_is_dangling(tmp_path: Path) -> None:
    """Lint must not error on a link the store can open, nor stay silent on one it cannot."""
    from hwiki.lint import lint_memory

    mem = tmp_path / "wiki" / "memory"
    mem.mkdir(parents=True)
    _page(tmp_path, "a/target.md", "Target")
    _page(tmp_path, "a/twin.md", "Twin")
    _page(tmp_path, "docs/plan.md", "lower plan")
    _page(tmp_path, "Docs/Plan.md", "upper plan")
    targets = ["b/target.md", "a/target", "A/TARGET", "nowhere.md", "nowhere",
               "../a/target", "../a/target.md", "/a/target",
               "[[a/target|alias]]", "DOCS/PLAN", "Docs/Plan", "docs/plan",
               "a/twin"]
    (mem / "hub.md").write_text(
        "---\ntitle: Hub\n---\n\n# Hub\n\n"
        + " ".join(target if target.startswith("[[") else f"[[{target}]]"
                   for target in targets) + "\n",
        encoding="utf-8")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    report = lint_memory(store)
    lines = [f.line for f in report.findings if f.check == "link.dangling"]
    expected = [t for t in targets if store.resolve_page(t) is None]
    assert len(lines) == len(expected), (lines, expected)
    for target in expected:
        raw = target.strip("[]").split("|")[0].split("#")[0]
        assert any(f"[[{raw}]]" in line for line in lines), (target, lines)
    # a heading-only link names no page and is not a dangling link
    assert store.resolve_page("[[#heading]]") is None


# -- review findings --------------------------------------------------------
#
# A literal page path is data, not wikilink syntax. ``wiki_memory_read`` and a
# backticked catalog entry pass the exact root-relative path, so a ``#`` or ``|`` in
# a filename must be matched before any wikilink parsing happens — otherwise the
# catalog lists a page nobody can open again.


def test_a_literal_page_path_with_a_hash_resolves(tmp_path: Path) -> None:
    _page(tmp_path, "a/ha#sh.md", "Hash")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    page = store.resolve_page("a/ha#sh.md")
    assert page is not None and page.path == "a/ha#sh.md"
    assert store.resolve_page("a/ha#sh") is not None


def test_a_literal_page_path_with_a_pipe_resolves(tmp_path: Path) -> None:
    _page(tmp_path, "a/we|ird.md", "Pipe")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    page = store.resolve_page("a/we|ird.md")
    assert page is not None and page.path == "a/we|ird.md"


def test_a_literal_path_wins_over_wikilink_syntax(tmp_path: Path) -> None:
    """``a#b`` must not be read as "page a, heading b" when a page is named that."""
    _page(tmp_path, "a#b.md", "Hash Page")
    _page(tmp_path, "a.md", "Plain")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    assert store.resolve_page("a#b.md").path == "a#b.md"
    # the wikilink form still means "page a, heading b"
    assert store.resolve_page("[[a#b]]").path == "a.md"


def test_links_inside_a_fence_closed_by_a_longer_fence_are_not_edges(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(
        "# A\n\n```\n[[ghost]]\n````\n\nreal [[live]]\n", encoding="utf-8")
    _page(tmp_path, "live.md", "Live")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    assert store.resolve_page("a").links() == ["live"]
    assert store.dangling_links() == {}


def test_an_unclosed_fence_swallows_the_rest_of_the_page(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(
        "# A\n\n~~~\n[[ghost]]\n\nstill code [[ghost-two]]\n", encoding="utf-8")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    assert store.resolve_page("a").links() == []


def test_tilde_and_backtick_fences_do_not_close_each_other(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(
        "# A\n\n~~~\n[[ghost]]\n```\n[[still-ghost]]\n~~~\n\n[[live]]\n",
        encoding="utf-8")
    _page(tmp_path, "live.md", "Live")
    store = WikiStore(tmp_path, memory_dir="wiki/memory")
    assert store.resolve_page("a").links() == ["live"]
