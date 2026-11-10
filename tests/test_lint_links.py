"""Lint link classification: dangling vs unindexed.

Two separate code paths report broken links, and both used to conflate "points at
nothing" with "points at a page the index deliberately omits":

* ``lint_memory`` walks the memory zone and reports ``link.dangling``.
* ``store.dangling_links()`` walks every indexed page; the `lint` CLI prints it.

A link to a page that exists in the vault but is excluded from the index is not a
broken link — a human can open it in Obsidian, only recall cannot serve it. On the
live vault this produced 4 of 4 errors, all unfixable, because the targets are
`exclude`d templates. A link to nothing must still be an error, so both halves are
pinned here and the fix cannot regress into silence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.lint import lint_memory, target_exists_on_disk  # noqa: E402
from hwiki.store import WikiStore  # noqa: E402

MEMORY = "Hermes/Memory/Auto"


def _vault() -> WikiStore:
    root = Path(tempfile.mkdtemp(prefix="vault-lint-links-"))
    (root / "Hermes" / "Templates").mkdir(parents=True)
    (root / "Hermes" / "Templates" / "Handoff Template.md").write_text(
        "---\ntitle: Handoff Template\n---\n\n# Handoff\n\nbody\n", encoding="utf-8")
    (root / MEMORY).mkdir(parents=True)
    return WikiStore(root, include=["Hermes/**/*.md"],
                     exclude=["Hermes/Templates/**"],
                     memory_dir=MEMORY)


def _note(store: WikiStore, name: str, body: str) -> None:
    """Write a hand-written note inside the memory zone (lint_memory's target)."""
    (store.root / MEMORY / name).write_text(
        f"---\ntitle: {name}\n---\n\n# {name}\n\n{body}\n", encoding="utf-8")


def _link_findings(store: WikiStore):
    """Only the reference-resolution checks, not orphan/catalog bookkeeping."""
    wanted = {"link.dangling", "link.unindexed"}
    report = lint_memory(store, memory_dir=store.root / MEMORY)
    return [(f.check, f.severity) for f in report.findings if f.check in wanted]


def test_link_to_excluded_page_is_unindexed_not_dangling():
    store = _vault()
    _note(store, "outward.md", "[[Hermes/Templates/Handoff Template|Handoff]]")

    checks = _link_findings(store)
    assert ("link.dangling", "error") not in checks, checks
    assert ("link.unindexed", "warn") in checks, checks


def test_link_to_a_page_that_does_not_exist_is_still_an_error():
    store = _vault()
    _note(store, "broken.md", "[[Hermes/Ghost Page]]")

    checks = _link_findings(store)
    assert ("link.dangling", "error") in checks, checks
    assert ("link.unindexed", "warn") not in checks, checks


def test_an_indexed_page_link_is_neither():
    store = _vault()
    (store.root / "Hermes" / "Notes.md").write_text(
        "---\ntitle: Notes\n---\n\n# Notes\n\nbody\n", encoding="utf-8")
    _note(store, "inward.md", "[[Hermes/Notes]]")

    assert _link_findings(store) == [], _link_findings(store)


def test_the_cli_dangling_report_also_separates_unindexed():
    """`store.dangling_links()` stays honest: it reports, the caller classifies."""
    store = _vault()
    (store.root / "Hermes" / "Home.md").write_text(
        "---\ntitle: Home\n---\n\n# Home\n\n"
        "[[Hermes/Templates/Handoff Template|Handoff]]\n[[Hermes/Ghost Page]]\n",
        encoding="utf-8")

    dangling = store.dangling_links()
    reported = dangling.get("Home", [])
    assert "Hermes/Ghost Page" in reported, dangling
    # The template is resolvable-by-no-one from the index, which is exactly why the
    # CLI must reclassify it rather than print it as broken.
    assert target_exists_on_disk(store.root, "Hermes/Templates/Handoff Template") is True
    assert target_exists_on_disk(store.root, "Hermes/Ghost Page") is False


def test_the_on_disk_probe_stays_inside_the_vault():
    store = _vault()
    outside = store.root.parent / "outside-secret.md"
    outside.write_text("nope\n", encoding="utf-8")
    try:
        assert target_exists_on_disk(store.root, "../outside-secret") is False
        assert target_exists_on_disk(store.root, "/etc/passwd") is False
    finally:
        outside.unlink(missing_ok=True)
