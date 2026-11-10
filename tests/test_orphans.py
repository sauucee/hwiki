"""Orphan detection: catalogued pages are navigable, unlisted unreferenced ones are not.

The `lint` CLI recomputed orphans from `link_graph` alone, ignoring `index.md` — so
every page reachable only through the catalog (which is how this wiki is meant to be
navigated) was printed as an orphan. On the live vault that was 11 of 11, all false.

Both halves are pinned here so the fix cannot regress into silence: a page nothing
links *and* nothing lists must still be reported.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.catalog import write_index  # noqa: E402
from hwiki.lint import orphan_pages  # noqa: E402
from hwiki.store import WikiStore  # noqa: E402

MEMORY = "Hermes/Memory/Auto"


def _vault() -> WikiStore:
    root = Path(tempfile.mkdtemp(prefix="vault-orphans-"))
    (root / MEMORY).mkdir(parents=True)
    (root / "Hermes" / "Home.md").write_text(
        "---\ntitle: Home\n---\n\n# Home\n\nbody\n", encoding="utf-8")
    return WikiStore(root, include=["Hermes/**/*.md"], memory_dir=MEMORY)


def _note(store: WikiStore, name: str) -> None:
    (store.root / MEMORY / f"{name}.md").write_text(
        f"---\ntitle: {name}\n---\n\n# {name}\n\nbody\n", encoding="utf-8")


def test_a_catalogued_page_is_not_an_orphan():
    """The regression: listed in index.md, linked by nothing, still navigable."""
    store = _vault()
    _note(store, "conventions")
    write_index(store, memory_dir=MEMORY)

    assert orphan_pages(store) == [], orphan_pages(store)


def test_a_page_neither_linked_nor_listed_is_an_orphan():
    store = _vault()
    _note(store, "lost-note")  # written after the catalog, so not listed
    write_index(store, memory_dir=MEMORY)
    _note(store, "written-after-catalog")

    orphans = orphan_pages(store)
    assert orphans == [f"{MEMORY}/written-after-catalog.md"], orphans


def test_an_inbound_link_is_navigation_even_without_a_catalog_entry():
    store = _vault()
    _note(store, "target")
    write_index(store, memory_dir=MEMORY)
    (store.root / MEMORY / "pointer.md").write_text(
        "---\ntitle: pointer\n---\n\n# pointer\n\nsee [[target]]\n", encoding="utf-8")

    orphans = orphan_pages(store)
    assert f"{MEMORY}/target.md" not in orphans, orphans


def test_generated_artifacts_are_never_orphans():
    from hwiki.maintain import MaintenanceReport, render

    store = _vault()
    (store.root / MEMORY / "maintenance.md").write_text(
        render(MaintenanceReport()), encoding="utf-8")
    write_index(store, memory_dir=MEMORY)

    assert f"{MEMORY}/maintenance.md" not in orphan_pages(store)
