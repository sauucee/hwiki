"""A path-shaped `page` must resolve, never become a new flattened page.

Observed bug: writing with ``page="Hermes/Memory/Auto/conventions"`` created
``hermesmemoryautoconventions.md`` beside the real ``conventions.md``. The memory zone
then held two pages claiming the same topic, neither linking the other, and the write was
reported as a success.

The dangerous property is not the odd filename — it is that a lookup miss silently became
a create. Splitting a topic across two pages makes recall quietly partial, which is
indistinguishable from the fact never having been written.

Run: python -m pytest tests/test_write_page_reference.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.store import WikiStore


def _store(tmp_path: Path) -> WikiStore:
    vault = tmp_path / "vault"
    (vault / "Hermes" / "Memory" / "Auto").mkdir(parents=True)
    (vault / "index.md").write_text("# Index\n", encoding="utf-8")
    return WikiStore(vault, memory_dir="Hermes/Memory/Auto")


def _memory_pages(store: WikiStore) -> list:
    return sorted(p.name for p in store.memory_root().glob("*.md"))


def test_a_topic_creates_a_page(tmp_path: Path):
    """The documented way to make a new page keeps working."""
    store = _store(tmp_path)
    result = store.append_note("The user prefers short replies", topic="Conventions")
    assert result["created"] is True
    assert str(result["path"]).endswith("conventions.md")
    assert _memory_pages(store) == ["conventions.md"]


def test_a_path_reference_appends_to_the_existing_page(tmp_path: Path):
    """A path is a lookup: it must land in the real page, not a sibling."""
    store = _store(tmp_path)
    store.append_note("first fact", topic="Conventions")

    store.append_note("second fact", page="Hermes/Memory/Auto/conventions")

    assert _memory_pages(store) == ["conventions.md"], (
        f"a path reference created a second page: {_memory_pages(store)}")
    body = (store.memory_root() / "conventions.md").read_text(encoding="utf-8")
    assert "first fact" in body and "second fact" in body, (
        "the second note did not land in the existing page")


def test_an_unresolvable_path_refuses_instead_of_inventing_a_page(tmp_path: Path):
    """A miss is a mistake in the call, not permission to create a near-duplicate."""
    store = _store(tmp_path)
    store.append_note("first fact", topic="Conventions")

    with pytest.raises(ValueError) as excinfo:
        store.append_note("second fact", page="Projects/Nowhere/missing")

    assert "no memory page matches" in str(excinfo.value)
    assert _memory_pages(store) == ["conventions.md"], (
        "a failed lookup still wrote a page")


def test_a_bare_page_name_is_still_accepted(tmp_path: Path):
    """The common case — a plain slug — must not require an existing page."""
    store = _store(tmp_path)
    result = store.append_note("a fact", page="vendors")
    assert result["created"] is True
    assert _memory_pages(store) == ["vendors.md"]
