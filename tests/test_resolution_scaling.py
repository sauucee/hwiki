"""Resolution must not walk the whole vault once per link.

``_resolve_in`` originally scanned ``pages`` linearly for every lookup. Callers
that resolve one link at a time are fine, but ``dangling_links`` and
``link_graph`` call it once per wikilink in the vault, which made them O(n^2) in
page count. Measured on synthetic vaults of 3 links per page:

    pages    before     after
      200    0.211s     (linear)
      400    0.801s
      800    3.094s
     1600   12.621s

That is ~4x per doubling. The fix builds the lookup tables once per pass and
reuses them, so cost is linear.

These tests pin two things: the result is identical to the unindexed scan, and
the scaling no longer quadruples when the vault doubles.

Run: python -m pytest tests/test_resolution_scaling.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.store import WikiStore

MEM = Path("mem")

RESOLUTION_CASES = [
    # (page set, reference, expected path or None)
    ("wiki/guides/release-process.md", "wiki/guides/release-process", "wiki/guides/release-process.md"),
    ("wiki/guides/release-process.md", "wiki/guides/release-process.md", "wiki/guides/release-process.md"),
    ("wiki/guides/release-process.md", "release-process", "wiki/guides/release-process.md"),
    ("wiki/guides/release-process.md", "Release Process", "wiki/guides/release-process.md"),
    ("wiki/concepts/glossary.md", "wiki/concepts/glossary", "wiki/concepts/glossary.md"),
    ("wiki/concepts/glossary.md", "concepts/glossary", None),
    ("wiki/concepts/glossary.md", "nowhere", None),
    ("wiki/concepts/glossary.md", "../escape", None),
    ("wiki/concepts/glossary.md", "/abs/page", None),
    ("wiki/concepts/glossary.md", "glossary#heading", "wiki/concepts/glossary.md"),
    ("wiki/concepts/glossary.md", "glossary|alias", "wiki/concepts/glossary.md"),
    ("wiki/concepts/glossary.md", "[[glossary]]", "wiki/concepts/glossary.md"),
]


def _build(root: Path, rels) -> WikiStore:
    for rel in rels:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {Path(rel).stem}\n\nbody\n", encoding="utf-8")
    return WikiStore(root, memory_dir=MEM.as_posix())


@pytest.mark.parametrize("rel,reference,expected", RESOLUTION_CASES)
def test_resolution_answers_are_unchanged(tmp_path: Path, rel, reference, expected):
    """The indexed path must return exactly what the linear scan returned."""
    store = _build(tmp_path, [rel, "wiki/index.md"])
    page = store.resolve_page(reference)
    assert (page.path if page else None) == expected, (
        f"resolve_page({reference!r}) returned {page.path if page else None!r}, "
        f"expected {expected!r}")


def test_case_twins_are_not_silently_merged(tmp_path: Path):
    """A case-folded reference matching two pages resolves to neither.

    Only the *folded* step refuses; an exact path still wins, which is what makes
    ``Docs/Plan.md`` and ``docs/plan.md`` individually addressable.
    """
    store = _build(tmp_path, ["Docs/Plan.md", "docs/plan.md", "home.md"])
    # Exact paths resolve to themselves.
    assert store.resolve_page("Docs/Plan.md").path == "Docs/Plan.md"
    assert store.resolve_page("docs/plan.md").path == "docs/plan.md"
    # A reference that only matches case-insensitively names neither.
    folded = store.resolve_page("DOCS/PLAN")
    assert (folded.path if folded else None) is None


def test_twin_stems_resolve_by_path_and_fall_back_to_first_slug(tmp_path: Path):
    """Documented behaviour: a bare stem is lossy, an explicit path is exact.

    ``resolve_page``'s fourth step is "a bare slug or title, first match wins,
    unchanged so old links that name no folder keep resolving". Widening that
    would break existing links, so the lossiness is a documented limit — the
    exact path is the way to name one twin.
    """
    store = _build(tmp_path, ["a/notes.md", "b/notes.md", "home.md"])
    assert store.resolve_page("a/notes").path == "a/notes.md"
    assert store.resolve_page("b/notes").path == "b/notes.md"
    # A bare stem picks the first match rather than refusing.
    assert store.resolve_page("notes").path in ("a/notes.md", "b/notes.md")


def _vault_with_links(root: Path, n: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        d = root / f"sec{i % 20}"
        d.mkdir(exist_ok=True)
        (d / f"page-{i}.md").write_text(
            f"# Page {i}\n\n"
            f"[[page-{(i + 1) % n}]]\n"
            f"[[sec{(i + 2) % 20}/page-{(i + 2) % n}]]\n"
            f"[[page-{(i + 3) % n}]]\n",
            encoding="utf-8")


def _time_full_pass(tmp_path: Path, n: int) -> float:
    root = tmp_path / f"v{n}"
    _vault_with_links(root, n)
    store = WikiStore(root, memory_dir=MEM.as_posix())
    start = time.perf_counter()
    store.dangling_links()
    store.link_graph()
    return time.perf_counter() - start



def test_cost_grows_roughly_linearly_not_quadratically(tmp_path: Path):
    """Doubling the vault must not quadruple the time.

    Generous bound (3.0x) so a noisy machine does not fail the build — the
    regression being guarded against is the ~4x quadratic curve, not a few percent.
    """
    small = _time_full_pass(tmp_path, 300)
    large = _time_full_pass(tmp_path, 600)
    ratio = large / small if small > 0 else float("inf")
    assert ratio < 3.0, (
        f"doubling the vault multiplied the cost by {ratio:.1f}x "
        f"({small:.3f}s -> {large:.3f}s); that is the quadratic regression returning")
