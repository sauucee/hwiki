"""A refused forget must not have deleted anything.

``_remove_stale_duplicates`` walked the stale copies and unlinked each safe one
*as it went*, then returned an error on the first unsafe one. The refusal came
after the deletions, so a forget that reported

    {"ok": false, "error": "a duplicate page for this fact carries other facts"}

could already have removed the last on-disk copy of the fact.

Probe G reached it the way it happens in practice: ``compile`` repoints the row at
a shared owner page and the standalone copy is left behind. A hand edit (or an
interrupted compile) removes the merged section from the owner, so the row's page
no longer contains the fact — and the standalone copy is now the only copy. A
forget then walks the stale copies, and the ordering decides whether the only
remaining copy survives.

The rule these tests pin: decide first, mutate second. A refusal must leave the
vault byte-identical.

Run: python -m pytest tests/test_forget_atomicity.py
"""

from __future__ import annotations

import re
from pathlib import Path

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.facts import (Embedder, ProposalStore, _compiled_section_records,
                         read_fact_pages, write_fact_page)
from hwiki.facts_provider import FactsEngine

MEM = Path("Hermes/Memory/Auto")

BASE = "The user prefers dark mode in editors"
MERGED = "The user prefers dark mode in editors and terminals"


def _engine(tmp: Path):
    """A real engine-owned vault + store, the way the provider builds them."""
    vault = tmp / "vault"
    (vault / MEM).mkdir(parents=True)
    home = tmp / "home"
    home.mkdir()
    engine = FactsEngine(str(home), vault, MEM,
                         config={"embeddings": {"mode": "hash"},
                                 "compile_enabled": True})
    return vault, engine, engine.store()


def _accept(store, claim: str):
    proposal = store.propose({"kind": "preference", "claim": claim,
                              "evidence": "stated by the user"},
                             user_text="stated by the user", session_id="s1")
    store.accept(proposal["id"])
    return store.get(proposal["id"])


def _strip_sections(text: str) -> str:
    out, keep = [], True
    for line in text.splitlines(keepends=True):
        if "<!-- hwiki:compiled:start -->" in line:
            keep = False
        if keep:
            out.append(line)
        if "<!-- hwiki:compiled:end -->" in line:
            keep = True
    return "".join(out).rstrip("\n") + "\n"


def _snapshot(directory: Path) -> dict:
    return {p.name: p.read_text(encoding="utf-8")
            for p in sorted(directory.glob("*.md"))}


def _interrupted_compile_state(tmp: Path):
    """Row points at an owner page that no longer carries the merged section.

    The standalone copy written by compile is the only place the fact survives.
    """
    vault, engine, store = _engine(tmp)
    base = _accept(store, BASE)
    engine._promote(base["id"])
    refined = _accept(store, MERGED)
    owner = Path(engine._promote(refined["id"]))
    owner.write_text(_strip_sections(owner.read_text(encoding="utf-8")),
                     encoding="utf-8")
    standalone = Path(write_fact_page(vault, MEM, refined))
    return vault, engine, store, refined, owner, standalone


def test_the_only_remaining_copy_is_not_deleted_by_a_refused_forget(tmp_path: Path):
    """Probe G: the standalone copy is the last one, and the forget refuses."""
    vault, engine, store, refined, owner, standalone = \
        _interrupted_compile_state(tmp_path)

    assert standalone.exists(), "fixture invalid: standalone copy missing"
    before = _snapshot(vault / MEM)

    result = store.forget(proposal_id=refined["id"])

    after = _snapshot(vault / MEM)
    if not result.get("ok"):
        assert after == before, (
            "a refused forget changed the vault; missing: "
            f"{sorted(set(before) - set(after))}")
        assert any(MERGED in text for text in after.values()), (
            "the fact now exists nowhere on disk after a refused forget")
        assert store.get(refined["id"])["status"] == "accepted", (
            "a refused forget must not change the row status")


def test_a_safe_copy_before_an_unsafe_one_is_not_deleted_on_refusal(tmp_path: Path):
    """Two stale copies where a later one is unsafe: refuse without mutating."""
    vault, engine, store = _engine(tmp_path)
    fact = _accept(store, BASE)

    # The row's page exists (so the duplicate scan runs) but carries no section.
    owner = vault / MEM / "aaa-owner.md"
    owner.write_text("---\ntitle: Owner page claim\n---\n\n# Owner page claim\n",
                     encoding="utf-8")
    store.set_page_path(fact["id"], str(owner))

    # Sorted order puts the deletable copy first and the unsafe one second.
    safe = vault / MEM / "fact-a-safe.md"
    safe.write_text(
        f"---\ntitle: {BASE}\nproposal_id: {fact['id']}\n---\n\n# {BASE}\n",
        encoding="utf-8")
    unsafe = vault / MEM / "fact-z-unsafe.md"
    unsafe.write_text(
        f"---\ntitle: Something else\nproposal_id: {fact['id']}\n---\n\n# "
        "A completely different claim\n", encoding="utf-8")

    stale = store._stale_duplicate_pages(store.get(fact["id"])) or []
    assert [p.name for p, _ in stale] == ["fact-a-safe.md", "fact-z-unsafe.md"], (
        f"fixture invalid: unexpected stale order {[p.name for p, _ in stale]}")

    before = _snapshot(vault / MEM)
    error = store._remove_stale_duplicates(store.get(fact["id"]))
    after = _snapshot(vault / MEM)

    assert error, "fixture should be refused: the second copy carries another claim"
    assert after == before, (
        "a refused duplicate cleanup deleted a page; missing: "
        f"{sorted(set(before) - set(after))}")


def test_duplicates_are_still_removed_when_every_copy_is_safe(tmp_path: Path):
    """The fix must not disable the cleanup it exists to perform."""
    vault, engine, store = _engine(tmp_path)
    fact = _accept(store, BASE)

    primary = write_fact_page(vault, MEM, fact)
    store.set_page_path(fact["id"], primary)
    duplicate = vault / MEM / "fact-a-duplicate.md"
    duplicate.write_text(
        f"---\ntitle: {BASE}\nproposal_id: {fact['id']}\n---\n\n# {BASE}\n",
        encoding="utf-8")
    assert duplicate.exists()

    result = store.forget(proposal_id=fact["id"])

    assert result.get("ok"), f"a safe forget should succeed, got {result}"
    assert not duplicate.exists(), "the stale duplicate should be gone"
    assert not Path(primary).exists(), "the primary page should be gone"
    assert store.get(fact["id"])["status"] == "deleted_by_edit"


def test_an_unreadable_duplicate_candidate_names_itself_in_the_refusal(tmp_path: Path):
    """The fail-closed block must be actionable, not a mystery.

    Forget refuses when any page in the memory directory cannot be read, because
    that page may be the duplicate that would resurrect the fact after a rebuild.
    Refusing is right; refusing *anonymously* left the user with a permanent block
    and no way to find it.
    """
    import os

    vault, engine, store = _engine(tmp_path)
    fact = _accept(store, BASE)
    page = Path(engine._promote(fact["id"]))
    assert page.exists()

    unrelated = vault / MEM / "someone-elses-note.md"
    unrelated.write_text("# handwritten\n", encoding="utf-8")
    os.chmod(unrelated, 0o000)
    try:
        result = store.forget(proposal_id=fact["id"])
        assert not result.get("ok"), (
            "an unreadable candidate must block the forget, got "
            f"{result}")
        assert "someone-elses-note.md" in (result.get("error") or ""), (
            f"the refusal does not name the blocking file: {result.get('error')!r}")
        assert page.exists(), "a refused forget must not delete the fact page"
        assert store.get(fact["id"])["status"] == "accepted"
    finally:
        os.chmod(unrelated, 0o644)
