"""A refusal must not claim the database is untouched when it promoted a fact.

``memory_supersede`` verifies that the replacement fact is visible in the vault.
When it is not, it calls ``_promote`` to repair a page lost after acceptance, and
then reports:

    "supersede failed: replacement is not visible in the vault; no database
     change was made"

That sentence was false: ``_promote`` writes ``page_path`` and ``promoted`` to the
row and puts a page on disk. A caller who reads the message and retries assumes a
clean slate, but the state has moved.

These tests pin the property that the message matches the mutation: either the
call reports what it changed, or it changes nothing.

The refusal path cannot be reached by arranging real files alone — an accepted
fact with no page is *repaired* by the promote, and the retry then sees it. So the
visibility check is patched to refuse both times, which is what makes the second
refusal reachable. Everything else in the flow is real.

Run: python -m pytest tests/test_supersede_message.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.facts import write_fact_page

MEM = Path("Hermes/Memory/Auto")


def _env(tmp: Path):
    """Engine-owned vault/store, so the tool and the assertions share one DB."""
    from hwiki.facts_provider import FactsEngine
    vault = tmp / "vault"
    (vault / MEM).mkdir(parents=True)
    home = tmp / "home"
    home.mkdir()
    engine = FactsEngine(str(home), vault, MEM,
                         config={"embeddings": {"mode": "hash"}})
    store = engine.store()
    return vault, engine, store


def _accepted(store, claim: str, vault: Path | None = None):
    proposal = store.propose({"kind": "preference", "claim": claim,
                              "evidence": "stated"}, user_text="stated",
                             session_id="s1")
    store.accept(proposal["id"])
    if vault is not None:
        page = write_fact_page(vault, MEM, store.get(proposal["id"]))
        store.set_page_path(proposal["id"], page)
    return store.get(proposal["id"])


def test_refusal_does_not_deny_a_change_it_made():
    """If the call mutated the row, it must not report 'no database change'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        vault, engine, store = _env(tmp)

        victim = _accepted(store, "An accepted fact that will be replaced", vault)
        replacement = _accepted(store, "An accepted replacement with no page yet")

        before = store.get(replacement["id"])
        assert before["page_path"] is None, (
            "fixture invalid: the replacement must start with no page, "
            f"got {before['page_path']!r}")
        assert not before["promoted"], "fixture invalid: replacement already promoted"

        with mock.patch.object(engine, "_fact_page_visible",
                               side_effect=[False, False]):
            result = json.loads(engine.handle_tool_call(
                "memory_supersede",
                {"proposal_id": replacement["id"], "supersede_id": victim["id"]},
                lambda message: json.dumps({"error": message})))

        after = store.get(replacement["id"])
        mutated = (after["page_path"] != before["page_path"]
                   or bool(after["promoted"]) != bool(before["promoted"]))

        # The refusal must still refuse: the victim stays accepted, not superseded.
        assert result.get("error"), f"expected a refusal, got {result}"
        assert store.get(victim["id"])["status"] == "accepted", (
            "a refused supersede must not supersede the victim")

        # And the fixture must actually have reached the promote, or this test
        # would pass for the wrong reason.
        assert mutated, (
            "fixture did not reach the promote step, so the message claim is not "
            f"exercised: page_path={after['page_path']!r} promoted={after['promoted']!r}")

        text = json.dumps(result)
        assert "no database change was made" not in text, (
            "the call mutated the row (page_path=%r promoted=%r) but reported "
            "that no database change was made: %s"
            % (after["page_path"], after["promoted"], text))
