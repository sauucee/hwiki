"""Ranking-quality tests: stopword hygiene + recency-aware rerank.

Complements test_store_search.py's selftest corpus checks with focused
regressions for the two recall-quality levers in Fts5Backend.search.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.search import Fts5Backend, build_match_query  # noqa: E402
from hwiki.store import WikiStore  # noqa: E402


def _make_store(root: Path) -> Fts5Backend:
    store = WikiStore(root)
    backend = Fts5Backend(store, root / "index.sqlite")
    return backend


def _write_page(root: Path, rel: str, text: str, updated: str | None = None) -> None:
    page = root / rel
    page.parent.mkdir(parents=True, exist_ok=True)
    fm = "---\nspine: concepts\ntype: concept\nstatus: maintained\n"
    if updated:
        fm += f"updated: {updated}\n"
    page.write_text(fm + "---\n\n" + text, encoding="utf-8")


def test_stopword_hygiene_query_tokens() -> None:
    # "user" must not survive into the MATCH expression: it would match every
    # memory page's "User prefers..." title and swamp real topic matches.
    match = build_match_query("what does the user prefer to drink")
    assert "user" not in match.lower()
    assert "prefer" in match.lower()
    assert "drink" in match.lower()


def test_personal_possessives_dropped() -> None:
    match = build_match_query("what is in my notes about their deployment")
    for banned in ("my", "their", "in "):
        assert f'"{banned}"' not in match.lower()


def test_recency_rerank_prefers_fresh_page() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stale_date = (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")
        _write_page(root, "wiki/facts/old.md",
                    "# Deployment\n\nThe deployment target is Fly.io.\n",
                    updated=stale_date)
        _write_page(root, "wiki/facts/new.md",
                    "# Deployment\n\nThe deployment target is Fly.io.\n",
                    updated=today)
        backend = _make_store(root)
        backend.sync(force=True)
        hits = backend.search("deployment target", limit=2)
        assert len(hits) == 2
        assert hits[0].path.endswith("new.md"), f"fresh page should rank first, got {hits[0].path}"
        assert hits[0].score > hits[1].score


def test_stale_page_is_demoted_but_never_hidden() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        stale_date = (datetime.now(timezone.utc) - timedelta(days=300)).strftime("%Y-%m-%d")
        _write_page(root, "wiki/facts/only.md",
                    "# Backup\n\nNightly backups run with restic.\n",
                    updated=stale_date)
        backend = _make_store(root)
        backend.sync(force=True)
        hits = backend.search("backups restic", limit=3)
        assert len(hits) == 1  # a lone stale answer stays reachable
        assert hits[0].score > 0
