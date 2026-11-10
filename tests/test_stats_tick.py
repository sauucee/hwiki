"""The maintenance tick must keep generated stats blocks current by itself.

``stats.py`` can render and replace a block; that proves nothing about whether it
ever runs. This is the autonomy test: run one real tick against a real vault and
assert the block moved — and that a page without markers was not touched, and that
a second tick is a no-op (no write, no diff), so a quiet vault stays quiet.

Run: python -m pytest tests/test_stats_tick.py
"""

from __future__ import annotations

from pathlib import Path

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki import maintain
from hwiki.facts import Embedder, ProposalStore
from hwiki.stats import STATS_END, STATS_START, has_stats_block

MEMORY_REL = "Hermes/Memory/Auto"


def _vault(tmp: Path):
    vault = tmp / "vault"
    (vault / MEMORY_REL).mkdir(parents=True)
    (vault / "concepts").mkdir()
    (vault / "concepts" / "seed.md").write_text("# Seed\n", encoding="utf-8")
    store = ProposalStore(tmp / "proposals.sqlite", embedder=Embedder(mode="hash"))
    return vault, store


def _tick(store, vault):
    return maintain.run_maintenance(store, vault, Path(MEMORY_REL), write=True)


def test_tick_refreshes_a_stale_stats_block(tmp_path: Path):
    vault, store = _vault(tmp_path)
    page = vault / "Projects" / "proj.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "# Project\n\nHand-written prose that must survive.\n\n"
        f"{STATS_START}\nstale numbers\n{STATS_END}\n\n"
        "Trailing hand-written note.\n",
        encoding="utf-8")

    report = _tick(store, vault)

    text = page.read_text(encoding="utf-8")
    assert "stale numbers" not in text, "the tick did not refresh the block"
    assert "Hand-written prose that must survive." in text
    assert "Trailing hand-written note." in text
    assert has_stats_block(text)
    jobs = {job.name: job for job in report.jobs}
    assert "stats" in jobs, f"no stats job in the tick: {list(jobs)}"
    assert jobs["stats"].ran, f"stats job did not run: {jobs['stats'].skipped}"


def test_tick_never_touches_a_page_without_markers(tmp_path: Path):
    vault, store = _vault(tmp_path)
    page = vault / "concepts" / "handwritten.md"
    body = "# Hand written\n\npages: 16, chunks: 88, totally handwritten.\n"
    page.write_text(body, encoding="utf-8")

    _tick(store, vault)

    assert page.read_text(encoding="utf-8") == body, (
        "the tick modified a page that never opted in to generated stats")


def test_a_second_tick_writes_nothing(tmp_path: Path):
    """Idempotence: a settled vault must not churn on every tick."""
    vault, store = _vault(tmp_path)
    page = vault / "Projects" / "proj.md"
    page.parent.mkdir(parents=True)
    page.write_text(f"# P\n\n{STATS_START}\nx\n{STATS_END}\n", encoding="utf-8")

    _tick(store, vault)
    first = page.read_text(encoding="utf-8")
    mtime = page.stat().st_mtime_ns

    report = _tick(store, vault)
    second = page.read_text(encoding="utf-8")

    assert first == second, "the second tick produced different bytes"
    assert page.stat().st_mtime_ns == mtime, (
        "the second tick rewrote an already-current page")
    jobs = {job.name: job for job in report.jobs}
    assert jobs["stats"].count == 0, (
        f"second tick reported {jobs['stats'].count} rewrite(s)")


def test_dry_run_does_not_write_but_reports_the_work(tmp_path: Path):
    vault, store = _vault(tmp_path)
    page = vault / "Projects" / "proj.md"
    page.parent.mkdir(parents=True)
    page.write_text(f"# P\n\n{STATS_START}\nstale\n{STATS_END}\n", encoding="utf-8")
    before = page.read_text(encoding="utf-8")

    report = maintain.run_maintenance(store, vault, Path(MEMORY_REL), write=False)

    assert page.read_text(encoding="utf-8") == before, "a dry run wrote to the vault"
    jobs = {job.name: job for job in report.jobs}
    assert "would refresh" in " ".join(jobs["stats"].notes), jobs["stats"].notes


def test_measured_counts_exclude_the_tick_own_artifacts(tmp_path: Path):
    """A metric must not measure the file that writes it.

    Two rounds of self-invalidation were found by the idempotence test above:
    the maintenance report lands inside the memory directory (inflating the fact
    count) and at the vault root (inflating the page count), and a page carrying a
    stats block changes when the block is rewritten. Any of these makes the value
    depend on whether the job already ran, so it can never settle.
    """
    from hwiki.stats import measure_stats

    vault, store = _vault(tmp_path)
    page = vault / "Projects" / "proj.md"
    page.parent.mkdir(parents=True)
    page.write_text(f"# P\n\n{STATS_START}\nx\n{STATS_END}\n", encoding="utf-8")
    before = measure_stats(vault, memory_rel=Path(MEMORY_REL))

    _tick(store, vault)          # writes maintenance.md and rewrites the block
    after = measure_stats(vault, memory_rel=Path(MEMORY_REL))

    assert before["pages"] == after["pages"], (
        f"page count changed because the tick ran: {before['pages']} -> "
        f"{after['pages']}")
    assert before["facts"] == after["facts"], (
        f"fact count changed because the tick ran: {before['facts']} -> "
        f"{after['facts']}")
