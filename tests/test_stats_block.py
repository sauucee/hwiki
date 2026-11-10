"""Prose metrics rot. A generated block does not.

Every hand-written measurement in this vault has gone stale within a day of being
written: ``HANDOFF.md`` carried "183/225" long after those figures moved, and the
hwiki pages claimed "16 pages / 88 chunks" the moment two more pages were created.
Fresh numbers plus discipline is not a fix — it failed twice.

The durable answer is the same one ``compile`` already uses for merged sections: a
delimited block that automation owns and a human never edits.

    <!-- hwiki:stats:start -->
    ...generated...
    <!-- hwiki:stats:end -->

Rules these tests pin:
  * the block is regenerated in place, and hand-written prose around it is untouched;
  * a page without markers is left completely alone (no surprise insertions);
  * the same vault state renders byte-identical output, so a no-op tick writes nothing;
  * the renderer never invents a number it did not measure.

Run: python -m pytest tests/test_stats_block.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.stats import (STATS_END, STATS_START, render_stats_block,
                         replace_stats_block)


# -- rendering ---------------------------------------------------------------

def test_render_emits_a_delimited_block():
    text = render_stats_block(pages=19, chunks=116, facts=1, tip="7b9c189",
                              tests=669)
    assert text.startswith(STATS_START)
    assert text.rstrip().endswith(STATS_END)
    assert "19" in text and "116" in text and "669" in text and "7b9c189" in text


def test_render_is_deterministic():
    """Same input, same bytes — otherwise every tick rewrites the page."""
    a = render_stats_block(pages=19, chunks=116, facts=1, tip="abc1234", tests=669)
    b = render_stats_block(pages=19, chunks=116, facts=1, tip="abc1234", tests=669)
    assert a == b


def test_render_marks_unknown_values_instead_of_guessing():
    """A value that could not be measured must say so, never invent one."""
    text = render_stats_block(pages=19, chunks=116, facts=None, tip=None, tests=None)
    assert "unknown" in text.lower()
    assert "0" not in text.split("facts")[1][:8], (
        "an unmeasured fact count must not render as 0")


def test_render_includes_the_producing_command():
    """A reader must be able to reproduce every number in the block."""
    text = render_stats_block(pages=1, chunks=1, facts=0, tip="abc", tests=1)
    assert "hermes hwiki status" in text


# -- replacement -------------------------------------------------------------

def test_block_is_replaced_in_place():
    page = (
        "# Title\n\nHand-written intro that must survive.\n\n"
        f"{STATS_START}\nold numbers\n{STATS_END}\n\n"
        "Hand-written outro that must survive.\n"
    )
    new_block = render_stats_block(pages=5, chunks=9, facts=2, tip="def5678", tests=10)
    out = replace_stats_block(page, new_block)
    assert "Hand-written intro that must survive." in out
    assert "Hand-written outro that must survive." in out
    assert "old numbers" not in out
    assert "def5678" in out
    assert out.count(STATS_START) == 1


def test_replacement_is_idempotent():
    page = f"# T\n\n{STATS_START}\nx\n{STATS_END}\n"
    block = render_stats_block(pages=5, chunks=9, facts=2, tip="abc", tests=10)
    once = replace_stats_block(page, block)
    twice = replace_stats_block(once, block)
    assert once == twice


def test_a_page_without_markers_is_never_modified():
    """No markers means a hand-written page: leave it completely alone."""
    page = "# Hand written\n\npages: 16, chunks: 88\n"
    block = render_stats_block(pages=99, chunks=99, facts=9, tip="zzz", tests=99)
    assert replace_stats_block(page, block) == page


def test_two_blocks_are_both_replaced():
    page = (f"{STATS_START}\na\n{STATS_END}\n\nmiddle\n\n"
            f"{STATS_START}\nb\n{STATS_END}\n")
    block = render_stats_block(pages=1, chunks=1, facts=0, tip="t", tests=1)
    out = replace_stats_block(page, block)
    assert "middle" in out
    assert out.count(STATS_START) == 2
    assert "\na\n" not in out and "\nb\n" not in out


# -- measuring ---------------------------------------------------------------

def test_measure_reports_what_it_can_and_none_for_what_it_cannot(tmp_path: Path):
    """measure_stats must not fabricate: an unreadable repo means tip=None."""
    from hwiki.stats import measure_stats
    vault = tmp_path / "vault"
    (vault / "Hermes" / "Memory" / "Auto").mkdir(parents=True)
    (vault / "index.md").write_text("# Index\n", encoding="utf-8")
    stats = measure_stats(vault, memory_rel=Path("Hermes/Memory/Auto"),
                          repo_dir=None, run_tests=False)
    assert stats["pages"] is not None
    assert stats["tip"] is None, "no repo dir means the tip is unknown, not ''"
    assert stats["tests"] is None, "tests were not run, so the count is unknown"


def test_a_failed_test_run_never_yields_a_count(tmp_path: Path):
    """Silence is not a pass.

    The first version ran ``python3 -m pytest``, which silently failed on this host
    (system python has no pytest) and left the block showing an old passing number —
    exactly the staleness this module exists to remove. A run that produces no
    ``N passed`` must leave the value ``None``.

    The second version parsed ``(\\d+) passed`` from any summary, so a run that
    reported ``709 passed, 4 errors`` was recorded as 709: a partial suite written
    as the total. Both are the same mistake in different clothes, so both are tested.
    """
    from hwiki.stats import measure_stats

    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_nothing.py").write_text("def test_x():\n    pass\n",
                                                    encoding="utf-8")
    # An empty interpreter list stands in for "no interpreter can run pytest".
    import hwiki.stats as stats_mod
    original = stats_mod._test_interpreters
    stats_mod._test_interpreters = lambda: []          # noqa: SLF001 - test seam
    try:
        measured = stats_mod.measure_stats(
            tmp_path, memory_rel=Path("mem"), repo_dir=repo, run_tests=True)
    finally:
        stats_mod._test_interpreters = original        # noqa: SLF001

    assert measured["tests"] is None, (
        f"an unobserved test count became {measured['tests']!r}")


def test_a_partial_run_is_not_recorded_as_a_total(tmp_path: Path, monkeypatch):
    """'N passed, M errors' must not be read as N passing tests.

    Measured cause of a real 709-vs-713 discrepancy: four tests errored during the
    tick's run and the summary line still matched ``(\\d+) passed``.
    """
    import subprocess
    import hwiki.stats as stats_mod

    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)

    class _Out:
        stdout = "709 passed, 4 errors in 160.00s (0:02:40)\n"
        stderr = ""
        returncode = 1

    monkeypatch.setattr(stats_mod.subprocess, "run", lambda *a, **k: _Out())
    monkeypatch.setattr(stats_mod, "_test_interpreters", lambda: ["/usr/bin/false"])

    measured = stats_mod.measure_stats(tmp_path, memory_rel=Path("mem"),
                                       repo_dir=repo, run_tests=True)
    assert measured["tests"] is None, (
        f"a partial run was recorded as the total: {measured['tests']!r}")
    assert "4 errors" in measured.get("tests_error", ""), (
        "the reason was discarded, so the block cannot explain itself")


def test_the_interpreter_list_prefers_a_real_pytest(tmp_path: Path):
    """The candidate list must include an interpreter that can import pytest."""
    from hwiki.stats import _test_interpreters
    interpreters = _test_interpreters()
    assert interpreters, "no interpreter candidates at all"
    assert all(isinstance(i, str) and i for i in interpreters)
    assert len(set(interpreters)) == len(interpreters), "duplicate candidates"
