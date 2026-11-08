"""Tests for autonomous maintenance (``hwiki.maintain``).

The tick is allowed to *find* work and to refresh derived artifacts. It is not
allowed to change what memory contains: no fact may be accepted, superseded, merged
or deleted by a tick. Several tests here exist only to pin that boundary.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import _hwiki_path  # noqa: F401,E402

from hwiki import maintain  # noqa: E402
from hwiki.facts import Embedder, ProposalStore  # noqa: E402

MEMORY_REL = "Hermes/Memory/Auto"


class _Vault(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.vault = root / "vault"
        self.memory = self.vault / MEMORY_REL
        self.memory.mkdir(parents=True)
        self.db = root / "proposals.sqlite"
        self.store = ProposalStore(str(self.db), embedder=Embedder(mode="hash"))
        (self.vault / "concepts").mkdir()
        (self.vault / "concepts" / "seed.md").write_text("# Seed\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def write_fact_page(self, slug, claim, kind="ops"):
        """A page shaped the way the extractor writes one (provenance matters:
        rebuild deliberately refuses hand-made pages rather than inventing it)."""
        (self.memory / f"{slug}.md").write_text(
            "---\n"
            f"title: {claim}\n"
            f"kind: {kind}\n"
            "status: accepted\n"
            "accepted_at: 2026-01-01\n"
            "extractor: hwiki-extractor\n"
            f"proposal_id: vault-test-{slug}\n"
            "peer: user\n"
            "---\n\n"
            f"# {claim}\n\n"
            "> **Evidence:** stated\n", encoding="utf-8")

    def accept(self, claim, kind="ops", peer="user"):
        proposal = self.store.propose({"kind": kind, "claim": claim, "evidence": "stated"},
                                      user_text="stated", session_id="s", peer=peer)
        try:
            self.store.accept(proposal["id"])
        except Exception:
            pass
        return proposal["id"]

    def tick(self, **kwargs):
        kwargs.setdefault("embedder", self.store.embedder)
        return maintain.run_maintenance(self.store, self.vault, Path(MEMORY_REL), **kwargs)


class JobTests(_Vault):
    def test_tick_reports_every_job(self):
        report = self.tick()
        names = [job.name for job in report.jobs]
        self.assertEqual(names, ["heal", "conflicts", "duplicates", "questions", "artifacts"])

    def test_negation_contradiction_is_found(self):
        self.accept("Deploys are allowed on Fridays")
        self.accept("Deploys are never allowed on Fridays")
        report = self.tick()
        self.assertEqual(len(report.conflicts), 1)
        self.assertIn("never", " ".join(report.conflicts[0][0:2]))

    def test_unrelated_facts_are_not_a_conflict(self):
        self.accept("The staging server stays on Fly.io")
        self.accept("The user takes notes in Obsidian daily")
        self.assertEqual(self.tick().conflicts, [])

    def test_duplicates_are_found_without_the_model(self):
        # two pages that say the same thing, imported from the vault rather than
        # proposed (the propose path would have deduplicated them)
        self.write_fact_page("backups-a", "Backups run nightly with restic to B2")
        self.write_fact_page("backups-b", "Backups run nightly with restic to B2")
        self.store.rebuild_from_vault(self.vault, Path(MEMORY_REL))
        report = self.tick()
        self.assertTrue(report.duplicates, "identical accepted facts must be listed")

    def test_heal_imports_pages_missing_from_the_index(self):
        self.write_fact_page("office-wifi", "The office wifi is eduroam", "environment")
        report = self.tick()
        heal = next(j for j in report.jobs if j.name == "heal")
        self.assertTrue(heal.ran)
        self.assertEqual(heal.count, 1)

    def test_stale_catalog_is_reported_and_regenerated(self):
        self.accept("Backups run nightly with restic to B2")
        (self.vault / "index.md").write_text("# Wiki Index\n\n- nothing\n", encoding="utf-8")
        report = self.tick()
        artifacts = next(j for j in report.jobs if j.name == "artifacts")
        self.assertTrue(report.stale_artifacts or artifacts.ran)


class BoundaryTests(_Vault):
    """A tick may not change what memory contains."""

    def test_no_fact_is_accepted_superseded_or_deleted(self):
        ids = [self.accept(c) for c in ("Deploys are allowed on Fridays",
                                        "Deploys are never allowed on Fridays")]
        before = {row["id"]: row["status"]
                  for row in self.store.list_proposals(status="accepted", limit=100)}
        before_proposed = len(self.store.list_proposals(status="proposed", limit=100))
        self.tick()
        after = {row["id"]: row["status"]
                 for row in self.store.list_proposals(status="accepted", limit=100)}
        self.assertEqual(before, after)
        self.assertEqual(before_proposed,
                         len(self.store.list_proposals(status="proposed", limit=100)))
        self.assertEqual(set(after), set(ids))

    def test_no_new_fact_pages_are_created(self):
        self.accept("Deploys are allowed on Fridays")
        self.accept("Deploys are never allowed on Fridays")
        before = sorted(p.name for p in self.memory.glob("*.md")
                        if p.name != maintain.MAINTENANCE_FILENAME)
        self.tick()
        after = sorted(p.name for p in self.memory.glob("*.md")
                       if p.name != maintain.MAINTENANCE_FILENAME)
        self.assertEqual(before, after)

    def test_the_report_page_is_not_recallable_as_a_fact(self):
        self.accept("Backups run nightly with restic to B2")
        self.tick()
        self.assertTrue((self.memory / maintain.MAINTENANCE_FILENAME).exists())
        hits = self.store.recall("what does the maintenance page say", limit=5)
        self.assertFalse([h for h in hits if "Generated by" in h.get("claim", "")])

    def test_dry_run_writes_nothing(self):
        report = self.tick(write=False)
        self.assertFalse(report.written)
        self.assertFalse((self.memory / maintain.MAINTENANCE_FILENAME).exists())
        self.assertEqual(self.store.get_meta(maintain.META_LAST_RUN), "")


class BudgetTests(_Vault):
    def test_a_spent_budget_skips_jobs_instead_of_overrunning(self):
        for i in range(40):
            self.accept(f"Fact number {i} about service {i} and its config")
        started = time.monotonic()
        report = self.tick(budget_ms=1)
        elapsed_ms = (time.monotonic() - started) * 1000
        self.assertLess(elapsed_ms, 3000, "a 1 ms budget must not take seconds")
        self.assertTrue(any(not job.ran for job in report.jobs))

    def test_budget_is_reported(self):
        self.assertEqual(self.tick(budget_ms=500).budget_ms, 500)


class QuestionTests(_Vault):
    def test_misses_are_recorded_and_deduplicated(self):
        buffer = maintain.QuestionBuffer()
        buffer.note("what is the user's favourite food?")
        buffer.note("What is the user's favourite food")
        buffer.note("when does the Lisbon trip start?")
        drained = buffer.drain()
        self.assertEqual(len(drained), 2)
        self.assertEqual(buffer.drain(), [])

    def test_statements_are_not_questions(self):
        buffer = maintain.QuestionBuffer()
        buffer.note("the user prefers dark mode")
        buffer.note("short")
        self.assertEqual(buffer.drain(), [])

    def test_questions_persist_into_meta_and_the_page(self):
        buffer = maintain.QuestionBuffer()
        buffer.note("what is the user's favourite food?")
        report = self.tick(question_buffer=buffer)
        self.assertEqual(report.questions, ["what is the user's favourite food?"])
        self.assertIn("what is the user's favourite food?",
                      Path(report.path).read_text(encoding="utf-8"))
        stored = json.loads(self.store.get_meta(maintain.META_OPEN_QUESTIONS))
        self.assertIn("what is the user's favourite food?", stored)

    def test_recall_of_nothing_feeds_the_buffer(self):
        from hwiki.facts import question_buffer

        shared = question_buffer()
        shared.drain()
        self.store.recall("what is the airspeed velocity of an unladen swallow?")
        self.assertIn("unladen swallow", " ".join(shared.drain()))

    def test_answered_questions_do_not_pollute_the_buffer(self):
        from hwiki.facts import question_buffer

        self.accept("Backups run nightly with restic to B2")
        shared = question_buffer()
        shared.drain()
        self.store.recall("what do backups run with")
        self.assertEqual(shared.drain(), [])

    def test_buffer_is_bounded(self):
        buffer = maintain.QuestionBuffer(limit=5)
        for i in range(40):
            buffer.note(f"what is unknown thing number {i}?")
        self.assertLessEqual(len(buffer.drain()), 5)


class ScheduleTests(_Vault):
    def test_due_when_never_run(self):
        self.assertTrue(maintain.due(self.store))

    def test_not_due_immediately_after_a_tick(self):
        self.tick()
        self.assertFalse(maintain.due(self.store))

    def test_due_after_the_interval(self):
        self.tick(now=datetime.now(timezone.utc) - timedelta(hours=48))
        self.assertTrue(maintain.due(self.store))

    def test_interval_zero_disables_the_tick(self):
        self.assertFalse(maintain.due(self.store, {"maintenance": {"interval_hours": 0}}))

    def test_interval_is_configurable(self):
        self.tick()
        self.assertTrue(maintain.due(self.store, {"maintenance": {"interval_hours": 0.0001}},
                                     now=datetime.now(timezone.utc) + timedelta(seconds=5)))

    def test_corrupt_timestamp_counts_as_due(self):
        self.store.set_meta(maintain.META_LAST_RUN, "not-a-date")
        self.assertTrue(maintain.due(self.store))


class EngineHookTests(_Vault):
    """The provider runs a tick on its own; this pins that it actually happens."""

    def _engine(self):
        from hwiki.config import DEFAULTS
        from hwiki.facts_provider import FactsEngine

        return FactsEngine(str(Path(self.tmp.name) / "home"), self.vault,
                           Path(MEMORY_REL), config=dict(DEFAULTS["facts"]))

    def test_hook_writes_the_page_when_a_tick_is_due(self):
        engine = self._engine()
        engine._maybe_maintain(self.store)
        self.assertTrue((self.memory / maintain.MAINTENANCE_FILENAME).exists())

    def test_hook_is_a_no_op_before_the_interval_elapses(self):
        engine = self._engine()
        engine._maybe_maintain(self.store)
        first = (self.memory / maintain.MAINTENANCE_FILENAME).read_text(encoding="utf-8")
        engine._maybe_maintain(self.store)
        second = (self.memory / maintain.MAINTENANCE_FILENAME).read_text(encoding="utf-8")
        self.assertEqual(first, second, "a second call inside the interval must not rewrite")

    def test_hook_never_raises_even_with_a_broken_store(self):
        engine = self._engine()

        class Broken:
            def get_meta(self, *_a, **_k):
                raise RuntimeError("db gone")

            def question_buffer(self):
                raise RuntimeError("db gone")

        engine._maybe_maintain(Broken())  # must not propagate

    def test_hook_flushes_questions_even_when_not_due(self):
        from hwiki.facts import question_buffer

        engine = self._engine()
        engine._maybe_maintain(self.store)          # sets last_run
        question_buffer().drain()
        question_buffer().note("what is the escalation policy for SEV1?")
        engine._maybe_maintain(self.store)          # not due, but the flush still runs
        stored = json.loads(self.store.get_meta(maintain.META_OPEN_QUESTIONS) or "[]")
        self.assertIn("what is the escalation policy for SEV1?", stored)


class GapHintTests(unittest.TestCase):
    """The recorded questions must actually reach the agent, or the job is theatre."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.wiki = root / "vault"
        (self.wiki / "Hermes" / "Memory" / "Auto").mkdir(parents=True)
        (self.wiki / "concepts").mkdir()
        (self.wiki / "concepts" / "seed.md").write_text("# Seed\n", encoding="utf-8")
        self.home = root / "home"
        self.home.mkdir()
        self.provider = self._provider()

    def tearDown(self):
        try:
            self.provider.shutdown()
        except Exception:
            pass
        self.tmp.cleanup()

    def _provider(self):
        from hwiki import WikiMemoryProvider

        provider = WikiMemoryProvider(config={
            "wiki_path": str(self.wiki), "memory_dir": "Hermes/Memory/Auto",
            "index_path": str(self.home / "index.sqlite"), "reindex_interval_s": 0,
            "_hermes_home": str(self.home), "embeddings": {"mode": "hash"},
            "facts": {"compile_enabled": False},
        })
        provider.initialize("s1", hermes_home=str(self.home), platform="cli")
        return provider

    def _record(self, questions):
        self.provider._facts.store().set_meta(maintain.META_OPEN_QUESTIONS,
                                              json.dumps(questions))

    def test_a_relevant_gap_is_raised_for_the_agent(self):
        self._record(["what is the escalation policy for SEV1 incidents?"])
        block = self.provider.prefetch("what is the escalation policy?")
        self.assertIn("escalation policy", block)
        self.assertIn("ask, do not assume", block)

    def test_an_unrelated_gap_stays_quiet(self):
        self._record(["what is the escalation policy for SEV1 incidents?"])
        self.assertEqual(self.provider.prefetch("what editor does the user prefer"), "")

    def test_it_is_a_question_not_an_assertion(self):
        self._record(["what is the user's favourite food?"])
        block = self.provider.prefetch("what is the user's favourite food?")
        self.assertIn("?", block)
        self.assertNotIn("Memory (recalled)\n\n- ", block)

    def test_no_gaps_means_no_block(self):
        self.assertEqual(self.provider.prefetch("what is the escalation policy?"), "")

    def test_a_corrupt_meta_value_does_not_break_prefetch(self):
        self.provider._facts.store().set_meta(maintain.META_OPEN_QUESTIONS, "{not json")
        self.assertEqual(self.provider.prefetch("what is the escalation policy?"), "")


class RenderTests(_Vault):
    def test_page_carries_the_required_frontmatter(self):
        report = self.tick()
        text = Path(report.path).read_text(encoding="utf-8")
        for field in ("type: maintenance", "status: generated", "source: hwiki"):
            self.assertIn(field, text)
        self.assertNotIn("\n# claim\n", text)

    def test_empty_report_reads_as_empty_not_as_success(self):
        report = self.tick()
        text = Path(report.path).read_text(encoding="utf-8")
        self.assertIn("None found.", text)
        self.assertEqual(report.work_items, 0)

    def test_report_json_is_stable(self):
        payload = self.tick().to_dict()
        self.assertEqual(set(payload), {"elapsed_ms", "budget_ms", "ran_at", "jobs",
                                        "work_items", "conflicts", "duplicates",
                                        "questions", "stale_artifacts", "written",
                                        "path"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
