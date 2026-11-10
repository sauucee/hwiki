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

from hwiki import catalog, maintain  # noqa: E402
from hwiki.store import WikiStore  # noqa: E402
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
        self.assertEqual(names, ["heal", "conflicts", "duplicates", "questions",
                                 "artifacts", "stats"])

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
        self.write_fact_page("backups", "Backups run nightly with restic to B2")
        (self.vault / "index.md").write_text("# Wiki Index\n\n- [[gone-page]]\n",
                                           encoding="utf-8")
        report = self.tick()
        # ``artifacts.ran`` is true even for "catalogs current", so assert the finding.
        self.assertTrue(report.stale_artifacts, "a stale catalog must be reported as work")

    def test_ambiguous_catalog_entry_is_work_for_the_artifacts_job(self):
        """A catalog link the resolver refuses to resolve is stale too — regenerate it.

        The lint check that reports this is newer than the artifacts job, which only
        looked for ``catalog.stale``/``catalog.entry_missing``; an ambiguous entry was
        therefore invisible and the tick said "catalogs current" while the index kept
        pointing at a target no reader can open.
        """
        self.write_fact_page("backups", "Backups run nightly with restic to B2")
        (self.vault / "concepts" / "backups.md").write_text(
            "# Backups\n\nA curated page.\n", encoding="utf-8")
        (self.vault / "concepts" / "Backups.md").write_text(
            "# BACKUPS\n\nSame folder, same name, other case.\n", encoding="utf-8")
        (self.vault / "index.md").write_text(
            "# Wiki Index\n\n"
            "- [[concepts/seed|Seed]]\n"
            "- [[concepts/backups|backups]]\n"
            "- [[concepts/Backups|Backups]]\n"
            f"- [[{MEMORY_REL}/backups|backups]]\n"
            "- [[concepts/BACKUPS]] — a stale link that names both case twins\n",
            encoding="utf-8")
        report = self.tick()
        self.assertTrue(any("catalog.ambiguous" in str(item)
                            for item in report.stale_artifacts),
                        report.stale_artifacts)
        # isolation: nothing here is missing, so ambiguity alone must be the trigger
        self.assertFalse(any("entry_missing" in str(item) or "stale:" in str(item)
                             for item in report.stale_artifacts),
                         report.stale_artifacts)


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


class ConflictHintTests(GapHintTests):
    """A stored disagreement must reach the agent, and never be resolved by it."""

    def _record_conflict(self, a="Deploys are allowed on Fridays",
                         b="Deploys are never allowed on Fridays"):
        self.provider._facts.store().set_meta(maintain.META_CONFLICTS, json.dumps(
            [{"a": a, "b": b, "score": 1.0, "at": "2026-11-08T00:00:00+00:00"}]))

    def test_a_relevant_disagreement_shows_both_sides(self):
        self._record_conflict()
        block = self.provider.prefetch("are deploys allowed on fridays?")
        self.assertIn("Deploys are allowed on Fridays", block)
        self.assertIn("Deploys are never allowed on Fridays", block)
        self.assertIn("ask which is right", block)

    def test_it_does_not_pick_a_side(self):
        self._record_conflict()
        block = self.provider.prefetch("are deploys allowed on fridays?")
        self.assertIn("Neither has been changed", block)
        # no verb that would present one claim as the answer
        for word in ("recommend", "should use", "prefer the", "correct one"):
            self.assertNotIn(word, block.lower())

    def test_an_unrelated_query_is_not_warned(self):
        self._record_conflict()
        self.assertEqual(self.provider.prefetch("which editor do they use"), "")

    def test_a_corrupt_conflict_list_does_not_break_prefetch(self):
        self.provider._facts.store().set_meta(maintain.META_CONFLICTS, "not json at all")
        self.assertEqual(self.provider.prefetch("are deploys allowed on fridays?"), "")

    def test_malformed_entries_are_ignored(self):
        self.provider._facts.store().set_meta(maintain.META_CONFLICTS, json.dumps(
            [{"a": None, "b": "x"}, {"a": "Deploys are allowed on Fridays"},
             "not a dict"]))
        self.assertEqual(self.provider.prefetch("deploys allowed"), "")

    def test_gaps_and_disagreements_can_both_appear(self):
        self._record(["when are deploys allowed on fridays?"])
        self._record_conflict()
        block = self.provider.prefetch("are deploys allowed on fridays?")
        self.assertIn("Memory gaps", block)
        self.assertIn("disagree", block)


class DuplicateHintTests(GapHintTests):
    """A fact already recorded twice must not get a third copy silently."""

    def _record_duplicate(self):
        self.provider._facts.store().set_meta(maintain.META_DUPLICATES, json.dumps([
            {"a": "Backups run nightly with restic to B2",
             "b": "Backups run nightly with restic into B2",
             "score": 0.93, "at": "2026-11-08T00:00:00+00:00"}]))

    def test_a_relevant_duplicate_is_raised(self):
        self._record_duplicate()
        block = self.provider.prefetch("when do backups run with restic?")
        self.assertIn("recorded twice", block)
        self.assertIn("Backups run nightly with restic to B2", block)

    def test_it_warns_against_a_third_copy(self):
        self._record_duplicate()
        block = self.provider.prefetch("when do backups run with restic?")
        self.assertIn("third copy", block)

    def test_an_unrelated_query_stays_quiet(self):
        self._record_duplicate()
        self.assertEqual(self.provider.prefetch("which editor do they use"), "")

    def test_conflicts_and_duplicates_do_not_hide_each_other(self):
        self.provider._facts.store().set_meta(maintain.META_DUPLICATES, json.dumps([
            {"a": "Backups run nightly with restic to B2",
             "b": "Backups run nightly with restic into B2", "score": 0.93, "at": "x"}]))
        self.provider._facts.store().set_meta(maintain.META_CONFLICTS, json.dumps([
            {"a": "Deploys are allowed on Fridays",
             "b": "Deploys are never allowed on Fridays", "score": 1.0, "at": "x"}]))
        dup = self.provider.prefetch("do backups run with restic nightly?")
        con = self.provider.prefetch("are deploys allowed on fridays?")
        self.assertIn("recorded twice", dup)
        self.assertIn("disagree", con)
        self.assertNotIn("disagree", dup)


class TokenBudgetTests(GapHintTests):
    """Prefetch must respect a prompt budget, not just a hit count."""

    def _provider_with(self, tokens):
        from hwiki import WikiMemoryProvider

        provider = WikiMemoryProvider(config={
            "wiki_path": str(self.wiki), "memory_dir": "Hermes/Memory/Auto",
            "index_path": str(self.home / "index.sqlite"), "reindex_interval_s": 0,
            "_hermes_home": str(self.home), "embeddings": {"mode": "hash"},
            "facts": {"compile_enabled": False}, "max_context_tokens": tokens,
        })
        provider.initialize("s1", hermes_home=str(self.home), platform="cli")
        return provider

    def test_low_priority_blocks_are_dropped_first(self):
        provider = self._provider_with(60)
        try:
            blocks = [(3, "w" * 2000), (0, "f" * 80)]
            kept = provider._apply_token_budget(blocks)
            self.assertEqual([text[:1] for _p, text in kept], ["f"])
        finally:
            provider.shutdown()

    def test_whole_blocks_are_never_cut(self):
        provider = self._provider_with(60)
        try:
            block = "f" * 80
            kept = provider._apply_token_budget([(0, block)])
            self.assertEqual(kept[0][1], block)
        finally:
            provider.shutdown()

    def test_zero_disables_the_cap(self):
        provider = self._provider_with(0)
        try:
            blocks = [(0, "f" * 80), (3, "w" * 4000)]
            self.assertEqual(len(provider._apply_token_budget(blocks)), 2)
        finally:
            provider.shutdown()

    def test_the_injected_block_fits_the_budget(self):
        provider = self._provider_with(90)
        try:
            store = provider._facts.store()
            store.set_meta(maintain.META_CONFLICTS, json.dumps([{
                "a": "Deploys are allowed on Fridays",
                "b": "Deploys are never allowed on Fridays", "score": 1.0, "at": "x"}]))
            block = provider.prefetch("are deploys allowed on fridays?")
            self.assertLessEqual((len(block) + 3) // 4, 90 + 40)  # header overhead
        finally:
            provider.shutdown()

    def test_a_short_query_still_gets_its_hint_under_a_tight_budget(self):
        provider = self._provider_with(40)
        try:
            provider._facts.store().set_meta(maintain.META_OPEN_QUESTIONS, json.dumps(
                ["when are deploys allowed on fridays?"]))
            block = provider.prefetch("are deploys allowed on fridays?")
            self.assertIn("Memory gaps", block)
        finally:
            provider.shutdown()


class ReportExclusionTests(GapHintTests):
    """The generated report is an artifact; it must never be recalled as knowledge."""

    def setUp(self):
        super().setUp()
        (self.wiki / "Hermes" / "Memory" / "Auto" / maintain.MAINTENANCE_FILENAME).write_text(
            "---\ntitle: Memory maintenance\ntype: maintenance\n---\n\n"
            "# Memory maintenance\n\nDeploys are allowed on Fridays vs never allowed.\n",
            encoding="utf-8")
        self.provider.shutdown()
        self.provider = self._provider()   # rebuild so the index sees the page

    def test_the_report_never_reaches_recall(self):
        block = self.provider.prefetch("deploys allowed fridays")
        self.assertNotIn(maintain.MAINTENANCE_FILENAME, block)

    def test_excluded_paths_covers_the_report_once(self):
        from hwiki.config import excluded_paths

        paths = excluded_paths({"memory_dir": "Hermes/Memory/Auto",
                                "exclude": ["raw/*"]})
        self.assertIn("Hermes/Memory/Auto/maintenance.md", paths)
        self.assertIn("raw/*", paths)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(paths.count("Hermes/Memory/Auto/maintenance.md"), 1)

    def test_a_users_own_maintenance_page_elsewhere_still_works(self):
        from hwiki.config import excluded_paths

        paths = excluded_paths({"memory_dir": "wiki/memory"})
        self.assertNotIn("notes/maintenance.md", paths)


class ConflictPersistenceTests(_Vault):
    """The tick's findings must reach the read path, and stop when they are resolved."""

    def _tick(self, *, write=True):
        return maintain.run_maintenance(self.store, self.vault, Path(MEMORY_REL),
                                        embedder=self.store.embedder, write=write,
                                        question_buffer=maintain.QuestionBuffer())

    def test_found_pairs_are_written_to_meta(self):
        self.accept("Deploys are allowed on Fridays")
        self.accept("Deploys are never allowed on Fridays")
        report = self._tick()
        self.assertEqual(len(report.conflicts), 1)
        stored = json.loads(self.store.get_meta(maintain.META_CONFLICTS))
        self.assertEqual(len(stored), 1)
        self.assertIn("Deploys are allowed on Fridays", (stored[0]["a"], stored[0]["b"]))

    def test_a_resolved_pair_stops_being_reported(self):
        self.accept("Deploys are allowed on Fridays")
        self.accept("Deploys are never allowed on Fridays")
        self._tick()
        self.assertTrue(json.loads(self.store.get_meta(maintain.META_CONFLICTS)))
        # a human supersedes one side; the next tick must forget the pair
        victim = self.store.list_proposals(status="accepted", limit=1)[0]
        survivor = self.store.list_proposals(status="accepted", limit=10)[1]
        self.store.supersede(survivor["id"], victim["id"])
        self._tick()
        self.assertEqual(json.loads(self.store.get_meta(maintain.META_CONFLICTS)), [])

    def test_dry_run_writes_no_meta(self):
        self.accept("Deploys are allowed on Fridays")
        self.accept("Deploys are never allowed on Fridays")
        self._tick(write=False)
        self.assertFalse(self.store.get_meta(maintain.META_CONFLICTS))


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



class DryRunWritesNothingTests(unittest.TestCase):
    """`--dry-run` is documented as "nothing written" — assert that on every job.

    A live dry run once regenerated the vault's index.md and rebuilt the facts DB: the
    `write` flag only gated the meta bookkeeping, not the jobs themselves.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.mem = self.root / "mem"
        self.mem.mkdir()
        self.store = ProposalStore(self.root / "facts.sqlite",
                                  embedder=Embedder(mode="hash"))

    def tearDown(self):
        self._tmp.cleanup()

    def _vault_page(self, name, text):
        path = self.mem / name
        path.write_text(
            f"---\ntitle: {name}\n---\n\n# {name}\n\n{text}\n", encoding="utf-8")
        return path

    def _snapshot(self):
        return {p.name: (p.stat().st_mtime_ns, p.read_bytes())
                for p in self.mem.glob("*.md")}

    def _tick(self, **kwargs):
        return maintain.run_maintenance(self.store, self.root, "mem", **kwargs)

    def test_a_dry_run_does_not_rewrite_the_index(self):
        self._vault_page("fact-a.md", "Staging stays on Fly.io.")
        catalog.write_index(WikiStore(self.root, memory_dir="mem"), memory_dir="mem")
        index = self.root / "index.md"
        index.write_text(
            "---\ntitle: Wiki Index\nsource: hwiki-catalog\n---\n\n# Wiki Index\n\n"
            "- [[gone-page]]\n", encoding="utf-8")
        before = (index.stat().st_mtime_ns, index.read_bytes())
        self._tick(write=False)
        self.assertEqual((index.stat().st_mtime_ns, index.read_bytes()), before)

    def test_a_dry_run_says_what_it_would_do(self):
        self._vault_page("fact-a.md", "Staging stays on Fly.io.")
        catalog.write_index(WikiStore(self.root, memory_dir="mem"), memory_dir="mem")
        (self.root / "index.md").write_text(
            "---\ntitle: Wiki Index\nsource: hwiki-catalog\n---\n\n# Wiki Index\n\n"
            "- [[gone-page]]\n", encoding="utf-8")
        report = self._tick(write=False)
        note = " ".join(n for job in report.jobs for n in job.notes)
        self.assertIn("would regenerate", note)

    def test_a_dry_run_does_not_heal_the_database(self):
        self._vault_page("fact-a.md", "Staging stays on Fly.io.")
        before = self.store.list_proposals(status="accepted", limit=100)
        report = self._tick(write=False)
        self.assertEqual(self.store.list_proposals(status="accepted", limit=100), before)
        note = " ".join(n for job in report.jobs for n in job.notes)
        self.assertIn("would recover", note)

    def test_a_dry_run_leaves_vault_pages_untouched(self):
        page = self._vault_page("fact-a.md", "Staging stays on Fly.io.")
        before = self._snapshot()
        self._tick(write=False)
        self.assertEqual(self._snapshot(), before)
        self.assertTrue(page.exists())

    def test_a_dry_run_does_not_flush_open_questions(self):
        buffer = maintain.QuestionBuffer()
        buffer.note("what is the deploy window")
        self._tick(write=False, question_buffer=buffer)
        self.assertIn("what is the deploy window", buffer.drain())

    def test_a_real_tick_does_write(self):
        self._vault_page("fact-a.md", "Staging stays on Fly.io.")
        catalog.write_index(WikiStore(self.root, memory_dir="mem"), memory_dir="mem")
        index = self.root / "index.md"
        index.write_text(
            "---\ntitle: Wiki Index\nsource: hwiki-catalog\n---\n\n# Wiki Index\n\n"
            "- [[gone-page]]\n", encoding="utf-8")
        self._tick(write=True)
        self.assertNotIn("gone-page", index.read_text(encoding="utf-8"))


class ClockSkewTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ProposalStore(Path(self._tmp.name) / "f.sqlite",
                                  embedder=Embedder(mode="hash"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_future_timestamp_does_not_disable_the_tick(self):
        """A clock that jumped forward once must not wedge maintenance until 2099."""
        self.store.set_meta(maintain.META_LAST_RUN, "2099-01-01T00:00:00+00:00")
        self.assertTrue(maintain.due(self.store, {}))

    def test_a_recent_timestamp_is_still_not_due(self):
        from datetime import datetime, timezone
        self.store.set_meta(maintain.META_LAST_RUN,
                            datetime.now(timezone.utc).isoformat())
        self.assertFalse(maintain.due(self.store, {}))



class ArtifactWriteTests(_Vault):
    """The artifacts job must converge: one regeneration, then nothing to do.

    Ambiguity is now a trigger. A hand-written index (no hwiki marker) is still never
    overwritten, so the tick must say so and leave the bytes alone instead of claiming
    it regenerated the catalog.
    """

    def test_regenerating_an_ambiguous_catalog_converges(self):
        """One regeneration fixes it, and the next tick has nothing to do.

        The tick writes `maintenance.md` after the artifacts job, so the index it just
        regenerated cannot list that page. Treating it as a page the catalog must
        contain made every later tick find an "entry_missing" and rewrite the index
        forever — a derived write per tick that changed nothing.
        """
        self.write_fact_page("backups", "Backups run nightly with restic to B2")
        (self.vault / "concepts" / "backups.md").write_text(
            "# Backups\n\nA curated page that shares the filename.\n", encoding="utf-8")
        (self.vault / "index.md").write_text(
            "---\ntitle: Wiki Index\ntype: index\nsource: hwiki-catalog\n---\n\n"
            "# Wiki Index\n\n- [[backups]] — names both pages\n", encoding="utf-8")
        first = self.tick()
        self.assertTrue(first.stale_artifacts, first.stale_artifacts)
        index = (self.vault / "index.md").read_text(encoding="utf-8")
        self.assertNotIn("[[backups]]", index)
        self.assertIn("[[concepts/backups|backups]]", index)
        second = self.tick()
        self.assertFalse(second.stale_artifacts, second.stale_artifacts)

    def test_a_hand_written_index_is_never_rewritten(self):
        self.write_fact_page("backups", "Backups run nightly with restic to B2")
        (self.vault / "concepts" / "backups.md").write_text(
            "# Backups\n\nA curated page.\n", encoding="utf-8")
        (self.vault / "concepts" / "Backups.md").write_text(
            "# BACKUPS\n\nSame folder, same name, other case.\n", encoding="utf-8")
        original = "# Wiki Index\n\n- [[concepts/BACKUPS]] — names both case twins\n"
        (self.vault / "index.md").write_text(original, encoding="utf-8")
        report = self.tick()
        self.assertTrue(any("catalog.ambiguous" in str(item)
                            for item in report.stale_artifacts), report.stale_artifacts)
        self.assertEqual((self.vault / "index.md").read_text(encoding="utf-8"), original)
        notes = " ".join(n for job in report.jobs for n in job.notes)
        self.assertIn("written by hand", notes)
    def test_an_unlinkable_page_does_not_cause_a_write_per_tick(self):
        self.write_fact_page("backups", "Backups run nightly with restic to B2")
        (self.vault / "concepts" / "we|ird.md").write_text(
            "# Weird\n", encoding="utf-8")
        first = self.tick()
        self.assertTrue(first.written)
        second = self.tick()
        self.assertFalse(second.stale_artifacts, second.stale_artifacts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
