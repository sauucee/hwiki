"""Read-only ingest planning.

The property under test is not "does it find good claims" — the splitter is
deliberately naive — but **"does it write anything"**. An ingest planner that
quietly proposed its own rows would be an unreviewed ingestion pipeline wearing a
preview's clothes, which is exactly what the roadmap forbids.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _hwiki_path  # noqa: F401,E402

from hwiki.facts import Embedder, ProposalStore, write_fact_page  # noqa: E402
from hwiki.plan import (  # noqa: E402
    CONFLICT, DUPLICATE, MERGE, NEW_PAGE, build_plan, candidate_claims, format_plan,
)

MEMORY_REL = Path("Hermes/Memory/Auto")


class CandidateClaimsTests(unittest.TestCase):
    def test_sentences_are_split(self):
        claims = candidate_claims(
            "Staging runs on Fly.io. Deploys are frozen on Fridays.")
        self.assertIn("Staging runs on Fly.io.", claims)
        self.assertIn("Deploys are frozen on Fridays.", claims)

    def test_bullets_are_unwrapped(self):
        claims = candidate_claims("- Staging runs on Fly.io\n- Backups use restic")
        self.assertIn("Staging runs on Fly.io", claims)
        self.assertIn("Backups use restic", claims)

    def test_markdown_noise_is_dropped(self):
        claims = candidate_claims("---\n# Heading\n| table |\nReal claim here.")
        self.assertEqual(claims, ["Real claim here."])

    def test_too_short_is_dropped(self):
        self.assertEqual(candidate_claims("ok.\nyes.\nno."), [])

    def test_duplicates_are_collapsed(self):
        claims = candidate_claims("Staging runs on Fly.io. Staging runs on Fly.io.")
        self.assertEqual(len(claims), 1)


class BuildPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.vault = self.root / "vault"
        (self.vault / MEMORY_REL).mkdir(parents=True)
        self.store = ProposalStore(self.root / "facts.sqlite",
                                   embedder=Embedder(mode="hash"))
        for claim in ("Staging server stays on Fly.io",
                      "Backups run nightly with restic to B2"):
            p = self.store.propose({"kind": "environment", "claim": claim,
                                    "evidence": "stated"}, user_text="stated",
                                   session_id="s")
            self.store.accept(p["id"])
            page = write_fact_page(self.vault, MEMORY_REL, self.store.get(p["id"]))
            self.store.set_page_path(p["id"], page)

    def tearDown(self):
        self.tmp.cleanup()

    # -- the property that matters ---------------------------------------

    def test_plan_writes_nothing(self):
        db = self.root / "facts.sqlite"
        before_rows = len(self.store.list_proposals(status="accepted", limit=1000))
        before_pages = sorted(p.name for p in (self.vault / MEMORY_REL).glob("*.md"))
        before_bytes = db.read_bytes()
        before_proposed = len(self.store.list_proposals(status="proposed", limit=1000))

        build_plan(self.store, ["Staging moves to Hetzner next month.",
                                "Deploys are frozen on Fridays."])

        self.assertEqual(len(self.store.list_proposals(status="accepted", limit=1000)),
                         before_rows)
        self.assertEqual(len(self.store.list_proposals(status="proposed", limit=1000)),
                         before_proposed, "a plan must not create proposal rows")
        self.assertEqual(sorted(p.name for p in (self.vault / MEMORY_REL).glob("*.md")),
                         before_pages, "a plan must not create pages")
        self.assertEqual(db.read_bytes(), before_bytes,
                         "a plan must not modify the facts database")

    def test_plan_reports_zero_writes_in_its_payload(self):
        plan = build_plan(self.store, ["Staging moves to Hetzner next month."])
        self.assertEqual(plan.writes_performed, 0)
        self.assertEqual(plan.to_dict()["writes_performed"], 0)

    # -- predictions ------------------------------------------------------

    def test_near_duplicate_is_predicted(self):
        plan = build_plan(self.store, ["Staging server stays on Fly.io"])
        self.assertEqual(plan.entries[0].action, DUPLICATE)
        self.assertIn("Fly.io", plan.entries[0].duplicate_of)

    def test_unrelated_claim_is_a_new_page(self):
        plan = build_plan(self.store, ["The user prefers aisle seats on flights"])
        self.assertIn(plan.entries[0].action, (NEW_PAGE, MERGE))

    def test_contradiction_is_flagged_and_never_merged(self):
        plan = build_plan(self.store, ["Staging server does not stay on Fly.io"])
        self.assertEqual(plan.entries[0].action, CONFLICT)
        self.assertTrue(plan.entries[0].conflicts_with)

    def test_merge_names_the_owning_page(self):
        plan = build_plan(self.store, ["Staging server stays on Fly.io and runs "
                                       "the queue worker in the same region"])
        self.assertIn(plan.entries[0].action, (MERGE, DUPLICATE, NEW_PAGE))
        if plan.entries[0].action == MERGE:
            self.assertTrue(plan.entries[0].owner_slug.startswith("fact-"))
            self.assertNotIn("/", plan.entries[0].owner_slug)

    def test_secret_shaped_candidate_is_refused_not_planned(self):
        plan = build_plan(self.store, [
            "Set the deploy key with token=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"])
        self.assertEqual(plan.entries[0].action, "refused_secret")
        self.assertEqual(plan.skipped, 1)

    def test_plan_is_deterministic(self):
        claims = ["Backups run nightly.", "Staging server stays on Fly.io"]
        first = format_plan(build_plan(self.store, claims))
        second = format_plan(build_plan(self.store, claims))
        self.assertEqual(first, second)

    def test_format_names_the_read_only_guarantee(self):
        text = format_plan(build_plan(self.store, ["Backups run nightly."]))
        self.assertIn("writes performed: 0", text)

    def test_empty_input_is_reported_not_crashed(self):
        plan = build_plan(self.store, [])
        self.assertEqual(plan.entries, [])
        self.assertIn("no candidate claims", format_plan(plan))


if __name__ == "__main__":
    unittest.main(verbosity=2)
