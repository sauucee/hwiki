"""Phase 2 — provenance and the reversible lifecycle, end to end.

The vault is the source of truth, so every one of these must hold:

* a hand edit in the vault becomes the fact's claim (and its embedding is updated,
  so recall finds the *new* text);
* deleting a page revokes the fact instead of leaving a ghost in recall;
* a superseded fact stays superseded across a rebuild — it must never be
  resurrected as current;
* a lost/fresh database is rebuilt from the vault alone;
* provenance (kind, peer, project, accepted date) survives that rebuild;
* a secret-shaped hand edit is refused while the original fact is kept.

All stores and vaults are throwaway. The live vault and live facts DB are never
opened.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _hwiki_path  # noqa: F401,E402

from hwiki.facts import (  # noqa: E402
    Embedder,
    ProposalStore,
    mark_superseded_page,
    read_fact_pages,
    write_fact_page,
)

MEMORY_REL = Path("Hermes/Memory/Auto")


class ProvenanceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.vault = self.root / "vault"
        (self.vault / MEMORY_REL).mkdir(parents=True)
        self.stores = []

    def tearDown(self):
        for store in self.stores:
            try:
                store.close()
            except Exception:
                pass
        self.tmp.cleanup()

    # -- helpers ---------------------------------------------------------

    def store(self, name="facts.sqlite"):
        st = ProposalStore(str(self.root / name), embedder=Embedder(mode="hash"))
        self.stores.append(st)
        return st

    def accept(self, store, claim, kind="preference", evidence="stated",
               project="", peer="user"):
        p = store.propose({"kind": kind, "claim": claim, "evidence": evidence},
                          user_text=evidence, session_id="s",
                          project=project, peer=peer)
        store.accept(p["id"])
        row = store.get(p["id"])
        page = write_fact_page(self.vault, MEMORY_REL, row)
        store.set_page_path(p["id"], page)
        return p["id"], Path(page)

    # -- edit ------------------------------------------------------------

    def test_hand_edit_updates_claim_and_embedding(self):
        store = self.store()
        pid, page = self.accept(store, "User's main editor is Sublime")

        text = page.read_text(encoding="utf-8")
        self.assertIn("# User's main editor is Sublime", text)
        page.write_text(text.replace("# User's main editor is Sublime",
                                     "# User's main editor is VS Code"),
                        encoding="utf-8")

        summary = store.sync_from_vault()
        self.assertEqual(summary["updated"], 1)

        row = store.get(pid)
        self.assertEqual(row["claim"], "User's main editor is VS Code")

        # The embedding must be refreshed too, or recall would still match the
        # superseded wording.
        hits = store.recall("which editor does the user use", limit=5)
        self.assertTrue(hits)
        self.assertIn("vs code", hits[0]["claim"].lower())

    def test_page_deletion_revokes_the_fact(self):
        store = self.store()
        pid, page = self.accept(store, "User prefers terse status updates")
        self.assertTrue(store.recall("terse updates", limit=5))

        page.unlink()
        summary = store.sync_from_vault()
        self.assertEqual(summary["revoked"], 1)
        self.assertEqual(store.get(pid)["status"], "deleted_by_edit")
        self.assertEqual(store.recall("terse updates", limit=5), [],
                         "a revoked fact must leave recall")

    def test_secret_shaped_edit_is_refused_and_original_kept(self):
        store = self.store()
        pid, page = self.accept(store, "Deploys use the staging key from the vault")

        text = page.read_text(encoding="utf-8")
        page.write_text(text.replace(
            "# Deploys use the staging key from the vault",
            "# Set the key with token=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            encoding="utf-8")

        summary = store.sync_from_vault()
        self.assertEqual(summary["skipped_secrets"], 1)
        row = store.get(pid)
        self.assertEqual(row["claim"], "Deploys use the staging key from the vault",
                         "a secret-shaped edit must not replace a safe fact")

    # -- rebuild ---------------------------------------------------------

    def test_rebuild_from_vault_restores_facts_with_provenance(self):
        store = self.store("first.sqlite")
        self.accept(store, "Web frontend uses Svelte", kind="decision",
                    project="web")
        self.accept(store, "Assistant should call itself Aurora",
                    kind="preference", peer="assistant")

        # Simulate total loss of the index/DB.
        fresh = self.store("rebuilt.sqlite")
        summary = fresh.rebuild_from_vault(self.vault, MEMORY_REL)
        self.assertGreaterEqual(summary["imported"], 2)

        rows = {r["claim"]: r for r in fresh.list_proposals(status="accepted")}
        self.assertIn("Web frontend uses Svelte", rows)
        web = rows["Web frontend uses Svelte"]
        self.assertEqual(web["kind"], "decision")
        self.assertEqual(web["project"], "web")
        aurora = rows["Assistant should call itself Aurora"]
        self.assertEqual(aurora["peer"], "assistant")
        self.assertTrue(aurora["accepted_at"], "accepted date must survive rebuild")

        hits = fresh.recall("frontend framework", limit=5)
        self.assertTrue(hits)
        self.assertIn("svelte", hits[0]["claim"].lower())

    def test_rebuild_is_idempotent(self):
        store = self.store()
        self.accept(store, "Staging database runs Postgres 15")
        # Rebuild into a FRESH database: rebuilding into the store that already
        # holds the fact inserts nothing on the first pass either, which would
        # make the first-vs-second comparison meaningless.
        fresh = self.store("rebuilt.sqlite")
        first = fresh.rebuild_from_vault(self.vault, MEMORY_REL)
        second = fresh.rebuild_from_vault(self.vault, MEMORY_REL)
        # First pass inserts the page; the second must insert nothing and must
        # not inflate the counter (a re-run is not new work).
        self.assertEqual(first["imported"], 1)
        self.assertEqual(second["imported"], 0)
        self.assertEqual(len(fresh.list_proposals(status="accepted")), 1,
                         "rebuilding twice must not duplicate rows")

    def test_superseded_fact_is_not_resurrected_by_rebuild(self):
        store = self.store()
        old_id, old_page = self.accept(
            store, "Staging server is moving to Hetzner VPS", project="web")
        new_id, _new_page = self.accept(
            store, "Staging server stays on Fly.io", project="web")
        store.supersede(new_id, old_id)
        mark_superseded_page(old_page, "Staging server stays on Fly.io")

        fresh = self.store("rebuilt.sqlite")
        fresh.rebuild_from_vault(self.vault, MEMORY_REL)

        # list_proposals filters by a single status, so collect every status
        # explicitly rather than relying on a default.
        statuses = {}
        for status_name in ("accepted", "superseded", "deleted_by_edit", "rejected"):
            for row in fresh.list_proposals(status=status_name):
                statuses[row["claim"]] = row["status"]
        self.assertEqual(statuses["Staging server is moving to Hetzner VPS"],
                         "superseded")
        self.assertEqual(statuses["Staging server stays on Fly.io"], "accepted")

        claims = [f["claim"] for f in read_fact_pages(self.vault, MEMORY_REL)]
        self.assertIn("Staging server stays on Fly.io", claims)
        self.assertNotIn("Staging server is moving to Hetzner VPS", claims,
                         "rebuild must not revive a superseded claim")

    def test_secret_shaped_page_is_not_imported(self):
        (self.vault / MEMORY_REL / "leaky.md").write_text(
            "---\nextractor: hwiki-extractor\nkind: environment\n"
            "status: accepted\n---\n\n"
            "# Set deploy token with password=hunter2024\n\n"
            "> **Evidence:** pasted\n", encoding="utf-8")
        store = self.store("rebuilt.sqlite")
        summary = store.rebuild_from_vault(self.vault, MEMORY_REL)
        self.assertEqual(summary["imported"], 0)
        self.assertEqual(summary["skipped"], 1)

    def test_rebuild_ignores_pages_without_the_extractor_marker(self):
        (self.vault / MEMORY_REL / "handwritten.md").write_text(
            "# My own notes\n\nNot written by the provider.\n", encoding="utf-8")
        store = self.store("rebuilt.sqlite")
        summary = store.rebuild_from_vault(self.vault, MEMORY_REL)
        self.assertEqual(summary["imported"], 0)
        self.assertEqual(summary["skipped"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
