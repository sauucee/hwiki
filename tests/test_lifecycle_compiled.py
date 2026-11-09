"""Regression tests: compiled facts must survive the vault lifecycle.

``compile`` merges a refinement into its owner's page as a delimited
``###`` section rather than a standalone page. Everything downstream that
walks pages (rebuild-from-vault, forget) must therefore understand those
sections; otherwise a rebuild silently drops accepted facts and a single
``forget`` deletes a page that still carries other accepted facts.
"""

from __future__ import annotations

import json
import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import _hwiki_path  # noqa: F401,E402

from hwiki.facts import (  # noqa: E402
    Embedder, ProposalStore, read_fact_pages,
)
from hwiki.facts_provider import FactsEngine  # noqa: E402

MEMORY_REL = Path("Hermes/Memory/Auto")
BASE = "The user prefers dark mode in editors"
MERGED = "The user prefers dark mode in editors and terminals"


def _slow_forget_worker(db_path, vault, memory_rel, proposal_id, delay, results):
    """A second writer: read the shared page, pause, then write its forget."""
    from hwiki.facts import Embedder, ProposalStore

    original = Path.read_text

    def slow_read(self, *args, **kwargs):
        text = original(self, *args, **kwargs)
        if self.suffix == ".md":
            time.sleep(delay)
        return text

    Path.read_text = slow_read
    store = ProposalStore(db_path, embedder=Embedder(mode="hash"))
    results.put(store.forget(proposal_id=proposal_id))


class CompiledLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.home = self.root / "hermes_home"
        self.vault = self.root / "vault"
        (self.vault / MEMORY_REL).mkdir(parents=True)
        self.home.mkdir()
        self._engines = []

    def tearDown(self):
        for engine in self._engines:
            try:
                store = engine._store
                if store is not None:
                    store.close()
            except Exception:
                pass
        self.tmp.cleanup()

    def engine(self):
        eng = FactsEngine(str(self.home), self.vault, MEMORY_REL,
                          config={"embeddings": {"mode": "hash"},
                                  "compile_enabled": True})
        self._engines.append(eng)
        eng.store()
        return eng

    def accept(self, engine, claim, evidence="stated by the user"):
        proposal = engine.store().propose(
            {"kind": "preference", "claim": claim, "evidence": evidence},
            user_text=evidence, session_id="s1",
        )
        engine.store().accept(proposal["id"])
        return engine.store().get(proposal["id"])

    def merged_pair(self):
        """Accept a base fact and a refinement that ``compile`` merges into it."""
        eng = self.engine()
        base = self.accept(eng, BASE)
        eng._promote(base["id"])
        refined = self.accept(eng, MERGED)
        page = eng._promote(refined["id"])
        self.assertIsNotNone(page)
        text = Path(page).read_text(encoding="utf-8")
        self.assertIn("<!-- hwiki:compiled:start -->", text,
                      "precondition: the refinement must have merged")
        return eng, base, refined, Path(page)

    # -- defect A: rebuild loses compiled facts --------------------------

    def test_rebuild_restores_compiled_facts(self):
        eng, base, refined, owner = self.merged_pair()

        fresh = ProposalStore(self.root / "fresh.sqlite",
                              embedder=Embedder(mode="hash"))
        summary = fresh.rebuild_from_vault(self.vault, MEMORY_REL)

        # Both accepted facts exist as pages/sections, so both must come back.
        self.assertEqual(summary["imported"], 2,
                         f"rebuild dropped a compiled fact (summary={summary})")
        restored = {r["claim"]: r for r in fresh.list_proposals(status="accepted")}
        self.assertIn(MERGED, restored,
                      "the compiled claim vanished from the rebuilt DB")
        self.assertEqual(restored[MERGED]["status"], "accepted")
        self.assertEqual(restored[MERGED]["kind"], "preference")
        self.assertEqual(restored[MERGED]["page_path"], str(owner))
        self.assertTrue(fresh.recall("dark mode terminals"),
                        "rebuilt compiled fact is not recallable")
        # idempotent: a second rebuild must not duplicate or lose anything
        again = fresh.rebuild_from_vault(self.vault, MEMORY_REL)
        self.assertEqual(again["imported"], 0)
        self.assertEqual(len(fresh.list_proposals(status="accepted")), 2)

    # -- defect B: forget must not delete facts it was not asked to ------

    def test_forget_compiled_fact_keeps_the_owner_and_its_other_facts(self):
        eng, base, refined, owner = self.merged_pair()

        result = eng.store().forget(proposal_id=refined["id"])
        self.assertTrue(result["ok"])
        self.assertEqual(eng.store().get(refined["id"])["status"],
                         "deleted_by_edit")

        # The page still carries the base claim, so it must survive.
        self.assertTrue(owner.exists(),
                        "forget deleted a page that still owns another accepted fact")
        text = owner.read_text(encoding="utf-8")
        self.assertNotIn(f"### {MERGED}", text,
                         "forgotten section is still on the page")
        self.assertIn(f"# {BASE}", text)

        self.assertEqual(eng.store().get(base["id"])["status"], "accepted")
        claims = [f["claim"] for f in read_fact_pages(self.vault, MEMORY_REL)]
        self.assertIn(BASE, claims)
        self.assertNotIn(MERGED, claims)

        # Rebuild after the forget must not resurrect the forgotten fact.
        fresh = ProposalStore(self.root / "after.sqlite",
                              embedder=Embedder(mode="hash"))
        fresh.rebuild_from_vault(self.vault, MEMORY_REL)
        rebuilt = [r["claim"] for r in fresh.list_proposals(status="accepted")]
        self.assertEqual(rebuilt, [BASE])

    # -- defect D: sync_from_vault must read the row's own section ----------

    def test_sync_does_not_rewrite_a_compiled_fact_as_the_owner_claim(self):
        eng, base, refined, owner = self.merged_pair()

        summary = eng.store().sync_from_vault()

        self.assertEqual(summary["updated"], 0, summary)
        self.assertEqual(eng.store().get(refined["id"])["claim"], MERGED,
                         "sync overwrote a merged fact with its owner's H1 claim")
        self.assertEqual(eng.store().get(base["id"])["claim"], BASE)

    def test_sync_applies_hand_edits_to_one_compiled_section_only(self):
        eng, base, refined, owner = self.merged_pair()
        edited = "The user prefers dark mode in editors and IDEs"
        text = owner.read_text(encoding="utf-8").replace(f"### {MERGED}",
                                                          f"### {edited}")
        owner.write_text(text, encoding="utf-8")

        summary = eng.store().sync_from_vault()

        self.assertEqual(summary["updated"], 1, summary)
        self.assertEqual(eng.store().get(refined["id"])["claim"], edited)
        self.assertEqual(eng.store().get(base["id"])["claim"], BASE)

    def test_sync_revokes_a_compiled_fact_deleted_from_the_page(self):
        eng, base, refined, owner = self.merged_pair()
        text = owner.read_text(encoding="utf-8")
        start = text.index("<!-- hwiki:compiled:start -->")
        end = text.index("<!-- hwiki:compiled:end -->") + len("<!-- hwiki:compiled:end -->")
        owner.write_text((text[:start] + text[end:]).strip() + "\n", encoding="utf-8")

        summary = eng.store().sync_from_vault()

        self.assertEqual(summary["revoked"], 1, summary)
        self.assertEqual(eng.store().get(refined["id"])["status"],
                         "deleted_by_edit")
        self.assertEqual(eng.store().get(base["id"])["status"], "accepted")

    def test_read_fact_pages_keeps_a_merged_sections_own_provenance(self):
        eng, base, refined, owner = self.merged_pair()
        text = owner.read_text(encoding="utf-8")
        text = text.replace(
            "*kind: preference · peer: user · accepted:",
            "*kind: decision · peer: assistant · accepted:", 1)
        owner.write_text(text, encoding="utf-8")

        facts = {f["claim"]: f for f in read_fact_pages(self.vault, MEMORY_REL)}

        self.assertEqual(facts[MERGED]["kind"], "decision",
                         "merged section reported the owner page's kind")
        self.assertEqual(facts[MERGED]["peer"], "assistant",
                         "merged section reported the owner page's peer")
        self.assertEqual(facts[BASE]["kind"], "preference")

    # -- defect C: supersede must not tombstone every fact on a shared page --

    def supersede(self, engine, new_id, old_id):
        return json.loads(engine.handle_tool_call(
            "memory_supersede",
            {"proposal_id": new_id, "supersede_id": old_id},
            lambda message: json.dumps({"error": message})))

    def test_supersede_compiled_fact_keeps_owner_and_other_facts_visible(self):
        eng, base, refined, owner = self.merged_pair()
        new = self.accept(eng, "Deploys are frozen on Fridays")
        eng._promote(new["id"])

        result = self.supersede(eng, new["id"], refined["id"])
        self.assertTrue(result.get("success"), result)
        self.assertEqual(eng.store().get(refined["id"])["status"], "superseded")

        # Only the superseded claim leaves recall; the page's other facts stay.
        self.assertTrue(owner.exists())
        self.assertIn("status: auto", owner.read_text(encoding="utf-8"))
        claims = [f["claim"] for f in read_fact_pages(self.vault, MEMORY_REL)]
        self.assertIn(BASE, claims,
                      "superseding a merged fact hid the owner claim")
        self.assertNotIn(MERGED, claims)
        self.assertEqual(eng.store().get(base["id"])["status"], "accepted")

        # Rebuild must agree with the vault: the superseded section stays
        # superseded instead of being revived as an accepted fact.
        fresh = ProposalStore(self.root / "after.sqlite",
                              embedder=Embedder(mode="hash"))
        fresh.rebuild_from_vault(self.vault, MEMORY_REL)
        by_claim = {r["claim"]: r for r in fresh.list_proposals(status="accepted")}
        self.assertIn(BASE, by_claim)
        self.assertNotIn(MERGED, by_claim)

    def test_supersede_owner_keeps_merged_refinements_visible(self):
        eng, base, refined, owner = self.merged_pair()
        new = self.accept(eng, "Deploys are frozen on Fridays")
        eng._promote(new["id"])

        result = self.supersede(eng, new["id"], base["id"])
        self.assertTrue(result.get("success"), result)
        claims = [f["claim"] for f in read_fact_pages(self.vault, MEMORY_REL)]
        self.assertNotIn(BASE, claims)
        self.assertIn(MERGED, claims,
                      "superseding the owner hid a still-live refinement")

    def test_forget_owner_with_compiled_sections_refuses_to_delete_page(self):
        eng, base, refined, owner = self.merged_pair()

        result = eng.store().forget(proposal_id=base["id"])
        self.assertFalse(result["ok"])
        self.assertIn("compiled facts", result["error"])
        self.assertTrue(owner.exists(),
                        "a refused forget must leave the page intact")
        self.assertEqual(eng.store().get(base["id"])["status"], "accepted")
        self.assertEqual(eng.store().get(refined["id"])["status"], "accepted")
        claims = [f["claim"] for f in read_fact_pages(self.vault, MEMORY_REL)]
        self.assertIn(BASE, claims)
        self.assertIn(MERGED, claims)


    # -- blocker 1: forget must fail closed on stale/unreadable pages -------

    def test_forget_stale_section_row_does_not_delete_the_owner_page(self):
        eng, base, refined, owner = self.merged_pair()
        text = owner.read_text(encoding="utf-8")
        start = text.index("<!-- hwiki:compiled:start -->")
        end = text.index("<!-- hwiki:compiled:end -->") + len("<!-- hwiki:compiled:end -->")
        owner.write_text((text[:start] + text[end:]).strip() + "\n", encoding="utf-8")

        result = eng.store().forget(proposal_id=refined["id"])

        self.assertFalse(result["ok"], result)
        self.assertTrue(owner.exists(),
                        "a stale section row deleted the owner's page")
        self.assertEqual(eng.store().get(base["id"])["status"], "accepted")
        self.assertEqual(eng.store().get(refined["id"])["status"], "accepted")

    def test_forget_unreadable_shared_page_fails_closed(self):
        eng, base, refined, owner = self.merged_pair()

        with mock.patch.object(Path, "read_text", side_effect=OSError("denied")):
            result = eng.store().forget(proposal_id=refined["id"])

        self.assertFalse(result["ok"], result)
        self.assertTrue(owner.exists(), "an unreadable page was deleted")
        self.assertEqual(eng.store().get(refined["id"])["status"], "accepted")

    def test_forget_failed_unlink_does_not_revoke_the_row(self):
        eng = self.engine()
        fact = self.accept(eng, "Deploys are frozen on Fridays")
        page = Path(eng._promote(fact["id"]))

        with mock.patch.object(Path, "unlink", side_effect=OSError("busy")):
            result = eng.store().forget(proposal_id=fact["id"])

        self.assertFalse(result["ok"], result)
        self.assertTrue(page.exists())
        self.assertEqual(eng.store().get(fact["id"])["status"], "accepted")

    # -- blocker 2: an id targets one section, never every same-claim copy ---

    def test_forget_by_id_does_not_remove_another_section_with_same_claim(self):
        eng, base, refined, owner = self.merged_pair()
        # Two independent facts may carry the same claim text (e.g. accepted on
        # different sessions); only the id identifies the one being forgotten.
        other_id = "f" * 32
        text = owner.read_text(encoding="utf-8")
        block = text[text.index("<!-- hwiki:compiled:start -->"):
                     text.index("<!-- hwiki:compiled:end -->") + len("<!-- hwiki:compiled:end -->")]
        duplicated = block.replace(f"proposal: {refined['id']}",
                                   f"proposal: {other_id}")
        self.assertNotEqual(duplicated, block)
        owner.write_text(text.rstrip("\n") + "\n\n" + duplicated + "\n",
                         encoding="utf-8")

        result = eng.store().forget(proposal_id=refined["id"])

        self.assertTrue(result["ok"], result)
        self.assertTrue(owner.exists())
        remaining = owner.read_text(encoding="utf-8")
        self.assertEqual(remaining.count("<!-- hwiki:compiled:start -->"), 1,
                         "forget by id removed a different section with the same claim")
        self.assertIn(f"proposal: {other_id}", remaining)

    # -- blocker 3: DB supersede cannot outrun a failed vault tombstone -----

    def test_supersede_tombstone_failure_leaves_fact_accepted_in_db(self):
        eng, base, refined, owner = self.merged_pair()
        new = self.accept(eng, "Deploys are frozen on Fridays")
        eng._promote(new["id"])

        with mock.patch.object(Path, "replace", side_effect=OSError("disk full")):
            result = self.supersede(eng, new["id"], refined["id"])

        self.assertFalse(result.get("success"), result)
        self.assertEqual(eng.store().get(refined["id"])["status"], "accepted",
                         "DB was superseded although the vault tombstone failed")
        self.assertIn("### " + MERGED, owner.read_text(encoding="utf-8"))
        fresh = ProposalStore(self.root / "after-failure.sqlite",
                              embedder=Embedder(mode="hash"))
        fresh.rebuild_from_vault(self.vault, MEMORY_REL)
        self.assertEqual(fresh.get(refined["id"])["status"], "accepted")

    # -- blocker 4: compile must not delete the standalone before repointing -

    def test_compile_repoint_failure_keeps_the_standalone_page(self):
        eng = self.engine()
        base = self.accept(eng, BASE)
        eng._promote(base["id"])
        refined = self.accept(eng, MERGED)
        store = eng.store()
        original = store.set_page_path
        calls = []

        def flaky(proposal_id, page_path):
            calls.append(proposal_id)
            if len(calls) == 2 and proposal_id == refined["id"]:
                raise OSError("database is locked")
            return original(proposal_id, page_path)

        with mock.patch.object(store, "set_page_path", side_effect=flaky):
            page = eng._promote(refined["id"])

        self.assertIsNotNone(page)
        self.assertTrue(Path(page).exists(),
                        "compile removed the standalone before the DB repoint")
        self.assertEqual(Path(eng.store().get(refined["id"])["page_path"]),
                         Path(page))
        summary = eng.store().sync_from_vault()
        self.assertEqual(summary["revoked"], 0, summary)
        self.assertEqual(eng.store().get(refined["id"])["status"], "accepted")

    # -- blocker 5: a secret owner must not hide safe compiled sections ------

    def test_rebuild_imports_safe_sections_of_a_secret_shaped_owner(self):
        eng, base, refined, owner = self.merged_pair()
        text = owner.read_text(encoding="utf-8")
        text = text.replace(f"# {BASE}", "# password=hunter2-not-a-real-secret", 1)
        owner.write_text(text, encoding="utf-8")

        fresh = ProposalStore(self.root / "secret-owner.sqlite",
                              embedder=Embedder(mode="hash"))
        fresh.rebuild_from_vault(self.vault, MEMORY_REL)

        restored = {r["claim"]: r for r in fresh.list_proposals(status="accepted")}
        self.assertIn(MERGED, restored,
                      "a secret-shaped owner suppressed a safe compiled section")
        self.assertNotIn("password=hunter2-not-a-real-secret", restored)

    # -- blocker 6: a hand-edited owner tombstone must sync to the DB --------

    def test_sync_marks_owner_superseded_when_vault_says_so(self):
        eng, base, refined, owner = self.merged_pair()
        text = owner.read_text(encoding="utf-8").replace(
            "status: auto", "status: superseded", 1)
        owner.write_text(text, encoding="utf-8")

        summary = eng.store().sync_from_vault()

        self.assertEqual(eng.store().get(base["id"])["status"], "superseded",
                         "manual owner tombstone was not mirrored to the DB")
        self.assertGreaterEqual(summary["updated"], 1, summary)
        self.assertEqual(eng.store().get(refined["id"])["status"], "accepted")

    # -- re-review 1: interrupted compile must not leave a resurrectable copy -

    def test_forget_after_interrupted_compile_prevents_resurrection(self):
        eng, base, refined, owner = self.merged_pair()
        x = self.accept(eng, "Friday deployments must be frozen")
        original_unlink = Path.unlink

        def fail_standalone_unlink(self, *args, **kwargs):
            if "friday-deployments" in self.name:
                raise OSError("simulated crash before standalone deletion")
            return original_unlink(self, *args, **kwargs)

        forced_owner = [{"path": str(owner), "claim": "owner", "score": 0.99}]
        with mock.patch.object(eng, "_candidate_owners", return_value=forced_owner), \
                mock.patch.object(Path, "unlink", fail_standalone_unlink):
            eng._promote(x["id"])

        # Compile repointed the row at the owner but could not remove the
        # standalone page: two copies of one fact now exist in the vault.
        self.assertEqual(Path(eng.store().get(x["id"])["page_path"]), owner)

        result = eng.store().forget(proposal_id=x["id"])
        self.assertTrue(result["ok"], result)

        fresh = ProposalStore(self.root / "after-interrupted.sqlite",
                              embedder=Embedder(mode="hash"))
        fresh.rebuild_from_vault(self.vault, MEMORY_REL)
        claims = [row["claim"]
                  for status in ("accepted", "superseded", "deleted_by_edit")
                  for row in fresh.list_proposals(status=status)]
        self.assertNotIn(x["claim"], claims,
                         "a stale standalone copy resurrected a forgotten fact")

    # -- re-review 2: concurrent forgets must not resurrect each other --------

    def test_concurrent_forgets_on_shared_page_do_not_resurrect(self):
        eng, base, refined, owner = self.merged_pair()
        x = self.accept(eng, "Friday deployments must be frozen")
        forced_owner = [{"path": str(owner), "claim": "owner", "score": 0.99}]
        with mock.patch.object(eng, "_candidate_owners", return_value=forced_owner):
            eng._promote(x["id"])
        self.assertIn("### " + x["claim"], owner.read_text(encoding="utf-8"))

        ctx = multiprocessing.get_context("fork")
        results = ctx.Queue()
        first = ctx.Process(target=_slow_forget_worker,
                            args=(str(eng.store().database), self.vault, MEMORY_REL,
                                  refined["id"], 0.5, results))
        first.start()
        time.sleep(0.15)
        second_result = eng.store().forget(proposal_id=x["id"])
        first.join(10)
        self.assertFalse(first.is_alive(), "second writer deadlocked on the page")
        first_result = results.get(timeout=2)

        self.assertTrue(first_result["ok"], first_result)
        self.assertTrue(second_result["ok"], second_result)
        text = owner.read_text(encoding="utf-8")
        self.assertNotIn(f"### {MERGED}", text,
                         "a concurrent forget resurrected the other section")
        self.assertNotIn(f"### {x['claim']}", text,
                         "a concurrent forget resurrected the other section")
        self.assertIn(f"# {BASE}", text)
        claims = [f["claim"] for f in read_fact_pages(self.vault, MEMORY_REL)]
        self.assertEqual(claims, [BASE])
        self.assertEqual(eng.store().get(refined["id"])["status"], "deleted_by_edit")
        self.assertEqual(eng.store().get(x["id"])["status"], "deleted_by_edit")

    def test_interleaved_writer_during_forget_cannot_resurrect(self):
        """A writer that lands inside our page write must not be overwritten."""
        eng, base, refined, owner = self.merged_pair()
        x = self.accept(eng, "Friday deployments must be frozen")
        forced_owner = [{"path": str(owner), "claim": "owner", "score": 0.99}]
        with mock.patch.object(eng, "_candidate_owners", return_value=forced_owner):
            eng._promote(x["id"])
        self.assertIn("### " + x["claim"], owner.read_text(encoding="utf-8"))

        from hwiki import facts as facts_module
        original = facts_module._atomic_write
        invoked = []

        def interleave(page, text):
            # The nested forget finishes while the outer one holds a stale
            # page snapshot; the outer write must not bring its section back.
            if not invoked:
                invoked.append(True)
                nested = eng.store().forget(proposal_id=x["id"])
                self.assertTrue(nested["ok"], nested)
            return original(page, text)

        with mock.patch.object(facts_module, "_atomic_write", interleave):
            result = eng.store().forget(proposal_id=refined["id"])

        self.assertTrue(result["ok"], result)
        self.assertTrue(invoked, "precondition: the interleaved write must run")
        claims = [f["claim"] for f in read_fact_pages(self.vault, MEMORY_REL)]
        self.assertNotIn(x["claim"], claims,
                         "an interleaved forget was resurrected")
        self.assertNotIn(MERGED, claims)
        self.assertIn(BASE, claims)

    # -- re-review 3: only an accepted, visible replacement may supersede -----

    def test_proposed_replacement_cannot_supersede_an_accepted_fact(self):
        eng, base, refined, owner = self.merged_pair()
        proposed = eng.store().propose(
            {"kind": "fact", "claim": "Unaccepted replacement claim",
             "evidence": "user said"},
            user_text="user said", session_id="s1")

        result = self.supersede(eng, proposed["id"], refined["id"])

        self.assertFalse(result.get("success"), result)
        self.assertEqual(eng.store().get(refined["id"])["status"], "accepted")
        self.assertEqual(eng.store().get(proposed["id"])["status"], "proposed")
        claims = [f["claim"] for f in read_fact_pages(self.vault, MEMORY_REL)]
        self.assertIn(MERGED, claims,
                      "a proposed replacement hid an accepted fact")

    # -- blocker 7: rebuild must restore lineage to a compiled replacement ---

    def test_rebuild_resolves_superseded_by_to_a_compiled_section(self):
        eng, base, refined, owner = self.merged_pair()
        result = self.supersede(eng, refined["id"], base["id"])
        self.assertTrue(result.get("success"), result)

        fresh = ProposalStore(self.root / "lineage.sqlite",
                              embedder=Embedder(mode="hash"))
        fresh.rebuild_from_vault(self.vault, MEMORY_REL)

        rebuilt_base = fresh.get(base["id"])
        self.assertEqual(rebuilt_base["status"], "superseded")
        self.assertEqual(rebuilt_base["superseded_by"], refined["id"],
                         "rebuild lost the link to the compiled replacement")
        self.assertEqual(fresh.get(refined["id"])["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
