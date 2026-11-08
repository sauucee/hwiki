import json
import time
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import _hwiki_path  # noqa: F401,E402
from hwiki.facts import (
    ProposalRejected, ProposalStore, Embedder, hash_embedding,
    related_and_conflict, resolve_vault, write_fact_page,
    read_fact_pages, mark_superseded_page, render_profile, load_builtin_entries,
    write_session_diary, write_fact_canvas,
)


class ProposalStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_plain_user_preference_is_only_a_proposal(self):
        store = ProposalStore(self.root / 'proposals.sqlite')
        proposal = store.propose(
            {'kind': 'preference', 'claim': 'User prefers short replies.',
             'evidence': 'I prefer short replies.'},
            user_text='I prefer short replies.', session_id='session-a',
        )
        self.assertEqual(proposal['status'], 'proposed')
        self.assertEqual([p['id'] for p in store.list_proposals()], [proposal['id']])
        self.assertEqual(store.list_proposals(status='accepted'), [])

    def test_duplicate_proposal_does_not_create_a_second_row(self):
        store = ProposalStore(self.root / 'proposals.sqlite')
        first = store.propose(
            {'kind': 'preference', 'claim': 'User prefers short replies.',
             'evidence': 'I prefer short replies.'},
            user_text='I prefer short replies.', session_id='session-a',
        )
        second = store.propose(
            {'kind': 'preference', 'claim': 'User prefers short replies.',
             'evidence': 'I prefer short replies.'},
            user_text='I prefer short replies.', session_id='session-a',
        )
        self.assertEqual(first['id'], second['id'])
        self.assertEqual(len(store.list_proposals()), 1)

    def test_secret_shaped_claim_is_rejected(self):
        store = ProposalStore(self.root / 'proposals.sqlite')
        with self.assertRaises(ProposalRejected):
            store.propose(
                {'kind': 'preference', 'claim': 'User prefers short replies.',
                 'evidence': 'api_key=sk-abc123456789012345'},
                user_text='x', session_id='session-a',
            )

    def test_rule_about_secrets_is_allowed(self):
        """Discussing secrets is fine — only secret MATERIAL is rejected."""
        store = ProposalStore(self.root / 'proposals.sqlite')
        proposal = store.propose(
            {'kind': 'blocker', 'claim': 'Never paste API keys or secrets in chat',
             'evidence': 'standing rule from the user'},
            user_text='x', session_id='session-a',
        )
        self.assertEqual(proposal['status'], 'proposed')

    def test_accept_transitions_proposed_to_accepted(self):
        store = ProposalStore(self.root / 'proposals.sqlite')
        proposal = store.propose(
            {'kind': 'preference', 'claim': 'User prefers short replies.',
             'evidence': 'I prefer short replies.'},
            user_text='I prefer short replies.', session_id='session-a',
        )
        self.assertTrue(store.accept(proposal['id']))
        self.assertEqual(store.list_proposals(status='accepted')[0]['id'], proposal['id'])
        self.assertFalse(store.accept(proposal['id']))

    def test_v1_database_is_migrated(self):
        """A v1 DB (no new columns) opens cleanly and keeps its rows."""
        db = self.root / 'old.sqlite'
        conn = sqlite3.connect(db)
        conn.execute('''CREATE TABLE proposals (
            id TEXT PRIMARY KEY, dedup_key TEXT NOT NULL, status TEXT NOT NULL,
            kind TEXT NOT NULL, claim TEXT NOT NULL, evidence TEXT NOT NULL,
            session_id TEXT NOT NULL, source_hash TEXT NOT NULL, project TEXT NOT NULL,
            created_at REAL NOT NULL)''')
        conn.execute("INSERT INTO proposals VALUES ('old1','dk','proposed','preference','c','e','s','h','',1)")
        conn.commit()
        conn.close()
        store = ProposalStore(db)
        rows = store.list_proposals()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['id'], 'old1')
        proposal = store.propose(
            {'kind': 'preference', 'claim': 'new fact', 'evidence': 'ev'},
            user_text='new fact', session_id='s')
        self.assertEqual(proposal['status'], 'proposed')

    def test_conflicting_claim_is_flagged(self):
        store = ProposalStore(self.root / 'proposals.sqlite')
        base = store.propose(
            {'kind': 'preference', 'claim': 'User prefers dark mode for coding',
             'evidence': 'I like dark mode'}, user_text='I like dark mode', session_id='s')
        store.accept(base['id'])
        conflict = store.propose(
            {'kind': 'preference', 'claim': 'User does not prefer dark mode for coding',
             'evidence': 'actually light mode now'}, user_text='actually light mode now', session_id='s')
        self.assertEqual(conflict['possible_conflict'], 1)
        self.assertTrue(any(r['key'] == base['id'] for r in conflict['related']))

    def test_related_claim_is_linked_without_conflict(self):
        store = ProposalStore(self.root / 'proposals.sqlite')
        base = store.propose(
            {'kind': 'environment', 'claim': 'The staging server runs on Fly.io',
             'evidence': 'deployed there'}, user_text='deployed there', session_id='s')
        store.accept(base['id'])
        related = store.propose(
            {'kind': 'environment', 'claim': 'The staging server runs Postgres',
             'evidence': 'pg on staging'}, user_text='pg on staging', session_id='s')
        self.assertEqual(related['possible_conflict'], 0)
        self.assertTrue(any(r['key'] == base['id'] for r in related['related']))

    def test_near_duplicate_of_accepted_fact_returns_existing(self):
        store = ProposalStore(self.root / 'proposals.sqlite')
        base = store.propose(
            {'kind': 'preference', 'claim': 'User prefers short replies',
             'evidence': 'said so'}, user_text='said so', session_id='s')
        store.accept(base['id'])
        near = store.propose(
            {'kind': 'preference', 'claim': 'User prefers short replies',
             'evidence': 'said so again'}, user_text='said so again', session_id='s')
        self.assertEqual(near['id'], base['id'])

    def test_duplicate_of_builtin_memory_is_annotated(self):
        store = ProposalStore(self.root / 'proposals.sqlite')
        proposal = store.propose(
            {'kind': 'preference', 'claim': 'User does not want API keys pasted in chat',
             'evidence': 'pasting keys unsafe'},
            user_text='pasting keys unsafe', session_id='s',
            builtin_entries=[{'text': "User does not want API keys or secrets pasted in chat",
                              'target': 'user'}])
        self.assertTrue(proposal['duplicate_of_builtin'])

    def test_supersede_marks_old_fact_and_recall_skips_it(self):
        store = ProposalStore(self.root / 'proposals.sqlite')
        old = store.propose(
            {'kind': 'environment', 'claim': 'Deployment target is Fly.io',
             'evidence': 'migrated'}, user_text='migrated', session_id='s')
        store.accept(old['id'])
        new = store.propose(
            {'kind': 'environment', 'claim': 'Deployment target is VPS Hetzner',
             'evidence': 'moved again'}, user_text='moved again', session_id='s')
        store.accept(new['id'])
        ok, old_row = store.supersede(new['id'], old['id'])
        self.assertTrue(ok)
        self.assertEqual(old_row['status'], 'superseded')
        self.assertEqual(old_row['superseded_by'], new['id'])
        self.assertEqual(store.get(old['id'])['status'], 'superseded')
        self.assertEqual(store.get(new['id'])['supersedes'], old['id'])
        hits = store.recall('deployment target fly hosting')
        self.assertTrue(all(h['id'] != old['id'] for h in hits))
        new_hit = next(h for h in hits if h['id'] == new['id'])
        self.assertEqual(new_hit['replaced'], 'Deployment target is Fly.io')
        self.assertTrue(new_hit['accepted'])

    def test_recall_project_filter(self):
        store = ProposalStore(self.root / 'proposals.sqlite')
        web = store.propose(
            {'kind': 'decision', 'claim': 'Web frontend uses Svelte',
             'evidence': 'chosen'}, user_text='svelte', session_id='s', project='web')
        store.accept(web['id'])
        ops = store.propose(
            {'kind': 'environment', 'claim': 'Ops tooling runs on Hetzner',
             'evidence': 'stated'}, user_text='hetzner', session_id='s', project='ops')
        store.accept(ops['id'])
        self.assertEqual([h['id'] for h in store.recall('frontend framework tooling', project='web')],
                         [web['id']])
        self.assertEqual([h['id'] for h in store.recall('frontend framework tooling', project='ops')],
                         [ops['id']])

    def test_temporal_query_boosts_recent_fact(self):
        store = ProposalStore(self.root / 'proposals.sqlite', embedder=Embedder(mode='hash'))
        old = store.propose(
            {'kind': 'environment', 'claim': 'Newsletter runs on Sendgrid',
             'evidence': 'e'}, user_text='x', session_id='s')
        store.accept(old['id'])
        conn = sqlite3.connect(self.root / 'proposals.sqlite')
        conn.execute("UPDATE proposals SET accepted_at=? WHERE id=?",
                     (time.time() - 90 * 86400, old['id']))  # 90 days old
        conn.commit(); conn.close()
        new = store.propose(
            {'kind': 'environment', 'claim': 'Newsletter runs on Mailgun now',
             'evidence': 'e'}, user_text='x', session_id='s')
        store.accept(new['id'])
        from hwiki.facts import _iso_date
        hits_plain = store.recall('newsletter runs', now=time.time())
        hits_temporal = store.recall('newsletter runs recently', now=time.time())
        self.assertEqual(hits_plain[0]['id'], old['id'])  # no time intent -> accept order stands
        self.assertEqual(hits_temporal[0]['id'], new['id'])  # "recently" surfaces the fresh fact
        self.assertTrue(hits_temporal[0]['accepted'])

    def test_recall_includes_evidence(self):
        store = ProposalStore(self.root / 'proposals.sqlite', embedder=Embedder(mode='hash'))
        p = store.propose(
            {'kind': 'decision', 'claim': 'Deploy target is Hetzner VPS',
             'evidence': 'user said migrated everything off Fly.io'}, user_text='x', session_id='s')
        store.accept(p['id'])
        hits = store.recall('deploy target hosting')
        self.assertTrue(hits)
        self.assertIn('migrated', hits[0]['evidence'])

    def test_sync_from_vault_updates_edits_and_revokes_deletions(self):
        store = ProposalStore(self.root / 'proposals.sqlite', embedder=Embedder(mode='hash'))
        p = store.propose(
            {'kind': 'environment', 'claim': 'Deploy target is Fly.io',
             'evidence': 'original evidence'}, user_text='x', session_id='s')
        store.accept(p['id'])
        vault = self.root / 'vault'
        page = write_fact_page(vault, Path('Hermes/Memory/Auto'), store.get(p['id']))
        store.set_page_path(p['id'], page)
        # user edits the claim in Obsidian
        page.write_text(page.read_text().replace('# Deploy target is Fly.io',
                                                 '# Deploy target is Hetzner VPS'))
        summary = store.sync_from_vault()
        self.assertEqual(summary['updated'], 1)
        row = store.get(p['id'])
        self.assertEqual(row['claim'], 'Deploy target is Hetzner VPS')
        self.assertEqual(row['evidence'], 'original evidence')
        # user deletes the page -> fact revoked
        page.unlink()
        summary = store.sync_from_vault()
        self.assertEqual(summary['revoked'], 1)
        self.assertEqual(store.get(p['id'])['status'], 'deleted_by_edit')
        self.assertEqual(store.recall('deploy target hosting'), [])

    def test_sync_skips_secret_shaped_edits(self):
        store = ProposalStore(self.root / 'proposals.sqlite', embedder=Embedder(mode='hash'))
        p = store.propose(
            {'kind': 'environment', 'claim': 'Deploy target is Fly.io',
             'evidence': 'e'}, user_text='x', session_id='s')
        store.accept(p['id'])
        vault = self.root / 'vault'
        page = write_fact_page(vault, Path('Hermes/Memory/Auto'), store.get(p['id']))
        store.set_page_path(p['id'], page)
        page.write_text(page.read_text().replace('# Deploy target is Fly.io',
                                                 '# api_key=sk-abcdefghijklmnop123456 stored in config'))
        summary = store.sync_from_vault()
        self.assertEqual(summary['skipped_secrets'], 1)
        self.assertEqual(store.get(p['id'])['claim'], 'Deploy target is Fly.io')

    def test_stats_snapshot(self):
        store = ProposalStore(self.root / 'proposals.sqlite', embedder=Embedder(mode='hash'))
        p = store.propose(
            {'kind': 'preference', 'claim': 'User prefers vim', 'evidence': 'e'},
            user_text='x', session_id='s')
        store.accept(p['id'])
        stats = store.stats()
        self.assertEqual(stats['counts'].get('accepted'), 1)
        self.assertEqual(stats['embedding_tier'], 'hash')
        self.assertTrue(stats['last_accepted'])

    def test_rebuild_from_vault_restores_empty_db(self):
        vault = self.root / 'vault'
        memory_rel = Path('Hermes/Memory/Auto')
        store = ProposalStore(self.root / 'db.sqlite', embedder=Embedder(mode='hash'))
        p = store.propose(
            {'kind': 'environment', 'claim': 'Deploy target is Fly.io',
             'evidence': 'original evidence'}, user_text='x', session_id='s')
        store.accept(p['id'])
        write_fact_page(vault, memory_rel, store.get(p['id']))
        p2 = store.propose(
            {'kind': 'environment', 'claim': 'Old stale fact about deploys',
             'evidence': 'e2'}, user_text='x', session_id='s')
        store.accept(p2['id'])
        page2 = write_fact_page(vault, memory_rel, store.get(p2['id']))
        store.supersede(p['id'], p2['id'])
        mark_superseded_page(page2, 'Deploy target is Fly.io')

        # fresh DB (lost store) -> rebuild from the pages
        fresh = ProposalStore(self.root / 'fresh.sqlite', embedder=Embedder(mode='hash'))
        summary = fresh.rebuild_from_vault(vault, memory_rel)
        self.assertEqual(summary['imported'], 2)
        restored = fresh.get(p['id'])
        self.assertEqual(restored['claim'], 'Deploy target is Fly.io')
        self.assertEqual(restored['peer'], 'user')
        self.assertEqual(restored['status'], 'accepted')
        stale = fresh.get(p2['id'])
        self.assertEqual(stale['status'], 'superseded')
        self.assertEqual(stale['superseded_by'], p['id'])  # lineage resolved on second pass
        hits = fresh.recall('deploy target hosting')
        self.assertTrue(hits and all(h['id'] != p2['id'] for h in hits))
        # idempotent: a re-run inserts nothing, so it must not report imports.
        # (The counter used to count attempts and blamed 2 every time, which hid
        # both true idempotency and real insert conflicts from callers.)
        again = fresh.rebuild_from_vault(vault, memory_rel)
        self.assertEqual(again['imported'], 0)
        self.assertEqual(len(fresh.list_proposals(status='accepted')), 1)

    def test_forget_revokes_by_id_and_exact_claim(self):
        store = ProposalStore(self.root / 'db.sqlite', embedder=Embedder(mode='hash'))
        vault = self.root / 'vault'
        p = store.propose(
            {'kind': 'preference', 'claim': 'User prefers terse changelogs',
             'evidence': 'e'}, user_text='x', session_id='s')
        store.accept(p['id'])
        page = write_fact_page(vault, Path('Hermes/Memory/Auto'), store.get(p['id']))
        store.set_page_path(p['id'], page)
        # fuzzy claims are refused
        result = store.forget(claim='terse changelogs')
        self.assertFalse(result['ok'])
        # exact claim works and removes the page
        result = store.forget(claim='User prefers terse changelogs')
        self.assertTrue(result['ok'])
        self.assertTrue(result['page_removed'])
        self.assertFalse(page.exists())
        self.assertEqual(store.get(p['id'])['status'], 'deleted_by_edit')
        self.assertEqual(store.recall('terse changelogs'), [])
        # by id, on a fact without a page
        p2 = store.propose(
            {'kind': 'preference', 'claim': 'User prefers short commits',
             'evidence': 'e'}, user_text='x', session_id='s')
        store.accept(p2['id'])
        result = store.forget(proposal_id=p2['id'])
        self.assertTrue(result['ok'])
        self.assertFalse(result['page_removed'])
        self.assertEqual(store.get(p2['id'])['status'], 'deleted_by_edit')

    def test_neighbors_link_related_and_superseded(self):
        store = ProposalStore(self.root / 'db.sqlite', embedder=Embedder(mode='hash'))
        base = store.propose(
            {'kind': 'environment', 'claim': 'Deploy target is Fly.io',
             'evidence': 'e'}, user_text='x', session_id='s')
        store.accept(base['id'])
        new = store.propose(
            {'kind': 'environment', 'claim': 'Deploy target is Hetzner VPS',
             'evidence': 'e'}, user_text='x', session_id='s')
        store.accept(new['id'])
        store.supersede(new['id'], base['id'])
        nb = store.neighbors([new['id']])
        self.assertTrue(any(n['id'] == base['id'] and n['relation'] == 'replaced' for n in nb))

    def test_consolidate_removes_duplicate_pending(self):
        store = ProposalStore(self.root / 'proposals.sqlite')
        first = store.propose(
            {'kind': 'preference', 'claim': 'User prefers vim keybindings', 'evidence': 'x'},
            user_text='x', session_id='s1')
        # simulate a legacy/crash duplicate row with the same dedup_key
        import sqlite3 as _sq
        dedup_key = ProposalStore._dedup_key('user', 'preference', '', 'User prefers vim keybindings')
        conn = _sq.connect(self.root / 'proposals.sqlite')
        conn.execute(
            "INSERT INTO proposals (id, dedup_key, status, kind, claim, evidence, session_id,"
            " source_hash, project, created_at) VALUES ('dup1', ?, 'proposed', 'preference',"
            " 'User prefers vim keybindings', 'y', 's2', 'h', '', 2)", (dedup_key,))
        conn.execute("UPDATE proposals SET created_at=? WHERE id='dup1'", (time.time() + 10,))
        conn.commit()
        conn.close()
        tidy = store.consolidate()
        self.assertEqual(tidy['duplicates'], 1)
        self.assertEqual(tidy['expired'], 0)
        remaining = store.list_proposals()
        self.assertEqual([p['id'] for p in remaining], [first['id']])

    def test_proposal_expiry(self):
        import time as _t
        store = ProposalStore(self.root / 'exp.sqlite')
        store.propose({'kind': 'preference', 'claim': 'Fresh proposal worth keeping',
                       'evidence': 'x'}, user_text='x', session_id='s1')
        stale = store.propose({'kind': 'preference', 'claim': 'Ancient proposal long forgotten',
                               'evidence': 'y'}, user_text='y', session_id='s1')
        conn = sqlite3.connect(self.root / 'exp.sqlite')
        conn.execute("UPDATE proposals SET created_at=? WHERE id=?",
                     (_t.time() - 30 * 86400, stale['id']))
        conn.commit(); conn.close()
        tidy = store.consolidate(proposal_ttl_days=14)
        self.assertEqual(tidy['expired'], 1)
        self.assertEqual(store.get(stale['id'])['status'], 'expired')
        fresh = store.propose({'kind': 'preference', 'claim': 'Fresh proposal worth keeping',
                               'evidence': 'x'}, user_text='x', session_id='s1')
        self.assertEqual(fresh['status'], 'proposed')

    def test_recall_ranks_relevant_fact_first(self):
        store = ProposalStore(self.root / 'proposals.sqlite')
        hit = store.propose(
            {'kind': 'environment', 'claim': 'Staging database runs Postgres 15',
             'evidence': 'checked'}, user_text='checked', session_id='s')
        store.accept(hit['id'])
        miss = store.propose(
            {'kind': 'preference', 'claim': 'User enjoys cooking pasta on weekends',
             'evidence': 'mentioned'}, user_text='mentioned', session_id='s')
        store.accept(miss['id'])
        hits = store.recall('which postgres version does staging run')
        self.assertTrue(hits)
        self.assertEqual(hits[0]['id'], hit['id'])


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_hash_embedding_is_deterministic_and_normalized(self):
        a = hash_embedding('User prefers VS Code for editing')
        b = hash_embedding('User prefers VS Code for editing')
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2048 * 4)

    def test_word_variants_match_via_stemming(self):
        from hwiki.facts import _cosine_bytes
        a = hash_embedding('the deployment failed yesterday')
        b = hash_embedding('deployments failing')
        c = hash_embedding('recipe for banana bread')
        self.assertGreater(_cosine_bytes(a, b), _cosine_bytes(a, c))

    def test_related_and_conflict_ranks_similar_higher(self):
        embedder = Embedder(mode='hash')
        others = [('id1', 'User prefers dark mode for coding'),
                  ('id2', 'Banana bread recipe uses three bananas')]
        related, conflict, dup = related_and_conflict(
            'User does not prefer dark mode for coding anymore', others, embedder)
        self.assertTrue(related)
        self.assertEqual(related[0]['key'], 'id1')
        self.assertTrue(conflict)
        self.assertIsNone(dup)

    def test_embedder_hash_mode_never_touches_onnx(self):
        embedder = Embedder(mode='hash', model_dir=self.root / 'nonexistent')
        vec = embedder.embed('anything')
        self.assertEqual(len(vec), 2048 * 4)

    def test_embedder_falls_back_when_model_missing(self):
        embedder = Embedder(mode='auto', model_dir=self.root / 'missing-model')
        vec = embedder.embed('fallback path')
        self.assertEqual(len(vec), 2048 * 4)


class EmbeddingCacheTests(unittest.TestCase):
    """A repeated query must not be re-embedded — but never a *different* query."""

    def _counting(self, size=4):
        emb = Embedder(mode="hash", cache_size=size)
        calls = {"n": 0}
        original = emb._embed_uncached

        def counted(text):
            calls["n"] += 1
            return original(text)

        emb._embed_uncached = counted
        return emb, calls

    def test_a_repeated_query_is_embedded_once(self):
        emb, calls = self._counting()
        first = emb.embed("where does staging run")
        second = emb.embed("where does staging run")
        self.assertEqual(calls["n"], 1)
        self.assertEqual(first, second)

    def test_different_queries_are_both_embedded(self):
        emb, calls = self._counting()
        emb.embed("where does staging run")
        emb.embed("how often do backups run")
        self.assertEqual(calls["n"], 2)

    def test_whitespace_does_not_defeat_the_cache(self):
        emb, calls = self._counting()
        emb.embed("where does staging run")
        emb.embed("  where   does staging   run  ")
        self.assertEqual(calls["n"], 1)

    def test_a_near_miss_is_not_a_hit(self):
        """Exact match only: an approximate key would answer the wrong question."""
        emb, calls = self._counting()
        emb.embed("where does staging run")
        emb.embed("where does production run")
        self.assertEqual(calls["n"], 2)

    def test_the_cache_is_bounded_and_evicts_oldest(self):
        emb, calls = self._counting(size=2)
        for text in ("a one", "b two", "c three"):
            emb.embed(text)
        info = emb.cache_info()
        self.assertEqual(info["size"], 2)
        emb.embed("a one")                  # evicted, so embedded again
        self.assertEqual(calls["n"], 4)

    def test_zero_disables_the_cache(self):
        emb, calls = self._counting(size=0)
        emb.embed("where does staging run")
        emb.embed("where does staging run")
        self.assertEqual(calls["n"], 2)
        self.assertEqual(emb.cache_info()["size"], 0)

    def test_hits_and_misses_are_counted(self):
        emb, _calls = self._counting()
        emb.embed("x query")
        emb.embed("x query")
        emb.embed("y query")
        self.assertEqual(emb.cache_info()["hits"], 1)
        self.assertEqual(emb.cache_info()["misses"], 2)

    def test_a_cached_vector_is_identical_not_recomputed(self):
        emb = Embedder(mode="hash", cache_size=8)
        first = emb.embed("backups run nightly")
        self.assertEqual(first, hash_embedding("backups run nightly"))
        self.assertEqual(emb.embed("backups run nightly"), first)

    def test_recall_benefits_from_a_warm_query_cache(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            store = ProposalStore(Path(tmp.name) / "f.sqlite",
                                  embedder=Embedder(mode="hash", cache_size=32))
            proposal = store.propose({"kind": "fact", "claim": "Staging stays on Fly.io",
                                      "evidence": "said"}, user_text="said",
                                     session_id="s")
            store.accept(proposal["id"])
            query = "where does staging run"
            store.recall(query)              # cold: query embedded once
            before = store.embedder.cache_info()["hits"]
            store.recall(query)              # warm: query embedding served from cache
            after = store.embedder.cache_info()["hits"]
            self.assertEqual(after, before + 1)
        finally:
            tmp.cleanup()


class VaultTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / 'vault'
        self.memory_rel = Path('Hermes/Memory/Auto')
        (self.root / 'home').mkdir()
        self._old_env = os.environ.get('HERMES_HOME')
        os.environ['HERMES_HOME'] = str(self.root / 'home')
        (self.root / 'home' / 'hwiki.json').write_text(json.dumps({
            'wiki_path': str(self.vault), 'memory_dir': str(self.memory_rel)}))

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop('HERMES_HOME', None)
        else:
            os.environ['HERMES_HOME'] = self._old_env
        self._tmp.cleanup()

    def test_resolve_vault_reads_hwiki_json(self):
        vault_root, memory_rel = resolve_vault()
        self.assertEqual(vault_root, self.vault)
        self.assertEqual(memory_rel, Path('Hermes/Memory/Auto'))

    def test_fact_page_roundtrip(self):
        proposal = {
            'id': 'abc123', 'claim': 'User prefers dark mode', 'kind': 'preference',
            'project': 'web', 'session_id': 's1', 'evidence': 'said so',
            'accepted_at': 1700000000.0,
        }
        page = write_fact_page(self.vault, self.memory_rel, proposal)
        self.assertTrue(page.exists())
        facts = read_fact_pages(self.vault, self.memory_rel)
        self.assertEqual([f['claim'] for f in facts], ['User prefers dark mode'])
        self.assertEqual(facts[0]['kind'], 'preference')
        self.assertEqual(facts[0]['project'], 'web')

    def test_superseded_pages_are_hidden(self):
        old = {'id': 'old1', 'claim': 'Deploys to Fly.io', 'kind': 'environment',
               'project': '', 'session_id': 's', 'evidence': 'e', 'accepted_at': 1.0}
        new = {'id': 'new1', 'claim': 'Deploys to Hetzner VPS', 'kind': 'environment',
               'project': '', 'session_id': 's', 'evidence': 'e', 'accepted_at': 2.0}
        old_page = write_fact_page(self.vault, self.memory_rel, old)
        write_fact_page(self.vault, self.memory_rel, new)
        self.assertTrue(mark_superseded_page(old_page, new['claim']))
        claims = [f['claim'] for f in read_fact_pages(self.vault, self.memory_rel)]
        self.assertEqual(claims, ['Deploys to Hetzner VPS'])
        text = old_page.read_text()
        self.assertIn('status: superseded', text)
        self.assertIn('superseded_by', text)

    def test_render_profile_is_bounded(self):
        facts = [{'claim': f'Fact number {i}', 'kind': 'milestone', 'project': 'p',
                  'accepted_at': '', 'path': ''} for i in range(200)]
        rendered = render_profile(facts, max_chars=800)
        self.assertLessEqual(len(rendered), 800)
        self.assertIn('Fact number 0', rendered)

    def test_render_profile_groups_by_project(self):
        facts = [
            {'claim': 'Web uses Svelte', 'kind': 'decision', 'project': 'web', 'accepted_at': '', 'path': ''},
            {'claim': 'Ops runs on Hetzner', 'kind': 'environment', 'project': 'ops', 'accepted_at': '', 'path': ''},
            {'claim': 'Likes terse replies', 'kind': 'preference', 'project': '', 'accepted_at': '', 'path': ''},
        ]
        rendered = render_profile(facts)
        self.assertIn('### web', rendered)
        self.assertIn('### ops', rendered)
        self.assertIn('### general', rendered)
        self.assertIn('Snapshot from session start', rendered)  # snapshot wording present
        single = render_profile([facts[0]])
        self.assertNotIn('###', single)  # one project -> no grouping

    def test_read_fact_pages_orders_preferences_first(self):
        milestone = {'id': 'm1', 'claim': 'Shipped version two', 'kind': 'milestone',
                     'project': '', 'session_id': 's', 'evidence': 'e', 'accepted_at': 1.0}
        preference = {'id': 'p1', 'claim': 'User prefers terse answers', 'kind': 'preference',
                      'project': '', 'session_id': 's', 'evidence': 'e', 'accepted_at': 2.0}
        write_fact_page(self.vault, self.memory_rel, milestone)
        write_fact_page(self.vault, self.memory_rel, preference)
        claims = [f['claim'] for f in read_fact_pages(self.vault, self.memory_rel)]
        self.assertEqual(claims.index('User prefers terse answers'), 0)

    def test_session_diary_write(self):
        page = write_session_diary(
            self.vault, self.memory_rel, session_id='sess-abc123',
            summary='Set up the extractor plugin and accepted two facts.',
            facts=['User prefers dark mode'], topics=['extractor', 'plugin'])
        self.assertIsNotNone(page)
        self.assertTrue(Path(page).exists())
        text = Path(page).read_text()
        self.assertIn('extractor: hwiki-extractor', text)
        self.assertIn('Set up the extractor plugin', text)

    def test_load_builtin_entries_splits_on_delimiter(self):
        mem_dir = self.root / 'memories'
        mem_dir.mkdir()
        (mem_dir / 'USER.md').write_text(
            'First entry about editor\n§\nSecond entry about RAM\n', encoding='utf-8')
        entries = load_builtin_entries(mem_dir)
        self.assertEqual([e['text'] for e in entries],
                         ['First entry about editor', 'Second entry about RAM'])
        self.assertEqual({e['target'] for e in entries}, {'user'})


if __name__ == '__main__':
    unittest.main()


class GuardrailsAndUsageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_guardrails_render_first(self):
        facts = [
            {'claim': 'Prefers dark mode', 'kind': 'preference', 'peer': 'user',
             'project': '', 'accepted_at': '2', 'path': '', 'proposal_id': 'p1'},
            {'claim': 'Never paste API keys in chat', 'kind': 'blocker', 'peer': 'user',
             'project': '', 'accepted_at': '3', 'path': '', 'proposal_id': 'p2'},
            {'claim': 'Deploys to Fly.io', 'kind': 'environment', 'peer': 'user',
             'project': '', 'accepted_at': '1', 'path': '', 'proposal_id': 'p3'},
        ]
        rendered = render_profile(facts)
        self.assertIn('### Guardrails (never do)', rendered)
        self.assertLess(rendered.index('Never paste API keys'), rendered.index('Prefers dark mode'))

    def test_usage_counts_increment_and_rank(self):
        store = ProposalStore(self.root / 'db.sqlite', embedder=Embedder(mode='hash'))
        a = store.propose({'kind': 'preference', 'claim': 'User prefers vim keybindings',
                           'evidence': 'e'}, user_text='x', session_id='s')
        b = store.propose({'kind': 'preference', 'claim': 'User prefers short commit messages',
                           'evidence': 'e'}, user_text='x', session_id='s')
        for p in (a, b):
            store.accept(p['id'])
        store.recall('vim keybindings')   # hits a
        store.recall('vim keybindings')   # hits a again
        usage = store.usage_counts()
        self.assertGreater(usage.get(a['id'], 0), usage.get(b['id'], 0))
        # usage breaks kind ties in the profile ordering
        facts = [
            {'claim': 'User prefers short commit messages', 'kind': 'preference', 'peer': 'user',
             'project': '', 'accepted_at': '1', 'path': '', 'proposal_id': b['id']},
            {'claim': 'User prefers vim keybindings', 'kind': 'preference', 'peer': 'user',
             'project': '', 'accepted_at': '2', 'path': '', 'proposal_id': a['id']},
        ]
        rendered = render_profile(facts, usage=usage)
        self.assertLess(rendered.index('vim keybindings'), rendered.index('short commit messages'))

    def test_audit_page_appends_and_trims(self):
        from hwiki.facts import append_audit_entry, AUDIT_MAX_LINES, AUDIT_KEEP_LINES
        vault = self.root / 'vault'
        memory_rel = Path('Hermes/Memory/Auto')
        page = append_audit_entry(vault, memory_rel, 'accepted fact "X" (preference)')
        self.assertTrue(page.exists())
        self.assertNotIn('extractor: hwiki-extractor', page.read_text())  # never a fact page
        for i in range(AUDIT_MAX_LINES + 20):
            append_audit_entry(vault, memory_rel, f'entry {i}')
        lines = page.read_text().splitlines()
        self.assertLessEqual(len(lines), AUDIT_MAX_LINES + 2)
        self.assertIn('entry %d' % (AUDIT_MAX_LINES + 19), page.read_text())  # newest kept
        self.assertNotIn('entry 0\n', page.read_text())  # oldest trimmed


class V25Tests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _seed(self, store):
        ids = []
        for f in [
            {'kind': 'preference', 'claim': "User's main editor is VS Code", 'evidence': 'e1'},
            {'kind': 'environment', 'claim': 'Staging server stays on Fly.io', 'evidence': 'e2'},
            {'kind': 'preference', 'claim': 'User wants short chat replies', 'evidence': 'e3'},
            {'kind': 'environment', 'claim': "User's laptop has 8 GB of RAM", 'evidence': 'e4'},
            {'kind': 'environment', 'claim': 'Staging database runs Postgres 15', 'evidence': 'e5'},
        ]:
            p = store.propose(f, user_text='x', session_id='s')
            store.accept(p['id'])
            ids.append(p['id'])
        return ids

    def test_broad_query_returns_cross_kind_coverage(self):
        store = ProposalStore(self.root / 'db.sqlite', embedder=Embedder(mode='hash'))
        self._seed(store)
        hits = store.recall('give an overview of the user preferences and setup', limit=8)
        kinds = {h['kind'] for h in hits}
        self.assertEqual(len(hits), 5)          # everything, not just the top cluster
        self.assertEqual(len(kinds), 2)          # both kinds represented
        # narrow queries still use the strict margin rule
        narrow = store.recall('which database does staging use')
        self.assertLessEqual(len(narrow), 3)

    def test_temporal_window_demotes_old_facts(self):
        store = ProposalStore(self.root / 'db.sqlite', embedder=Embedder(mode='hash'))
        ids = self._seed(store)
        old_id = ids[1]  # staging stays on Fly.io
        conn = sqlite3.connect(self.root / 'db.sqlite')
        conn.execute("UPDATE proposals SET accepted_at=? WHERE id=?",
                     (time.time() - 40 * 86400, old_id))  # 40 days old
        conn.commit(); conn.close()
        hits = store.recall('staging this week', limit=8)
        ranked = [h['id'] for h in hits]
        self.assertIn(old_id, ranked)            # demoted, not hidden
        self.assertNotEqual(ranked[0], old_id)   # a fresher staging fact leads

    def test_never_used_older_than_finds_decay_candidates(self):
        store = ProposalStore(self.root / 'db.sqlite', embedder=Embedder(mode='hash'))
        ids = self._seed(store)
        old_id = ids[2]
        conn = sqlite3.connect(self.root / 'db.sqlite')
        conn.execute("UPDATE proposals SET accepted_at=? WHERE id=?",
                     (time.time() - 120 * 86400, old_id))
        conn.commit(); conn.close()
        candidates = store.never_used_older_than(days=90)
        self.assertEqual(len(candidates), 1)
        self.assertIn('short chat replies', candidates[0])
        store.recall('short chat replies')  # using the fact clears it
        self.assertEqual(store.never_used_older_than(days=90), [])

    def test_canvas_generation(self):
        import json as _json
        store = ProposalStore(self.root / 'db.sqlite', embedder=Embedder(mode='hash'))
        vault = self.root / 'vault'
        memory_rel = Path('Hermes/Memory/Auto')
        old = store.propose({'kind': 'environment', 'claim': 'Deploys to Fly.io',
                             'evidence': 'e'}, user_text='x', session_id='s')
        store.accept(old['id'])
        write_fact_page(vault, memory_rel, store.get(old['id']))
        new = store.propose({'kind': 'environment', 'claim': 'Deploys to Hetzner VPS',
                             'evidence': 'e'}, user_text='x', session_id='s')
        store.accept(new['id'])
        store.supersede(new['id'], old['id'])
        page = write_fact_canvas(store, vault, memory_rel)
        self.assertIsNotNone(page)
        canvas = _json.loads(page.read_text())
        self.assertEqual(len(canvas['nodes']), 2)
        edges = canvas['edges']
        self.assertTrue(any(e['label'] == 'superseded' for e in edges))
        node_ids = {n['id'] for n in canvas['nodes']}
        self.assertTrue(all(e['fromNode'] in node_ids and e['toNode'] in node_ids for e in edges))
        # empty store -> no canvas
        empty = ProposalStore(self.root / 'empty.sqlite', embedder=Embedder(mode='hash'))
        self.assertIsNone(write_fact_canvas(empty, vault, memory_rel))

    def test_duplicate_of_long_quoted_builtin_entry(self):
        """Real builtin entries carry quotes and history — dedup must not miss them.

        Pinned to explicit embeddings so the assertion is deterministic across
        test contexts (the auto tier depends on machine model availability)."""
        import unittest as _u
        from hwiki.facts import Embedder
        real_models = Path(os.path.expanduser(
            '~/.hermes/plugins-data/hwiki-extractor/models/minilm'))
        if not (real_models / 'model.onnx').exists():
            self.skipTest('local MiniLM model not available')
        store = ProposalStore(self.root / 'proposals.sqlite',
                              embedder=Embedder(mode='onnx', model_dir=real_models))
        long_entry = ("User wants no periodic reminder/progress pings in chat — Honcho recovery "
                      "updates, hourly cron progress reports, and system reminder services were "
                      "all silenced on 2026-11-07; don't recreate or re-enable them without asking.")
        from hwiki.facts import _cosine_bytes, _tokens, related_and_conflict as _rac
        import sys as _s
        print('DEBUG tier:', 'onnx' if store.embedder._onnx is not None else 'hash', file=_s.stderr)
        t120 = long_entry[:120]
        cos = _cosine_bytes(store.embedder.embed(claim_probe := 'No periodic reminder pings in chat'), store.embedder.embed(t120))
        jac = len(set(_tokens(claim_probe)) & set(_tokens(t120))) / max(1, len(set(_tokens(claim_probe)) | set(_tokens(t120))))
        print('DEBUG cos/jac/blended:', round(cos, 4), round(jac, 4), round(max(cos, 0.5 * cos + 0.5 * jac), 4), file=_s.stderr)
        proposal = store.propose(
            {'kind': 'preference', 'claim': claim_probe,
             'evidence': 'user silenced them all'},
            user_text='x', session_id='s',
            builtin_entries=[{'text': long_entry, 'target': 'user'}])
        print('DEBUG dup:', repr(proposal['duplicate_of_builtin'])[:60], file=_s.stderr)
        self.assertTrue(proposal['duplicate_of_builtin'])
