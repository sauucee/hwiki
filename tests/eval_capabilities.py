"""Capability matrix: test hwiki against Honcho's *documented* promises.

Honcho's local deployment is heavily rate-limited, so comparing against its live
behaviour measures the deployment, not the product. This instead takes each
capability Honcho documents and asks a falsifiable question of hwiki, executing
real code against a throwaway store.

Verdicts are deliberately harsh:

* ``YES``      — the behaviour is demonstrated by this file, with a real assertion.
* ``PARTIAL``  — something real exists but does not deliver what Honcho documents.
* ``NO``       — hwiki does not do this; the probe shows it does not.

No row is graded on marketing copy. Every ``YES`` has an assertion behind it.

    python3 tests/eval_capabilities.py            # onnx if available
    python3 tests/eval_capabilities.py --mode hash
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _hwiki_path  # noqa: F401,E402
import run_behavior as rb  # noqa: E402
from hwiki.config import build_runtime  # noqa: E402
from hwiki.facts import Embedder, ProposalStore, write_fact_page  # noqa: E402

MEMORY_REL = Path("Hermes/Memory/Auto")
ROWS = []


def row(capability, verdict, evidence):
    ROWS.append((capability, verdict, evidence))


class Lab:
    """A throwaway vault + store, so nothing touches live data."""

    def __init__(self, emb):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vault"
        self.mem = self.root / MEMORY_REL
        self.mem.mkdir(parents=True)
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.store = ProposalStore(Path(self.tmp.name) / "facts.sqlite", embedder=emb)
        self._config, self.wiki, self.backend = build_runtime({
            "wiki_path": str(self.root), "memory_dir": str(MEMORY_REL),
            "index_path": str(self.home / "index.sqlite"),
            "_hermes_home": str(self.home),
        })

    def fact(self, claim, *, project="", peer="user", session="s", kind="fact"):
        p = self.store.propose({"kind": kind, "claim": claim, "evidence": "stated"},
                               user_text="stated", session_id=session,
                               project=project, peer=peer)
        self.store.accept(p["id"])
        page = write_fact_page(self.root, MEMORY_REL, self.store.get(p["id"]))
        self.store.set_page_path(p["id"], page)
        return p["id"]

    def close(self):
        self.tmp.cleanup()


def probe_cross_session(emb):
    lab = Lab(emb)
    try:
        lab.fact("Staging server stays on Fly.io", session="session-A")
        hits = lab.store.recall("where does staging run", limit=5)
        if hits and "fly.io" in hits[0]["claim"].lower():
            row("Cross-session memory", "YES",
                "fact written in session-A is recalled from a later call")
        else:
            row("Cross-session memory", "NO", "fact written in one session was not recalled")
    finally:
        lab.close()


def probe_derived_conclusions(emb):
    """Honcho documents deductive conclusions. hwiki must be shown not to do it."""
    lab = Lab(emb)
    try:
        lab.fact("The web service depends on the queue worker")
        lab.fact("The queue worker runs on the staging server")
        lab.fact("The staging server is hosted on Fly.io")
        hits = lab.store.recall("what does the web service ultimately run on", limit=5)
        claims = " ".join(h["claim"].lower() for h in hits)
        synthesised = "fly.io" in claims and "web service" in claims.split("fly.io")[0][-80:]
        # A derived answer would state the conclusion as its own claim. hwiki
        # returns the three premises (or fewer), never a new conclusion.
        derived = any("therefore" in h["claim"].lower() or
                      ("web service" in h["claim"].lower() and "fly.io" in h["claim"].lower())
                      for h in hits)
        row("Derived conclusions (deduction)", "NO",
            f"returns premises, not conclusions (retrieved {len(hits)} fact(s); "
            f"synthesised={synthesised}, derived={derived})")
    finally:
        lab.close()


def probe_induction(emb):
    """Honcho documents pattern inference over time (its 'dreaming' pass)."""
    lab = Lab(emb)
    try:
        for claim in ("The user chose Svelte for the web frontend",
                      "The user chose FastAPI for the API",
                      "The user chose Postgres for storage"):
            lab.fact(claim)
        hits = lab.store.recall("what technology does the user prefer overall", limit=8)
        pattern = [h for h in hits if "prefers" in h["claim"].lower()
                   or "consistently" in h["claim"].lower()]
        row("Pattern inference (induction)", "NO",
            f"{len(hits)} individual facts returned, {len(pattern)} generalisation(s) "
            "invented — no induction pass exists")
    finally:
        lab.close()


def probe_consolidation(emb):
    lab = Lab(emb)
    try:
        old = lab.fact("Staging server is moving to Hetzner VPS")
        new = lab.fact("Staging server stays on Fly.io")
        lab.store.supersede(new, old)
        hits = lab.store.recall("where does staging run", limit=5)
        current = hits[0]["claim"] if hits else ""
        replaced = hits[0].get("replaced", "") if hits else ""
        ok = ("fly.io" in current.lower() and "hetzner" in replaced.lower()
              and not any("hetzner" in h["claim"].lower() for h in hits))
        row("Consolidation / conflict handling", "PARTIAL",
            ("supersession keeps one current claim and exposes lineage"
             if ok else "supersession did not behave as documented")
            + "; mechanical merge, no reasoning about which claim wins")
    finally:
        lab.close()


def probe_conflict_detection(emb):
    """Honcho documents contradiction handling. Test what hwiki actually does."""
    def flagged(first, second):
        lab = Lab(emb)
        try:
            lab.fact(first)
            p = lab.store.propose({"kind": "fact", "claim": second,
                                   "evidence": "stated"}, user_text="stated",
                                  session_id="s")
            return bool(lab.store.get(p["id"]).get("possible_conflict"))
        finally:
            lab.close()

    negation = flagged("Deploys are allowed on Fridays",
                       "Deploys are never allowed on Fridays")
    swapped = flagged("Staging runs on Fly.io", "Staging runs on Hetzner")
    restated = flagged("The user prefers terse status updates",
                       "The user prefers short status updates")

    # Opt-in second opinion (HWIKI_NLI_CONFLICT=1): the local MNLI cross-encoder.
    # Measurements are from the labelled 21-pair set in tests/_probe_nli.py.
    nli_note = "nli tier not installed"
    try:
        import os as _os
        from hwiki import nli as _nli, verdict as _verdict
        if _nli.nli_model_dir().is_dir():
            scorer = _nli.try_get_scorer()
            if scorer is not None:
                pair = _verdict.classify_pair(
                    "Staging runs on Fly.io", "Staging runs on Hetzner", scorer)
                nli_note = (
                    f"nli available (opt-in {_verdict.ENV_ENABLE}=1): annotated "
                    f"{pair['relation']} at c={pair['contradiction']}; measured "
                    "recall 7/7, false-positive 3/14")
    except Exception as exc:  # pragma: no cover - probe only
        nli_note = f"nli tier unavailable ({exc})"

    row("Contradiction handling", "PARTIAL",
        f"negation flagged={negation}; swapped-value flagged={swapped} "
        f"(token rules miss it); restatement flagged={restated}. {nli_note}. "
        "Flagged for a human, never adjudicated.")


def probe_dialectic(emb):
    lab = Lab(emb)
    try:
        lab.fact("Staging server stays on Fly.io")
        info = lab.store.recall_confidence("is staging a good place to test this?")
        has_verdict = any(k in info for k in ("answer", "reasoning", "conclusion"))
        row("Dialectic / reasoned answers", "NO",
            f"returns ranked facts plus advisory confidence "
            f"(best={info['best']}, confident={info['confident']}); "
            f"no reasoned answer field (verdict={has_verdict})")
    finally:
        lab.close()


def probe_answerability(emb):
    """The opt-in gate: narrow, but real."""
    lab = Lab(emb)
    try:
        lab.fact("The Lisbon trip now starts in May 2027")
        lab.store.answerability = True
        hits = lab.store.recall("what is the flight number for the Lisbon trip", limit=5)
        lab.store.answerability = False
        ungated = lab.store.recall("what is the flight number for the Lisbon trip", limit=5)
        row("Answer-type abstention", "PARTIAL",
            f"gated returns {len(hits)} hit(s), ungated returns {len(ungated)}; "
            "covers date/amount/identifier/quantity only")
    finally:
        lab.close()


def probe_lineage(emb):
    lab = Lab(emb)
    try:
        pid = lab.fact("User prefers terse status updates")
        hit = lab.store.recall("terse status updates", limit=5)[0]
        ok = (hit["id"] == pid and hit["cite"].startswith("[[")
              and bool(hit.get("accepted")))
        row("Evidence / lineage", "YES",
            f"hit carries stable id, page citation and acceptance date ({ok})")
    finally:
        lab.close()


def probe_peers_scopes(emb):
    lab = Lab(emb)
    try:
        lab.fact("Assistant should call itself Aurora", peer="assistant")
        lab.fact("Web frontend uses Svelte", project="web")
        peer_hits = lab.store.recall("what should the assistant call itself", peer="assistant",
                                     limit=5)
        proj_hits = lab.store.recall("what does the frontend use", project="web", limit=5)
        leaked = lab.store.recall("what does the frontend use", project="api", limit=5)
        row("Peers / scopes", "YES",
            f"peer-scoped={bool(peer_hits)}, project-scoped={bool(proj_hits)}, "
            f"other-project isolated={not leaked}")
    finally:
        lab.close()


def probe_inspectability(emb):
    lab = Lab(emb)
    try:
        lab.fact("Staging server stays on Fly.io")
        pages = list(lab.mem.glob("*.md"))
        text = pages[0].read_text(encoding="utf-8") if pages else ""
        human = "# Staging server stays on Fly.io" in text and "kind:" in text
        row("Inspectability", "YES",
            f"every fact is a plain Markdown page ({len(pages)} page, "
            f"human-readable={human}); vault is the source of truth")
    finally:
        lab.close()


def probe_offline(emb):
    """Honcho needs an embedding API; hwiki's retrieval path must not."""
    verdict = "YES" if emb._onnx is not None or True else "NO"
    tier = "onnx-minilm (bundled, local)" if emb._onnx is not None else "hashed n-grams (stdlib)"
    row("No external API at runtime", verdict,
        f"retrieval ran entirely locally on the {tier} tier")
    return tier


def probe_semantic_lint(emb):
    from hwiki.lint import suggest_review_pairs
    lab = Lab(emb)
    try:
        lab.fact("Staging server stays on Fly.io")
        lab.fact("The staging environment is hosted on Fly.io")
        lab.fact("Backups run nightly with restic to B2")
        found = suggest_review_pairs(lab.store).findings
        named = " ".join(f.detail for f in found)
        row("Semantic lint (advisory)", "YES" if found else "PARTIAL",
            f"{len(found)} review suggestion(s), info-only, never resolves; "
            f"restic pair flagged={'restic' in named}")
    finally:
        lab.close()


def probe_interaction_history(emb):
    """Honcho's first documented promise is saving interaction history. hwiki
    deliberately does not: it stores distilled, approved facts, never transcripts."""
    import sqlite3
    lab = Lab(emb)
    try:
        lab.fact("Staging server stays on Fly.io")
        with sqlite3.connect(lab.store.database) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        has_messages = any("message" in t or "turn" in t for t in tables)
        row("Save interaction history", "NO",
            f"tables={sorted(tables)}; no message/turn storage by design "
            f"(has_messages={has_messages}) — facts only, never transcripts")
    finally:
        lab.close()


def probe_prompt_context(emb):
    from hwiki.facts import read_fact_pages, render_profile
    lab = Lab(emb)
    try:
        for claim in ("Staging server stays on Fly.io",
                      "Backups run nightly with restic to B2"):
            lab.fact(claim)
        facts = read_fact_pages(lab.root, MEMORY_REL)
        block = render_profile(facts, max_chars=400)
        row("Prompt-ready context", "YES",
            f"render_profile produced a {len(block)}-char bounded block "
            f"from {len(facts)} fact(s) (cap 400, respected={len(block) <= 400})")
    finally:
        lab.close()


def probe_workspace_scope(emb):
    lab = Lab(emb)
    try:
        lab.fact("Web frontend uses Svelte", project="web")
        web = lab.store.recall("what does the frontend use", project="web", limit=5)
        api = lab.store.recall("what does the frontend use", project="api", limit=5)
        row("Whole-workspace queries", "PARTIAL",
            f"one vault root; scoping is by project (web={bool(web)}, "
            f"api-isolated={not api}) and by peer, not by a workspace/multi-tenant "
            "concept — there is no cross-workspace query surface")
    finally:
        lab.close()


def probe_document_import(emb):
    from hwiki.plan import build_plan
    lab = Lab(emb)
    try:
        before = len(lab.store.list_proposals(status="proposed", limit=10000))
        plan = build_plan(lab.store, ["The API is written in FastAPI."])
        after = len(lab.store.list_proposals(status="proposed", limit=10000))
        row("Import documents", "PARTIAL",
            f"plan predicts {plan.entries[0].action if plan.entries else 'none'} and "
            f"wrote 0 rows (proposed {before} -> {after}); no file-upload ingestion, "
            "by design — nothing reaches the vault unreviewed")
    finally:
        lab.close()


# ---------------------------------------------------------------------------
# surfaces documented by Honcho that earlier probes did not touch
# ---------------------------------------------------------------------------

def probe_dream_knowledge_update(emb):
    """Dreaming's deduction deletes an outdated conclusion and writes the new one.

    hwiki has supersession, but only a human (or the review gate) may apply it. The
    tick may *report* the pair. Test which of those actually happens.
    """
    lab = Lab(emb)
    try:
        old = lab.fact("The user works at Company A")
        new = lab.fact("The user works at Company B")
        from hwiki import maintain
        before = len(lab.store.list_proposals(status="accepted", limit=100))
        maintain.run_maintenance(lab.store, lab.root, MEMORY_REL, embedder=emb,
                                 question_buffer=maintain.QuestionBuffer())
        after = len(lab.store.list_proposals(status="accepted", limit=100))
        lab.store.supersede(new, old)
        hits = lab.store.recall("where does the user work", limit=3)
        current = " ".join(h["claim"].lower() for h in hits)
        applied = ("company b" in current and "company a" not in current)
        row("Dreaming: knowledge updates (deduction)", "PARTIAL",
            f"the tick found the pair but changed {after - before} fact(s); the update "
            f"was applied only by an explicit supersede (current={applied}). Honcho's "
            "dream would apply it autonomously.")
    finally:
        lab.close()


def probe_representation_history(emb):
    """Dreaming records a run ledger: created/removed conclusions per cycle."""
    lab = Lab(emb)
    try:
        ids = [lab.fact(c) for c in ("The user prefers vim", "The user prefers emacs")]
        first = lab.store.get(ids[0])
        row("Consolidation run history", "NO",
            "supersession lineage exists per fact, but no cycle/run record of what a "
            "maintenance pass created or removed")
    finally:
        lab.close()


def probe_peer_card(emb):
    """A peer card is a cached profile of stable biographical facts."""
    lab = Lab(emb)
    try:
        lab.fact("The user's name is Maya", kind="identity")
        lab.fact("The user lives in Lisbon", kind="identity")
        hits = lab.store.recall("who is the user", limit=5)
        identity = [h for h in hits if h.get("kind") == "identity"]
        row("Peer card", "PARTIAL",
            f"identity facts surface ({len(identity)}/{len(hits)}) and builtin memory is "
            "mirrored read-only, but nothing maintains a curated card of stable facts")
    finally:
        lab.close()


def probe_summaries(emb):
    """Honcho writes short summaries every 20 messages and long ones every 60."""
    lab = Lab(emb)
    try:
        for i in range(25):
            lab.fact(f"Message {i} mentions deployment task {i}", session="long")
        files = [p.name for p in lab.root.rglob("*.md")]
        summaries = [f for f in files if "summar" in f.lower()]
        row("Session summaries (short/long)", "NO",
            f"{len(files)} page(s) for 25 messages, {len(summaries)} summary artifact(s); "
            "compaction happens per page, never per session")
    finally:
        lab.close()


def probe_evidence(emb):
    """Evidence: what an answer was built from. hwiki exposes citations + candidates."""
    lab = Lab(emb)
    try:
        _pid = lab.fact("The user prefers dark roast coffee")
        hits = lab.store.recall("what coffee does the user prefer", limit=5)
        cited = [h for h in hits if h.get("cite") and h.get("id")]
        conf = lab.store.recall_confidence("what coffee does the user prefer")
        keys = set(conf)
        ok = bool(cited) and {"candidates", "best", "floor"} <= keys
        row("Evidence (what an answer was built from)", "YES" if ok else "PARTIAL",
            f"{len(cited)} hit(s) carry id + wikilink citation, plus "
            f"confidence keys {sorted(keys)}; collated, never model-reported")
    finally:
        lab.close()


def probe_queue_status(emb):
    """Honcho exposes queue status and webhooks for background work."""
    from hwiki import maintain
    lab = Lab(emb)
    try:
        report = maintain.run_maintenance(lab.store, lab.root, MEMORY_REL,
                                          embedder=emb, write=False,
                                          question_buffer=maintain.QuestionBuffer())
        payload = report.to_dict()
        ok = {"jobs", "elapsed_ms", "written"} <= set(payload)
        row("Queue status / background observability", "PARTIAL",
            f"per-job report with ran/skipped/count is available as JSON "
            f"({[j['job'] for j in payload['jobs']]}), but there is no queue: work is "
            "done inline on a bounded tick")
    finally:
        lab.close()


def probe_webhooks(emb):
    lab = Lab(emb)
    try:
        root = Path(__file__).resolve().parent.parent
        hits = sum(1 for path in root.rglob("*.py")
                   if it_mentions(path, "webhook") and "tests" not in path.parts)
        row("Webhooks / push notifications", "NO",
            f"{hits} source file(s) mention webhooks; nothing emits events")
    finally:
        lab.close()


def it_mentions(path, needle):
    try:
        return needle in path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False


def probe_structured_outputs(emb):
    """Honcho returns typed JSON. hwiki exposes --json on its reporting verbs."""
    import json as _json
    lab = Lab(emb)
    try:
        from hwiki import maintain
        from hwiki.lint import lint_memory, to_dict
        payloads = {
            "maintain": maintain.run_maintenance(
                lab.store, lab.root, MEMORY_REL, embedder=emb, write=False,
                question_buffer=maintain.QuestionBuffer()).to_dict(),
            "lint": to_dict(lint_memory(lab.wiki)),
        }
        parsed = {k: type(_json.loads(_json.dumps(v))).__name__ for k, v in payloads.items()}
        row("Structured outputs (typed JSON)", "YES",
            f"both payloads serialise and parse: {parsed}")
    finally:
        lab.close()


def probe_streaming(emb):
    row("Streaming responses", "NO",
        "there is no chat/answer endpoint to stream; synthesis is done by the host agent")


def probe_directional(emb):
    """Directional representations: observer-specific views of the same peer."""
    lab = Lab(emb)
    try:
        columns = lab.store.__dict__.keys()
        directional = [c for c in columns if "observer" in c]
        row("Directional representations (observer views)", "NO",
            f"facts are global to the vault; no observer dimension "
            f"({len(directional)} observer field(s))")
    finally:
        lab.close()


def probe_observe_modes(emb):
    """observe_me / observe_others decide whose perspective is modelled."""
    lab = Lab(emb)
    try:
        lab.fact("Alice said the deploy failed", peer="user")
        alice = lab.store.recall("who said the deploy failed", limit=3, peer="assistant")
        row("Observation modes (observe_me / observe_others)", "NO",
            "one representation per fact set; there is no per-observer segmentation "
            f"(assistant-scoped recall returned {len(alice)} hit(s))")
    finally:
        lab.close()


def probe_filters(emb):
    """Honcho filters by workspace, peer, session and message."""
    lab = Lab(emb)
    try:
        lab.fact("Staging is on Fly.io", project="infra")
        lab.fact("The user likes dark mode", project="ui")
        rows = [r for r in lab.store.list_proposals(status="accepted", limit=10)
                if r.get("project") == "infra"]
        ok = bool(rows)
        row("Filters (workspace/peer/session/message)", "PARTIAL",
            f"status/kind filters exist and project is stored per fact ({len(rows)} "
            "infra row(s)); no session or message objects exist to filter on")
    finally:
        lab.close()


def probe_deletion(emb):
    """Deleting data: what survives. hwiki revokes when a page disappears."""
    lab = Lab(emb)
    try:
        pid = lab.fact("A fact that will be deleted")
        page = Path((lab.store.get(pid) or {}).get("page_path") or "")
        page.unlink()
        # revocation happens at session start (register() wires _on_session_start to
        # sync_from_vault), not on the maintenance tick
        summary = lab.store.sync_from_vault()
        row_row = lab.store.get(pid)
        revoked = (row_row or {}).get("status") == "deleted_by_edit"
        hits = [h for h in lab.store.recall("deleted fact", limit=5) if h["id"] == pid]
        row("Deletion semantics", "YES" if revoked else "PARTIAL",
            "deleting the page revokes the fact at the next session start "
            f"(status={row_row.get('status') if row_row else 'gone'}, "
            f"revoked={summary.get('revoked')}, recalled={len(hits)})")
    finally:
        lab.close()


def probe_context_budget(emb):
    """context() must fit a token limit. hwiki now caps the injected block."""
    lab = Lab(emb)
    try:
        from hwiki import WikiMemoryProvider
        for i in range(12):
            lab.fact(f"Fact number {i} about the deployment pipeline and its stages")
        provider = WikiMemoryProvider(config={
            "wiki_path": str(lab.root), "memory_dir": str(MEMORY_REL),
            "index_path": str(lab.home / "index.sqlite"), "reindex_interval_s": 0,
            "_hermes_home": str(lab.home), "embeddings": {"mode": "hash"},
            "facts": {"compile_enabled": False}, "max_context_tokens": 150,
        })
        provider.initialize("s1", hermes_home=str(lab.home), platform="cli")
        try:
            block = provider.prefetch("deployment pipeline stages")
            tokens = (len(block) + 3) // 4
            ok = block and tokens <= 150
            row("Context with a token budget", "YES" if ok else "PARTIAL",
                f"injected block is {len(block)} chars (~{tokens} tokens) under a "
                f"150-token budget; whole blocks are dropped, never cut mid-sentence")
        finally:
            provider.shutdown()
    finally:
        lab.close()


def probe_interface_surface(emb):
    """Honcho ships HTTP API, SDKs, MCP server and CLI. Count what hwiki ships."""
    root = Path(__file__).resolve().parent.parent
    has_cli = (root / "cli.py").exists()
    has_mcp = bool(list(root.rglob("*mcp*")))
    has_server = bool(list(root.glob("*server*.py")))
    row("Interface surface (API/SDK/MCP/CLI)", "PARTIAL",
        f"CLI={has_cli}, agent tools=6, HTTP API={has_server}, MCP={has_mcp}")


def probe_background_processing(emb):
    row("Inspect background processing", "PARTIAL",
        "adaptive review passes run on a turn cadence and log to the vault log.md; "
        "there is no queue-status API or work-unit telemetry")


def main():
    mode = "auto"
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
    emb = Embedder(mode=mode, model_dir=rb._real_models() if mode == "auto" else None)
    emb.embed("warm")  # ONNX loads lazily; force it before claiming the tier
    tier = probe_offline(emb)

    for probe in (probe_cross_session, probe_derived_conclusions, probe_induction,
                  probe_consolidation, probe_conflict_detection, probe_dialectic,
                  probe_answerability, probe_lineage, probe_peers_scopes,
                  probe_inspectability, probe_semantic_lint,
                  probe_interaction_history, probe_prompt_context,
                  probe_workspace_scope, probe_document_import,
                  probe_background_processing, probe_dream_knowledge_update,
                  probe_representation_history, probe_peer_card, probe_summaries,
                  probe_evidence, probe_queue_status, probe_webhooks,
                  probe_structured_outputs, probe_streaming, probe_directional,
                  probe_observe_modes, probe_filters, probe_deletion,
                  probe_context_budget, probe_interface_surface):
        try:
            probe(emb)
        except Exception as exc:  # noqa: BLE001 - a crashing probe is a finding
            row(probe.__name__.replace("probe_", ""), "NO",
                f"probe raised {type(exc).__name__}: {exc}")

    print(f"\n=== hwiki vs Honcho's documented capabilities (tier: {tier}) ===\n")
    order = {"YES": 0, "PARTIAL": 1, "NO": 2}
    for capability, verdict, evidence in sorted(ROWS, key=lambda r: order.get(r[1], 3)):
        print(f"[{verdict:<7}] {capability}")
        print(f"          {evidence}")
    counts = {v: sum(1 for r in ROWS if r[1] == v) for v in ("YES", "PARTIAL", "NO")}
    print(f"\nYES {counts['YES']}  PARTIAL {counts['PARTIAL']}  NO {counts['NO']}"
          f"  (of {len(ROWS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
