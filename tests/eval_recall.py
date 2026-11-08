"""Recall-quality regression eval, ported from the retired extractor.

Seeds a throwaway vault with a fixed fact corpus, runs 40 queries across seven
categories, and scores hit rates per embedding tier. Run before and after any
recall-affecting change:

    <bundled python> tests/eval_recall.py              # both tiers
    <bundled python> tests/eval_recall.py --mode hash  # one tier

Exit code 0 iff every run tier meets its gate (onnx >= 0.90, hash >= 0.78).

Deliberate differences from the v2.5.1 original, so old and new numbers are
never conflated:

* it targets the **live** ``hwiki.facts`` API, not the deleted ``core`` module;
* the tier is asserted *after* an actual embed (ONNX loads lazily, so reading
  the flag before embedding reported "hash" for a working ONNX session);
* if ONNX is unavailable the ``auto`` run is reported as **SKIPPED** and does not
  count as a pass — a silent hash fallback must never look like an ONNX result.

Deterministic: no network, no LLM, fixed seed data, no touch of the live vault
or the live facts database.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _hwiki_path  # noqa: F401,E402

from hwiki.facts import (  # noqa: E402
    Embedder,
    ProposalStore,
    mark_superseded_page,
    write_fact_page,
)

MEMORY_REL = Path("Hermes/Memory/Auto")

# --- fixed fact corpus ------------------------------------------------------
FACTS = [
    {"kind": "preference", "claim": "User's main editor is VS Code", "evidence": "switched from Sublime", "project": ""},
    {"kind": "preference", "claim": "User wants short chat replies and no progress pings", "evidence": "asked explicitly", "project": ""},
    {"kind": "preference", "claim": "User prefers dark mode in every tool", "evidence": "stated", "project": ""},
    {"kind": "preference", "claim": "User takes notes in Obsidian daily", "evidence": "stated", "project": ""},
    {"kind": "environment", "claim": "User's laptop has 8 GB of RAM", "evidence": "constraint", "project": ""},
    {"kind": "environment", "claim": "Staging server stays on Fly.io", "evidence": "move cancelled", "project": "web"},
    {"kind": "environment", "claim": "Staging database runs Postgres 15", "evidence": "stated", "project": "web"},
    {"kind": "environment", "claim": "Production runs on Hetzner VPS", "evidence": "stated", "project": "web"},
    {"kind": "environment", "claim": "CI runners are self-hosted on bare metal", "evidence": "stated", "project": "web"},
    {"kind": "decision", "claim": "Web frontend uses Svelte", "evidence": "chosen", "project": "web"},
    {"kind": "decision", "claim": "API layer uses FastAPI with Pydantic v2", "evidence": "chosen", "project": "web"},
    {"kind": "decision", "claim": "Background jobs use Celery with Redis", "evidence": "chosen", "project": "ops"},
    {"kind": "environment", "claim": "Ops dashboards run Grafana on port 3001", "evidence": "stated", "project": "ops"},
    {"kind": "environment", "claim": "Backups run nightly with restic to B2", "evidence": "stated", "project": "ops"},
    {"kind": "milestone", "claim": "Web v2 launched on 2026-10-01", "evidence": "shipped", "project": "web"},
    {"kind": "blocker", "claim": "Never paste API keys or secrets in chat", "evidence": "standing rule", "project": ""},
    {"kind": "blocker", "claim": "Do not touch production database without a backup", "evidence": "standing rule", "project": "ops"},
    {"kind": "correction", "claim": "The staging migration was cancelled, not postponed", "evidence": "Replaces: staging move", "project": "web"},
    {"kind": "preference", "claim": "Assistant should call itself Aurora", "evidence": "naming instruction", "project": "", "peer": "assistant"},
    {"kind": "environment", "claim": "Assistant runs on the same 8 GB laptop", "evidence": "stated", "project": "", "peer": "assistant"},
]

STALE_FACT = {"kind": "environment", "claim": "Staging server is moving to Hetzner VPS",
              "evidence": "old plan", "project": "web"}

# --- queries: (category, query, expected substring / tuple / None=refusal) ---
CASES = [
    ("direct", "which database does staging use", "postgres"),
    ("direct", "where does production run", "hetzner"),
    ("direct", "what framework does the frontend use", "svelte"),
    ("direct", "how much ram does the machine have", "8 gb"),
    ("direct", "what do backups use", "restic"),
    ("direct", "what does the api layer use", "fastapi"),
    ("direct", "what should the assistant call itself", "aurora"),
    ("direct", "what runs grafana", "grafana"),
    ("paraphrase", "what code editor does the user like", "vs code"),
    ("paraphrase", "how much memory does their computer have", "8 gb"),
    ("paraphrase", "does the user like light or dark interfaces", "dark mode"),
    ("paraphrase", "where does the test environment live", "fly.io"),
    ("paraphrase", "any rules about private credentials in conversation", "api keys"),
    ("paraphrase", "what tool collects their daily journal", "obsidian"),
    ("paraphrase", "who handles the background task queue", "celery"),
    ("paraphrase", "what naming should the agent use for itself", "aurora"),
    ("temporal", "what happened with staging recently", "fly.io"),
    ("temporal", "any milestones this quarter", "v2"),
    ("temporal", "recent decisions about the web stack", ("svelte", "fastapi", "v2")),
    ("temporal", "latest changes to the ops tooling", ("celery", "grafana")),
    ("temporal", "what did the user settle on last week for staging", "fly.io"),
    ("refusal", "what is the user's favorite food", None),
    ("refusal", "what car does the user drive", None),
    ("refusal", "what is the user's star sign", None),
    ("refusal", "what pets does the user own", None),
    ("refusal", "user's blood type", None),
    ("project", "what database does the project use|web", "postgres"),
    ("project", "where is production deployed|web", "hetzner"),
    ("project", "what runs the scheduled jobs|ops", "celery"),
    ("project", "backup tool|ops", "restic"),
    ("project", "dashboard port|ops", "3001"),
    ("peer", "what is the assistant named|assistant", "aurora"),
    ("peer", "where does the assistant run|assistant", "8 gb"),
    ("peer", "assistant naming|assistant", "aurora"),
    ("peer", "assistant hardware|assistant", "8 gb"),
    ("broad", "give me an overview of the user's preferences and setup", None),
    ("broad", "summarize what you know about the user", None),
    ("broad", "recap everything about the web project", None),
    ("broad", "profile of the user's tools and decisions", None),
    ("broad", "what do you know", None),
]

GATES = {"onnx": 0.90, "hash": 0.78}


def _real_models() -> Path:
    return Path(os.path.expanduser(
        "~/.hermes/plugins-data/hwiki-extractor/models/minilm"))


def _tier(embedder) -> str:
    """Actual tier, resolved AFTER an embed (ONNX loads lazily)."""
    embedder.embed("tier probe")
    return "onnx" if embedder._onnx is not None else "hash"


def seed(root: Path):
    home = root / "home"
    vault = root / "vault"
    (home / "memories").mkdir(parents=True)
    (vault / MEMORY_REL).mkdir(parents=True)
    (home / "hwiki.json").write_text(json.dumps(
        {"wiki_path": str(vault), "memory_dir": str(MEMORY_REL)}), encoding="utf-8")

    store = ProposalStore(str(home / "facts.sqlite"),
                          embedder=Embedder(mode="auto", model_dir=_real_models()))
    for fact in FACTS:
        payload = {k: v for k, v in fact.items()
                   if k in ("kind", "claim", "evidence", "peer")}
        p = store.propose(payload, user_text="seed", session_id="eval",
                          project=fact.get("project", ""))
        store.accept(p["id"])
        write_fact_page(vault, MEMORY_REL, store.get(p["id"]))

    # The stale claim must exist and then be superseded by the current one, so
    # the eval exercises real lineage rather than only "the old fact is absent".
    stale = store.propose(STALE_FACT, user_text="seed", session_id="eval", project="web")
    store.accept(stale["id"])
    stale_page = write_fact_page(vault, MEMORY_REL, store.get(stale["id"]))
    current = next(p["id"] for p in store.list_proposals(status="accepted")
                   if store.get(p["id"])["claim"].startswith("Staging server stays"))
    store.supersede(current, stale["id"])
    mark_superseded_page(stale_page, "Staging server stays on Fly.io")
    return store, vault


def run_cases(store):
    results = []
    for category, raw, expect in CASES:
        query, _, scope = raw.partition("|")
        kwargs = {}
        if scope:
            kwargs["project" if category == "project" else "peer"] = scope
        hits = store.recall(query, limit=8, **kwargs)
        if category == "refusal":
            passed = not hits
            detail = "empty" if passed else f"returned {hits[0]['claim'][:60]}"
        elif category == "broad":
            covered = len({h["claim"] for h in hits})
            passed = covered >= 4
            detail = f"covered {covered} facts"
        else:
            texts = " ".join(f"{h['claim']} {h.get('evidence', '')}"
                             for h in hits[:5]).lower()
            expects = expect if isinstance(expect, tuple) else (expect,)
            passed = any(e.lower() in texts for e in expects)
            detail = hits[0]["claim"][:70] if hits else "(empty)"
        results.append((category, query, passed, detail))
    return results


def _reembed(store, embedder):
    """Re-embed seeded rows in the tier under test so scores are consistent.

    Resolve the tier *before* writing a single vector: writing rows first and
    only then asking which tier produced them lets the ONNX session load during
    the write loop, so a run configured for hash would report and score as onnx.
    """
    tier = _tier(embedder)
    conn = sqlite3.connect(store.database)
    try:
        for pid, claim in conn.execute("SELECT id, claim FROM proposals"):
            conn.execute("UPDATE proposals SET embedding=? WHERE id=?",
                         (embedder.embed(claim), pid))
        conn.commit()
    finally:
        conn.close()
    return tier


def main() -> int:
    only = None
    if "--mode" in sys.argv:
        only = sys.argv[sys.argv.index("--mode") + 1]

    summary = {}
    skipped = {}
    all_ok = True
    for mode in [m for m in ("auto", "hash") if only in (None, m)]:
        tmp = tempfile.TemporaryDirectory()
        old_env = os.environ.get("HERMES_HOME")
        try:
            os.environ["HERMES_HOME"] = str(Path(tmp.name) / "home")
            store, _vault = seed(Path(tmp.name))
            embedder = Embedder(mode=mode, model_dir=_real_models())
            tier = _reembed(store, embedder)
            store.embedder = embedder
            if mode == "auto" and tier != "onnx":
                # Not a pass and not a failure: the tier under test never loaded.
                # Exit non-zero, because "no ONNX result" must not be reported as
                # a passing quality gate by an unwatching caller.
                skipped[mode] = ("ONNX tier unavailable (hash fallback); "
                                 "run under the bundled python 3.14")
                summary[mode] = run_cases(store)
                all_ok = False
                continue
            summary[tier] = run_cases(store)
        except Exception:
            traceback.print_exc()
            print(f"\n[{mode}] ERROR — this run does not count as a pass")
            all_ok = False
        finally:
            if old_env is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = old_env
            shutil.rmtree(tmp.name, ignore_errors=True)

    print("\n=== hwiki recall regression eval ===")
    for tier, results in summary.items():
        if tier in skipped:
            print(f"\n[{tier}] SKIPPED — {skipped[tier]}")
            continue
        total = sum(1 for _c, _q, ok, _d in results if ok)
        rate = total / len(results)
        gate = GATES[tier]
        ok = rate >= gate
        all_ok = all_ok and ok
        print(f"\n[{tier}] {total}/{len(results)} = {rate:.0%} "
              f"(gate {gate:.0%}) -> {'PASS' if ok else 'FAIL'}")
        by_cat = {}
        for cat, query, passed, detail in results:
            by_cat.setdefault(cat, []).append((query, passed, detail))
        for cat, cases in by_cat.items():
            hits = sum(1 for _q, p, _d in cases if p)
            print(f"  {cat:<10} {hits}/{len(cases)}")
            for query, passed, detail in cases:
                if not passed:
                    print(f"    MISS: {query!r} -> {detail}")

    print("\noverall:", "PASS" if all_ok else "FAIL")
    if skipped:
        print("skipped (not counted as pass):", ", ".join(sorted(skipped)))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
