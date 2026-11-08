"""Run the offline behavioural suite.

    <bundled python> tests/run_behavior.py                # dev + holdout, both tiers
    <bundled python> tests/run_behavior.py --set dev
    <bundled python> tests/run_behavior.py --mode hash --out artifacts/behavior.json

Exit 0 iff every run set meets its preregistered gate **and** every provenance
invariant holds and the graders themselves pass their mutation checks.

A tier that fails to load is reported as SKIPPED and fails the run — a silent
hash fallback must never be scored as an ONNX result.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _hwiki_path  # noqa: F401,E402

from behavior_cases import QUALITY_TARGET, REGRESSION_FLOOR, stories_for  # noqa: E402
from behavior_graders import grade, report_mutations  # noqa: E402
from hwiki.facts import (  # noqa: E402
    Embedder,
    ProposalRejected,
    ProposalStore,
)
import sqlite3  # noqa: E402


def _reembed(store, embedder) -> str:
    """Re-embed every stored row in the tier under test.

    Rows are written by ``seed_story`` using whatever embedder the store was
    built with (which resolves under the temp HERMES_HOME and therefore falls
    back to hash). Swapping ``store.embedder`` without re-embedding leaves rows
    in a different vector space, so every cosine becomes meaningless and every
    query returns no hits - a harness bug that looks exactly like a product bug.
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


def _real_models() -> Path:
    return Path(os.path.expanduser(
        "~/.hermes/plugins-data/hwiki-extractor/models/minilm"))


def _tier(embedder) -> str:
    embedder.embed("tier probe")  # ONNX loads lazily on first embed
    return "onnx" if embedder._onnx is not None else "hash"


def seed_story(story, root: Path, embedder):
    """Build a throwaway store for one story; return (store, ids, problems).

    The embedder is injected rather than constructed here: a store built with a
    default ``Embedder()`` resolves its model under the temp ``HERMES_HOME``,
    fails to find ONNX, and floods the run with a fallback traceback for every
    story before the tier is ever selected.
    """
    store = ProposalStore(str(root / "facts.sqlite"), embedder=embedder)
    problems = []
    ids = []

    for spec in story.facts:
        payload = {"kind": spec.kind, "claim": spec.claim, "evidence": spec.evidence}
        p = store.propose(payload, user_text="seed", session_id="story",
                          project=spec.project, peer=spec.peer)
        if not store.accept(p["id"]):
            problems.append(f"could not accept {spec.claim!r}")
        ids.append(p["id"])

    for new_i, old_i in story.supersedes:
        ok, old_row = store.supersede(ids[new_i], ids[old_i])
        if not ok:
            problems.append(f"supersede {new_i}->{old_i} failed")
            continue
        if old_row["status"] != "superseded":
            problems.append(f"old fact not marked superseded: {old_row['status']}")
        new_row = store.get(ids[new_i])
        if new_row.get("supersedes") != ids[old_i]:
            problems.append("lineage not recorded on the replacing fact")

    # A superseded fact must never come back as current.
    for _new_i, old_i in story.supersedes:
        old_claim = store.get(ids[old_i])["claim"]
        if any(h["id"] == ids[old_i] for h in store.recall(old_claim, limit=10)):
            problems.append(f"superseded fact still recalled: {old_claim!r}")

    for claim in story.expect_reject:
        try:
            store.propose({"kind": "blocker", "claim": claim, "evidence": "seed"},
                          user_text="seed", session_id="story")
        except ProposalRejected:
            pass
        except Exception as exc:  # noqa: BLE001 - any other error is a real defect
            problems.append(f"secret rejection raised {type(exc).__name__}: {exc}")
        else:
            problems.append(f"secret-shaped claim was accepted: {claim[:50]!r}")

    for a_i, b_i in story.unresolved_conflicts:
        a, b = store.get(ids[a_i]), store.get(ids[b_i])
        if a["status"] != "accepted" or b["status"] != "accepted":
            problems.append("unresolved conflict had a side silently resolved")
        if b.get("supersedes") == a["id"] or a.get("supersedes") == b["id"]:
            problems.append("contradiction was silently superseded")

    return store, ids, problems


def run_story(story, embedder, root: Path):
    store, _ids, problems = seed_story(story, root, embedder)
    store.embedder = embedder
    _reembed(store, embedder)

    rows = []
    for case in story.cases:
        kwargs = {}
        if case.project:
            kwargs["project"] = case.project
        if case.peer:
            kwargs["peer"] = case.peer
        try:
            hits = store.recall(case.query, limit=8, **kwargs)
        except Exception as exc:  # noqa: BLE001
            rows.append({"story": story.id, "set": story.set, "case": case.id,
                         "category": case.category, "passed": False,
                         "reason": f"recall raised {type(exc).__name__}: {exc}",
                         "hits": []})
            continue
        result = grade(hits, case.expect)
        rows.append({
            "story": story.id, "set": story.set, "case": case.id,
            "category": case.category, "query": case.query,
            "passed": result.passed, "reason": result.reason,
            "hits": [{"claim": h["claim"][:110], "replaced": h.get("replaced", ""),
                      "score": h.get("score")} for h in hits[:5]],
        })
    return rows, problems


def main() -> int:
    only_set = None
    if "--set" in sys.argv:
        only_set = sys.argv[sys.argv.index("--set") + 1]
    only_mode = None
    if "--mode" in sys.argv:
        only_mode = sys.argv[sys.argv.index("--mode") + 1]
    out_path = None
    if "--out" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("--out") + 1])

    print("grader mutation checks")
    if not report_mutations():
        print("\ngraders are unsound — refusing to score anything")
        return 1

    all_rows = []
    all_problems = []
    summary = {}
    skipped = {}
    ok = True

    for mode in [m for m in ("auto", "hash") if only_mode in (None, m)]:
        for set_name in [s for s in ("dev", "holdout") if only_set in (None, s)]:
            tmp = tempfile.TemporaryDirectory()
            old_env = os.environ.get("HERMES_HOME")
            label = None
            try:
                os.environ["HERMES_HOME"] = str(Path(tmp.name) / "home")
                embedder = Embedder(mode=mode, model_dir=_real_models())
                tier = _tier(embedder)
                if mode == "auto" and tier != "onnx":
                    skipped["auto"] = ("ONNX tier unavailable (hash fallback); "
                                       "run under the bundled python 3.14")
                    ok = False
                    continue
                label = tier
                tier_rows = []
                for story in stories_for(set_name):
                    root = Path(tmp.name) / story.id
                    root.mkdir(parents=True, exist_ok=True)
                    rows, problems = run_story(story, embedder, root)
                    tier_rows.extend(rows)
                    all_problems.extend(f"[{story.id}] {p}" for p in problems)
                all_rows.extend(tier_rows)
            except Exception:
                traceback.print_exc()
                print(f"\n[{mode}/{set_name}] ERROR — not counted as a pass")
                ok = False
            finally:
                if old_env is None:
                    os.environ.pop("HERMES_HOME", None)
                else:
                    os.environ["HERMES_HOME"] = old_env
                shutil.rmtree(tmp.name, ignore_errors=True)

            if label is None:
                continue
            rows = tier_rows
            if not rows:
                continue
            total = sum(1 for r in rows if r["passed"])
            rate = total / len(rows)
            floor = REGRESSION_FLOOR[label][set_name]
            target = QUALITY_TARGET[label][set_name]
            passed = rate >= floor
            ok = ok and passed
            summary[(label, set_name)] = (total, len(rows), rate, floor, target,
                                          passed, rows)

    print("\n=== hwiki behavioural suite ===")
    for (tier, set_name), (total, n, rate, floor, target, passed, rows) in summary.items():
        verdict = "PASS" if passed else "FAIL"
        meets = "meets target" if rate >= target else f"below target {target:.0%}"
        print(f"\n[{tier}/{set_name}] {total}/{n} = {rate:.0%} "
              f"(floor {floor:.0%}, {meets}) -> {verdict}")
        by_cat = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r)
        for cat in sorted(by_cat):
            cases = by_cat[cat]
            hits = sum(1 for c in cases if c["passed"])
            print(f"  {cat:<12} {hits}/{len(cases)}")
            for c in cases:
                if not c["passed"]:
                    print(f"    MISS {c['case']}: {c['reason']}")

    if skipped:
        print(f"\nSKIPPED (not counted as pass): {', '.join(sorted(skipped))}")
        for why in skipped.values():
            print(f"  - {why}")

    print(f"\nprovenance/safety invariants: "
          f"{'PASS' if not all_problems else 'FAIL'}")
    for problem in all_problems:
        print(f"  ! {problem}")
    ok = ok and not all_problems

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "tiers": {f"{t}/{s}": {"passed": v[0], "total": v[1],
                                   "rate": v[2], "floor": v[3], "target": v[4],
                                   "ok": v[5]}
                      for (t, s), v in summary.items()},            "skipped": skipped,
            "invariant_problems": all_problems,
            "cases": all_rows,
        }, indent=2), encoding="utf-8")
        print(f"\nartifacts -> {out_path}")

    print("\noverall:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
