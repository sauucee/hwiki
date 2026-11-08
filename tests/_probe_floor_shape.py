"""Probe 3: is the 0.26 absolute noise floor the right shape?

`runbooks` ("where are the operational procedures kept" -> "Runbooks live in the
ops vault folder") scores 0.235 and is missed by 0.025. Lowering the constant is
forbidden, but the *shape* of the rule may be wrong: an absolute floor on raw
cosine ignores how the whole corpus scores for that query.

Measures, per case: the top score, and the distribution of every score the same
query produces across the corpus. Reports whether a distribution-relative rule
would accept the right things and still refuse the unknowns.
"""
import os, statistics, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _hwiki_path  # noqa
import behavior_cases as bc
import run_behavior as rb
from hwiki.facts import Embedder, _cosine_batch, _tokens


def all_scores(store, query, kw):
    """Score every accepted fact for this query (no floor, no cutoff)."""
    from hwiki.facts import _db_connection
    qv = store.embedder.embed(query)
    qt = set(_tokens(query))
    sql = ("SELECT claim, embedding FROM proposals WHERE status='accepted' "
           "AND (superseded_by IS NULL OR superseded_by='')")
    if kw.get("project"):
        sql += " AND project=?"
    with _db_connection(store.database) as conn:
        rows = conn.execute(sql, ((kw["project"],) if kw.get("project") else ())).fetchall()
    cos = _cosine_batch(qv, [r["embedding"] or b"" for r in rows])
    out = []
    for row, c in zip(rows, cos):
        toks = set(_tokens(row["claim"]))
        ov = len(qt & toks) / max(1, len(qt | toks))
        out.append((max(c, 0.5 * c + 0.5 * ov), row["claim"]))
    return sorted(out, reverse=True)


def main():
    emb = Embedder(mode="auto", model_dir=rb._real_models())
    rb._tier(emb)
    print(f"tier={rb._tier(emb)}\n")
    print(f"{'set':<8}{'case':<26}{'expect':<9}{'top':>6}{'rank':>5}"
          f"{'med':>7}{'p90':>7}{'z':>7}")
    rows = []
    for story in bc.ALL_STORIES:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name) / story.id
        root.mkdir(parents=True)
        old = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(Path(tmp.name) / "home")
        try:
            store, _i, _p = rb.seed_story(story, root, emb)
            store.embedder = emb
            rb._reembed(store, emb)
            for case in story.cases:
                kw = {}
                if case.project:
                    kw["project"] = case.project
                scores = all_scores(store, case.query, kw)
                if not scores:
                    continue
                vals = [s for s, _ in scores]
                top, claim = scores[0]
                med = statistics.median(vals)
                p90 = sorted(vals)[int(len(vals) * 0.9) - 1] if len(vals) > 1 else vals[0]
                z = (top - statistics.mean(vals)) / (statistics.pstdev(vals) or 1e-9)
                expect = "refuse" if case.expect.refuse else "answer"
                hits = any(need.lower() in claim.lower()
                           for need in (case.expect.needs_any or []))
                mark = "" if (expect == "refuse") != hits else "  <-- MISS"
                print(f"{story.set:<8}{case.id:<26}{expect:<9}{top:>6.3f}"
                      f"{1:>5}{med:>7.3f}{p90:>7.3f}{z:>7.2f}{mark}")
                rows.append((expect, top, med, p90, z, hits, case.id))
        finally:
            if old is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = old
            tmp.cleanup()

    print("\nWould `top >= med + 1.5(MAD)` decide correctly?")
    good = bad = 0
    for expect, top, med, p90, z, hits, cid in rows:
        vals = [r[1] for r in rows]
        accept = top >= med + 0.35 * (p90 - med if p90 > med else 0.05)
        if (expect == "answer") == accept:
            good += 1
        else:
            bad += 1
            print(f"  wrong: {cid} expect={expect} accept={accept} top={top:.3f} med={med:.3f}")
    print(f"  correct: {good}   wrong: {bad}")


if __name__ == "__main__":
    main()
