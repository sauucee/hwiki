"""Probe 2: would a deterministic 'head-term anchoring' gate work?

Rule under test: after removing terms shared with the claim (the entity), the
remaining *focus* terms must appear in the claim, with naive stemming.

Prints true-abstain (good) vs false-abstain (breaks a correct answer).
"""
import os, re, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _hwiki_path  # noqa
import behavior_cases as bc
import run_behavior as rb
from hwiki.facts import Embedder, _tokens

STOP = {"what", "which", "who", "when", "where", "how", "does", "do", "is", "are",
        "the", "a", "an", "of", "for", "to", "in", "on", "their", "they", "user",
        "users", "should", "be", "any", "about", "much", "many", "and", "or",
        "there", "this", "that", "with", "at", "by", "it", "its"}


def stem(t):
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def anchored(query, claim):
    qt = {stem(t) for t in _tokens(query)} - STOP
    ct = {stem(t) for t in _tokens(claim)}
    focus = qt - ct
    if not focus:
        return True, []
    return bool(focus & ct), sorted(focus)


def main():
    emb = Embedder(mode="auto", model_dir=rb._real_models())
    rb._tier(emb)
    good = bad = miss = 0
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
                hits = store.recall(case.query, limit=8, **kw)
                if not hits:
                    if case.expect.refuse:
                        good += 1
                    continue
                ok, focus = anchored(case.query, hits[0]["claim"])
                if case.expect.refuse:
                    if ok:
                        miss += 1   # gate would let a non-answer through
                    else:
                        good += 1   # correctly abstains
                else:
                    if ok:
                        pass        # gate keeps a correct answer
                    else:
                        bad += 1    # FALSE ABSTAIN: breaks a correct answer
                        rows.append((story.set, case.id, hits[0]["score"],
                                     focus[:6], hits[0]["claim"][:46]))
        finally:
            if old is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = old
            tmp.cleanup()

    print(f"\ncorrectly abstained on unknowns : {good}")
    print(f"would let a non-answer through  : {miss}")
    print(f"FALSE ABSTAINS (breaks answers) : {bad}")
    if rows:
        print("\nfalse abstains:")
        for s, c, sc, f, cl in rows:
            print(f"  {s:<8}{c:<26} score={sc:.3f} focus={f} :: {cl}")


if __name__ == "__main__":
    main()
