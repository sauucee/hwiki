"""Experiment: can an attribute-focus signal separate 'related' from 'answering'?

For the top hit of each case, compare:
  entity_sim = the score the retriever already produces (whole query vs claim)
  focus_sim  = similarity between the query MINUS terms shared with the claim
               (i.e. the asked attribute) and the claim

Hypothesis: for a fact that is merely about the same entity, entity_sim is high
but focus_sim is low. For a fact that actually answers, both are comparable.

Read-only: temp stores only.
"""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _hwiki_path  # noqa
import behavior_cases as bc
import run_behavior as rb
from hwiki.facts import Embedder, _tokens, _cosine_batch

real = rb._real_models()


def focus_of(query, claim, embedder):
    qt = set(_tokens(query))
    ct = set(_tokens(claim))
    shared = qt & ct
    focus = sorted(qt - shared)
    if not focus:
        return None
    phrase = " ".join(focus)
    return _cosine_batch(embedder.embed(phrase), [embedder.embed(claim)])[0]


def main():
    emb = Embedder(mode="auto", model_dir=real)
    rb._tier(emb)
    print(f"tier={ 'onnx' if emb._onnx is not None else 'hash'}\n")
    print(f"{'set':<8}{'case':<26}{'expect':<10}{'score':>7}{'focus':>8}  top claim")
    print("-" * 110)
    for story in bc.ALL_STORIES:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name) / story.id
        root.mkdir(parents=True)
        old = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(Path(tmp.name) / "home")
        try:
            store, _ids, _p = rb.seed_story(story, root, emb)
            store.embedder = emb
            rb._reembed(store, emb)
            for case in story.cases:
                kw = {}
                if case.project:
                    kw["project"] = case.project
                hits = store.recall(case.query, limit=8, **kw)
                if not hits:
                    print(f"{story.set:<8}{case.id:<26}{'refuse':<10}{'-':>7}{'-':>8}  (no hits)")
                    continue
                top = hits[0]
                fs = focus_of(case.query, top["claim"], emb)
                kind = "refuse" if case.expect.refuse else "answer"
                fss = f"{fs:.3f}" if fs is not None else "n/a"
                print(f"{story.set:<8}{case.id:<26}{kind:<10}"
                      f"{top['score']:>7.3f}{fss:>8}  {top['claim'][:52]}")
        finally:
            if old is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = old
            tmp.cleanup()


if __name__ == "__main__":
    main()
