"""Latency benchmark: p50 / p95 for the retrieval path.

The project has never measured this. An old "~0.02 s" figure was quoted for code
that no longer exists, so it must not be repeated. This measures what the current
code actually does, per tier, on a named corpus.

Measures `ProposalStore.recall` only — the fact-retrieval hot path that a turn
hits. Vault FTS5 search is a separate surface and is not included.

    python3 tests/bench_latency.py            # both tiers
    python3 tests/bench_latency.py --mode hash
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _hwiki_path  # noqa: F401,E402
import behavior_cases as bc  # noqa: E402
import run_behavior as rb  # noqa: E402
from hwiki.facts import Embedder  # noqa: E402

WARMUP = 5
CORPUS_NAME = "behaviour suite corpus (5 stories)"


def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = max(0, min(len(ordered) - 1, int(round((pct / 100) * len(ordered) + 0.5)) - 1))
    return ordered[k]


def measure(mode: str) -> dict:
    model_dir = rb._real_models() if mode == "auto" else None
    emb = Embedder(mode=mode, model_dir=model_dir)
    # ONNX loads lazily on the first embed; force it before timing so the first
    # sample is not a model load.
    emb.embed("warmup")
    tier = rb._tier(emb)

    queries, latencies, first_embed = [], [], []
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
                kwargs = {}
                if case.project:
                    kwargs["project"] = case.project
                if case.peer:
                    kwargs["peer"] = case.peer
                queries.append(case.query)
                # One untimed warm pass, then timed runs.
                for _ in range(WARMUP):
                    store.recall(case.query, limit=8, **kwargs)
                samples = []
                for _ in range(20):
                    start = time.perf_counter()
                    store.recall(case.query, limit=8, **kwargs)
                    samples.append((time.perf_counter() - start) * 1000.0)
                latencies.append(statistics.median(samples))
            # separate: cost of a cold embed for a new string
            start = time.perf_counter()
            emb.embed("a cold string the model has never seen before")
            first_embed.append((time.perf_counter() - start) * 1000.0)
        finally:
            if old is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = old
            tmp.cleanup()
    return {
        "tier": tier,
        "n_queries": len(latencies),
        "p50_ms": round(percentile(latencies, 50), 3),
        "p95_ms": round(percentile(latencies, 95), 3),
        "max_ms": round(max(latencies), 3),
        "embed_p50_ms": round(percentile(first_embed, 50), 3),
    }


def main() -> int:
    modes = ["auto", "hash"] if "--mode" not in sys.argv else [sys.argv[sys.argv.index("--mode") + 1]]
    print(f"corpus: {CORPUS_NAME}")
    print(f"{'tier':<12}{'n':>4}{'p50 (ms)':>11}{'p95 (ms)':>11}{'max (ms)':>11}{'embed p50':>12}")
    for mode in modes:
        result = measure(mode)
        print(f"{result['tier']:<12}{result['n_queries']:>4}"
              f"{result['p50_ms']:>11.3f}{result['p95_ms']:>11.3f}"
              f"{result['max_ms']:>11.3f}{result['embed_p50_ms']:>12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
