"""Footprint benchmark: what each tier costs in RSS, latency and disk.

Every mode runs in its own subprocess so ``VmRSS`` is not contaminated by a previous
tier (the ONNX runtime never unloads). Numbers are printed as JSON so they can be pasted
into a benchmark table without transcription mistakes.

    python3 tests/bench_footprint.py
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _hwiki_path  # noqa: F401,E402

MODES = ("hash", "onnx", "onnx+nli")
CLAIMS = [
    "Staging server stays on Fly.io", "Production runs on a Hetzner VPS",
    "The database is Postgres 15", "Backups run nightly with restic to B2",
    "The user prefers terse status updates", "Incidents are graded from SEV1 to SEV4",
    "The on-call rotation changes every Monday", "The user takes notes in Obsidian daily",
]


def _vmrss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return 0.0


def child(mode: str) -> None:
    """One tier, measured in isolation."""
    import tempfile

    from hwiki.facts import (Embedder, ProposalStore, question_buffer,
                             write_fact_page)
    from hwiki.maintain import run_maintenance

    if mode in ("onnx", "onnx+nli"):
        os.environ["HWIKI_BACKEND"] = "onnx"
        os.environ.setdefault("HWIKI_EMBEDDINGS", "onnx")
    if mode == "onnx+nli":
        os.environ["HWIKI_NLI_CONFLICT"] = "1"

    base = _vmrss_mb()
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name) / "vault"
    memory = root / "Hermes/Memory/Auto"
    memory.mkdir(parents=True)
    emb_mode = "hash" if mode == "hash" else "auto"
    store = ProposalStore(Path(tmp.name) / "f.sqlite", embedder=Embedder(mode=emb_mode))
    store.embedder.embed("warm")
    for claim in CLAIMS:
        proposal = store.propose({"kind": "fact", "claim": claim, "evidence": "said"},
                                 user_text="said", session_id="s")
        store.accept(proposal["id"])
        page = write_fact_page(root, Path("Hermes/Memory/Auto"), store.get(proposal["id"]))
        store.set_page_path(proposal["id"], page)
    if mode == "onnx+nli":
        # the NLI bundle is lazy: force one classification so the reported RSS is the
        # cost of actually using the tier, not the cost of importing it
        from hwiki import verdict
        verdict.classify_pair("Deploys are allowed on Fridays",
                              "Deploys are never allowed on Fridays")
    loaded = _vmrss_mb()

    # Cold and warm are measured separately and reported separately: the query cache
    # makes repeats ~9x cheaper, and a single blended number would hide which is which.
    queries = ["where does staging run", "which database do we use",
               "how often do backups run", "who is on call"]
    cold = []
    for _ in range(5):
        for query in queries:
            started = time.perf_counter()
            store.recall(f"{query} {time.perf_counter_ns()}", limit=5)  # never a hit
            cold.append((time.perf_counter() - started) * 1000.0)
    warm = []
    for _ in range(5):
        for query in queries:
            started = time.perf_counter()
            store.recall(query, limit=5)
            warm.append((time.perf_counter() - started) * 1000.0)
    cold.sort()
    warm.sort()
    latencies = cold

    started = time.perf_counter()
    report = run_maintenance(store, root, Path("Hermes/Memory/Auto"),
                             embedder=store.embedder, write=False,
                             question_buffer=question_buffer())
    tick_ms = (time.perf_counter() - started) * 1000.0

    pages = list(memory.glob("*.md"))
    print(json.dumps({
        "mode": mode,
        "rss_base_mb": base,
        "rss_loaded_mb": loaded,
        "rss_delta_mb": round(loaded - base, 1),
        "recall_cold_p50_ms": round(statistics.median(cold), 2),
        "recall_cold_p95_ms": round(cold[int(len(cold) * 0.95) - 1], 2),
        "recall_warm_p50_ms": round(statistics.median(warm), 2),
        "recall_warm_p95_ms": round(warm[int(len(warm) * 0.95) - 1], 2),
        "embed_cache": store.embedder.cache_info(),
        "maintain_ms": round(tick_ms, 1),
        "maintain_work_items": report.work_items,
        "page_bytes": round(statistics.mean(p.stat().st_size for p in pages), 0) if pages else 0,
        "db_bytes_per_fact": round((Path(tmp.name) / "f.sqlite").stat().st_size / len(CLAIMS), 0),
        "model_disk_mb": _model_disk_mb(mode),
    }))
    tmp.cleanup()


def _model_disk_mb(mode: str) -> float:
    if mode == "hash":
        return 0.0
    data = Path.home() / ".hermes/plugins-data/hwiki-extractor/models"
    total = 0
    targets = ["minilm"] if mode == "onnx" else ["minilm", "nli-deberta-small"]
    for name in targets:
        directory = data / name
        if directory.is_dir():
            total += sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    return round(total / 1e6, 1)


def main() -> int:
    rows = []
    for mode in MODES:
        out = subprocess.run([sys.executable, __file__, "--child", mode],
                             capture_output=True, text=True, timeout=600)
        if out.returncode != 0:
            rows.append({"mode": mode, "error": out.stderr.strip().splitlines()[-1:]})
            continue
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        child(sys.argv[sys.argv.index("--child") + 1])
    else:
        raise SystemExit(main())
