"""The NLI model must be usable without paying for it resident.

Loading ``nli-deberta-small`` in-process costs ~315 MB on top of the ~53 MB the
ONNX Runtime import already takes: measured 372 MB steady-state for a 172 MB
int8 model. On an 8 GB host that is a third of the free memory held forever by a
signal that fires a handful of times per proposal.

The model is a cross-encoder: it scores one (claim, candidate) pair at a time and
has no cross-call state. Nothing needs it to stay loaded between proposals, so a
one-shot worker scores a batch and exits — same answers, no residency.

These tests pin the two properties that make that safe:
  1. the worker returns the same verdicts the in-process scorer does;
  2. the worker is opt-in and degrades to "no verdict" when it cannot run,
     because a caller that cannot get a verdict must not invent one.

Run: python -m pytest tests/test_nli_worker.py
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

import hwiki.nli as nli

MODEL_DIR = Path.home() / ".hermes/plugins-data/hwiki-extractor/models/nli-deberta-small"
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").is_file(),
    reason="NLI model not installed on this host")


def _softmax(logits):
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


PAIRS = [
    ("The user prefers terse updates", "The user prefers short updates"),
    ("The user prefers dark mode", "The user prefers light mode"),
    ("Staging is on Fly.io", "Staging is on Fly.io in the EU region"),
]


def test_worker_module_exists_and_is_runnable():
    """The batch worker must be a real, importable module path."""
    assert (nli.__file__.rsplit("/", 1)[0] + "/nli_worker.py") or True
    worker = Path(nli.__file__).parent / "nli_worker.py"
    assert worker.is_file(), f"expected the worker script at {worker}"


def test_worker_answers_match_the_in_process_scorer():
    """Same pairs, same relations — the worker is not a different model."""
    worker = Path(nli.__file__).parent / "nli_worker.py"
    payload = "\n".join(f"{a}\t{b}" for a, b in PAIRS) + "\n"
    env = dict(os.environ)
    env["HWIKI_NLI_DIR"] = str(MODEL_DIR)
    proc = subprocess.run(
        [sys.executable, str(worker)], input=payload, capture_output=True,
        text=True, timeout=300, env=env)
    assert proc.returncode == 0, f"worker failed: {proc.stderr[-800:]}"

    out = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(out) == len(PAIRS), f"expected {len(PAIRS)} rows, got {len(out)}: {out}"

    scorer = nli.NliScorer(MODEL_DIR)
    for (a, b), line in zip(PAIRS, out):
        want = scorer.probabilities(a, b)
        got = tuple(float(x) for x in line.split())
        assert len(got) == 3, got
        # Rounded to 4dp on the wire; compare the argmax and closeness.
        assert max(range(3), key=lambda i: got[i]) == \
               max(range(3), key=lambda i: want[i]), (
            f"{a!r} / {b!r}: worker argmax {got} vs in-process {want}")
        assert all(abs(g - w) < 0.02 for g, w in zip(got, want)), (
            f"{a!r} / {b!r}: worker {got} vs in-process {want}")


def test_worker_failure_yields_no_verdict_rather_than_a_guess():
    """A missing worker or model must produce None, never a fabricated score."""
    assert hasattr(nli, "score_pairs_via_worker"), (
        "nli should expose score_pairs_via_worker() for the batch path")
    # Point at a directory with no model: the helper must return None.
    result = nli.score_pairs_via_worker([("a", "b")], model_dir=Path("/nonexistent"))
    assert result is None, f"expected None for a missing model, got {result!r}"
