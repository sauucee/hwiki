"""Worker mode must reach the same verdicts as the resident scorer.

``HWIKI_NLI_WORKER=1`` swaps the in-process cross-encoder for a one-shot
subprocess: same model, same numbers, ~1 s cold start instead of ~315 MB held
forever. The danger is a silent behaviour split — two code paths for one decision
drift, and the cheaper one quietly becomes the one that is wrong.

That risk is removed structurally: both paths call
``verdict.relation_from_probabilities`` to turn probabilities into a relation. These
tests pin that the function is the single decision point, that it implements the
documented rules, and that the worker's probabilities agree with the scorer's.

Run: python -m pytest tests/test_nli_worker_parity.py
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

import hwiki.verdict as verdict

MODEL_DIR = Path.home() / ".hermes/plugins-data/hwiki-extractor/models/nli-deberta-small"
needs_model = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").is_file(),
    reason="NLI model not installed on this host")


# -- the decision function is shared ------------------------------------------

def test_the_worker_path_uses_the_one_relation_function():
    """facts.py must not re-implement the relation rules inline.

    A second copy of "contradiction >= 0.90 else restatement else compatible" is
    exactly how two paths drift apart. Assert the source calls the shared function.
    """
    import hwiki.facts as facts
    source = inspect.getsource(facts)
    assert "relation_from_probabilities" in source, (
        "facts.py does not call verdict.relation_from_probabilities; the worker "
        "path has grown its own copy of the relation rules")


def test_classify_pair_delegates_to_the_shared_function():
    """The resident scorer must not decide the relation itself either."""
    source = inspect.getsource(verdict.classify_pair)
    assert "relation_from_probabilities(" in source, (
        "classify_pair decides the relation inline instead of delegating")


# -- the decision rules -------------------------------------------------------

def test_contradiction_wins_over_entailment():
    assert verdict.relation_from_probabilities(0.99, 0.99, None) == "contradiction"


def test_restatement_requires_mutual_entailment():
    """A specific claim entailing a general one is a refinement, not a duplicate."""
    assert verdict.relation_from_probabilities(0.0, 0.99, 0.99) == "restatement"
    assert verdict.relation_from_probabilities(0.0, 0.99, 0.10) == "compatible"


def test_missing_reverse_never_claims_a_restatement():
    """Without the reverse score, the safe answer is 'compatible'."""
    assert verdict.relation_from_probabilities(0.0, 0.99, None) == "compatible"


def test_nan_reverse_never_claims_a_restatement():
    nan = float("nan")
    assert verdict.relation_from_probabilities(0.0, 0.99, nan) == "compatible"


def test_low_scores_are_compatible():
    assert verdict.relation_from_probabilities(0.01, 0.01, None) == "compatible"


# -- the worker agrees with the scorer ---------------------------------------

PAIRS = [
    ("The user prefers terse updates", "The user prefers short updates"),
    ("Staging is on Fly.io", "Staging is on Fly.io in the EU region"),
    ("The user prefers dark mode", "The user prefers light mode"),
]


@needs_model
def test_worker_probabilities_agree_with_the_scorer():
    """Same model, same numbers — within rounding on the wire."""
    import hwiki.nli as nli
    reads = nli.score_pairs_via_worker(PAIRS, model_dir=MODEL_DIR)
    assert reads is not None, "worker returned no verdicts"
    scorer = nli.NliScorer(MODEL_DIR)
    for (a, b), got in zip(PAIRS, reads):
        want = scorer.probabilities(a, b)
        assert max(range(3), key=lambda i: got[i]) == \
               max(range(3), key=lambda i: want[i]), (
            f"{a!r} / {b!r}: worker {got} vs scorer {want}")
        assert all(abs(g - w) < 0.02 for g, w in zip(got, want)), (
            f"{a!r} / {b!r}: worker {got} vs scorer {want}")


@needs_model
def test_worker_and_scorer_choose_the_same_relation():
    """End to end: the shared decision function over each path's numbers."""
    import hwiki.nli as nli
    reads = nli.score_pairs_via_worker(PAIRS, model_dir=MODEL_DIR)
    assert reads is not None
    scorer = nli.NliScorer(MODEL_DIR)

    reversed_pairs = [(b, a) for a, b in PAIRS]
    reverse_reads = nli.score_pairs_via_worker(reversed_pairs, model_dir=MODEL_DIR)
    assert reverse_reads is not None

    for i, (a, b) in enumerate(PAIRS):
        # Worker: forward + reverse from the batch, decision from the shared fn.
        c, e, _ = reads[i]
        rev = reverse_reads[i][1]
        worker_relation = verdict.relation_from_probabilities(c, e, rev)
        # Resident: classify_pair does its own reverse lookup internally.
        resident_relation = verdict.classify_pair(a, b, scorer)["relation"]
        assert worker_relation == resident_relation, (
            f"{a!r} / {b!r}: worker says {worker_relation!r}, "
            f"resident says {resident_relation!r}")
