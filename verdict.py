"""Fact-pair relations from the local NLI model, gated by a shared frame.

The shipped conflict flag is negation-only. Measured on a labelled 21-pair set
(``tests/_probe_nli.py``), that catches 1 of 7 genuine contradictions.

The NLI model alone catches 7 of 7, but also calls three *compatible* pairs
contradictions — it reads "runs on Fly.io" against "runs Postgres" as incompatible
because both values fill the same syntactic slot, and it called a scoped pair
("aisle seats on long flights" / "window seats on short flights") contradictory.

A frame-overlap gate was tried and **rejected**: requiring a shared content-word
frame (>= 0.50) removed two false positives but also dropped three genuine
contradictions, ending at 4/7 recall — worse than the model alone. Real value swaps
*lower* word overlap, so overlap is the wrong gate for this decision.

So the model's own label is reported as-is, with the probabilities, and the measured
error rate travels with it:

    recall 7/7 genuine contradictions, 3/14 compatible pairs wrongly flagged

That is why the result is an advisory annotation for a reviewer and never the
`possible_conflict` flag: negation keeps that flag at high precision, and the model
adds recall under an honest label.

Everything here is opt-in (``HWIKI_NLI_CONFLICT=1``) so the default retrieval path
stays byte-for-byte what it was. When the model is absent the call raises
``NliUnavailable`` and the caller keeps the negation-only behaviour — it never
falls back to a guess.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple

try:  # normal package import
    from .nli import NliScorer, NliUnavailable, nli_model_dir, try_get_scorer
except ImportError:  # loaded as a top-level module
    from nli import NliScorer, NliUnavailable, nli_model_dir, try_get_scorer  # type: ignore

logger = logging.getLogger(__name__)

ENV_ENABLE = "HWIKI_NLI_CONFLICT"
MIN_CONTRADICTION = 0.90
MIN_ENTAILMENT = 0.90

_STOP = frozenset("""
a an the and or but if then than so is are was were be been being am
of in on at to for from by with without into onto over under about
it its this that these those there here we you they i he she
do does did done have has had can could will would should may might must
not no never always every all any some more most less least very
user users they them their our your my
""".split())


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "").strip().lower() in {"1", "true", "yes", "on"}


def content_tokens(text: str) -> set:
    tokens = set()
    for raw in str(text or "").lower().replace("'s", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch in ".-_")
        if len(token) > 1 and token not in _STOP:
            tokens.add(token)
    return tokens


def frame_overlap(a: str, b: str) -> float:
    """Jaccard over content words — how much of a shared sentence frame the two share."""
    left, right = content_tokens(a), content_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def classify_pair(a: str, b: str, scorer: Optional[NliScorer] = None) -> Dict[str, Any]:
    """Relation between two claims. Raises ``NliUnavailable`` when no model exists."""
    active = scorer if scorer is not None else try_get_scorer()
    if active is None:
        raise NliUnavailable("no NLI model available")
    contradiction, entailment, neutral = active.probabilities(a, b)
    overlap = frame_overlap(a, b)
    if contradiction >= MIN_CONTRADICTION:
        relation = "contradiction"
    elif entailment >= MIN_ENTAILMENT:
        relation = "restatement"
    else:
        relation = "compatible"
    return {
        "relation": relation,
        "contradiction": round(contradiction, 3),
        "entailment": round(entailment, 3),
        "neutral": round(neutral, 3),
        "frame_overlap": round(overlap, 3),
    }


def flags_contradiction(a: str, b: str, scorer: Optional[NliScorer] = None) -> bool:
    try:
        return classify_pair(a, b, scorer)["relation"] == "contradiction"
    except NliUnavailable:
        return False


def available() -> bool:
    """True when a model can be loaded (cheap directory check first)."""
    try:
        return nli_model_dir().is_dir()
    except Exception:  # pragma: no cover - defensive
        return False
