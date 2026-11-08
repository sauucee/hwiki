"""Local natural-language-inference scorer (DeBERTa-v3-small MNLI, int8 ONNX).

Why this exists
---------------
Retrieval answers "is this related?". It does not answer "does this fact answer the
question?", and three deterministic attempts to bridge that gap were built and
measured against the frozen behavioural corpus:

  1. attribute-focus similarity — signal ran *backwards*
  2. head-term anchoring        — 21/21 false abstains
  3. slot-value divergence      — flagged different slots as contradictions

Inference is the principled fourth try: an MNLI cross-encoder was trained on
exactly "does premise P entail/contradict hypothesis H". See
``tests/_probe_nli.py`` for the measurement and
``benchmarks/phase4-nli-verdict.md`` for the verdict on the verdict.

Constraints honoured
--------------------
* **No network at runtime.** The model is bundled under
  ``<hermes_home>/plugins-data/hwiki-extractor/models/nli-deberta-small``
  (``model.onnx`` + ``tokenizer.json``). If it is missing, every method raises
  ``NliUnavailable`` — callers must degrade to "no verdict", never to a guess.
* **No numpy.** Importing any numpy build before onnxruntime initialises breaks
  the vendored capi on this platform (the same hazard documented in
  ``facts._cosine_batch``), so the softmax is pure Python.
* **Lazy.** Nothing loads until the first ``entailment()`` call, so a store that
  never asks for a verdict pays nothing.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MODEL_DIRNAME = "nli-deberta-small"
# cross-encoder/nli-deberta-v3-small label order (read from the bundled config):
#   0 contradiction, 1 entailment, 2 neutral
LABELS = ("contradiction", "entailment", "neutral")
CONTRADICTION, ENTAILMENT, NEUTRAL = 0, 1, 2
ENV_MODEL_DIR = "HWIKI_NLI_DIR"
MAX_LENGTH = 256


class NliUnavailable(RuntimeError):
    """The bundled model is missing or unusable. Callers must not guess."""


def nli_model_dir(home: Optional[Path] = None) -> Path:
    """Where the bundled cross-encoder lives; ``HWIKI_NLI_DIR`` overrides."""
    override = os.environ.get(ENV_MODEL_DIR)
    if override:
        return Path(override).expanduser()
    if home is None:
        try:
            from .facts import hermes_home  # local import: avoid an import cycle
        except ImportError:  # loaded as a top-level module
            from facts import hermes_home
        home = hermes_home()
    return Path(home) / "plugins-data" / "hwiki-extractor" / "models" / MODEL_DIRNAME


_SESSIONS: Dict[str, "NliScorer"] = {}


class NliScorer:
    """Entailment / contradiction probabilities for a (premise, hypothesis) pair."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else nli_model_dir()
        model_path = self.model_dir / "model.onnx"
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not model_path.is_file() or not tokenizer_path.is_file():
            raise NliUnavailable(
                f"no NLI model under {self.model_dir} "
                f"(expected model.onnx + tokenizer.json)"
            )
        try:
            vendored = self.model_dir.parent / "site-packages"
            if vendored.is_dir() and str(vendored) not in sys.path:
                sys.path.insert(0, str(vendored))  # Hermes runs -I: user site is invisible
            import onnxruntime  # type: ignore
            from tokenizers import Tokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional tier
            raise NliUnavailable(f"onnxruntime/tokenizers unavailable: {exc}") from exc
        try:
            self._session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"])
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:  # pragma: no cover - corrupt download
            raise NliUnavailable(f"could not load the NLI model: {exc}") from exc
        self._tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self._input_names = {i.name for i in self._session.get_inputs()}

    # -- scoring -----------------------------------------------------------

    def probabilities(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        """(contradiction, entailment, neutral) probabilities."""
        encoded = self._tokenizer.encode(premise or "", hypothesis or "")
        feed = {"input_ids": [encoded.ids], "attention_mask": [encoded.attention_mask]}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = [encoded.type_ids]
        logits = self._session.run(None, feed)[0][0]
        values = [float(v) for v in logits]
        peak = max(values)
        exps = [math.exp(v - peak) for v in values]
        total = sum(exps) or 1.0
        probs = [v / total for v in exps]
        return probs[CONTRADICTION], probs[ENTAILMENT], probs[NEUTRAL]

    def entailment(self, premise: str, hypothesis: str) -> float:
        return self.probabilities(premise, hypothesis)[ENTAILMENT]

    def contradiction(self, premise: str, hypothesis: str) -> float:
        return self.probabilities(premise, hypothesis)[CONTRADICTION]

    def classify(self, premise: str, hypothesis: str) -> str:
        probs = self.probabilities(premise, hypothesis)
        return LABELS[probs.index(max(probs))]

    # -- batch helpers -----------------------------------------------------

    def entailment_batch(self, premise: str,
                         hypotheses: Sequence[str]) -> List[float]:
        return [self.entailment(premise, h) for h in hypotheses]


def get_scorer(model_dir: Optional[Path] = None) -> "NliScorer":
    """Process-wide shared scorer (one InferenceSession per model dir).

    Raises ``NliUnavailable`` rather than returning a stub: a caller that cannot
    get a verdict must say so, not invent one.
    """
    directory = Path(model_dir) if model_dir else nli_model_dir()
    key = str(directory.expanduser().resolve())
    shared = _SESSIONS.get(key)
    if shared is None:
        shared = NliScorer(directory)
        _SESSIONS[key] = shared
    return shared


def try_get_scorer(model_dir: Optional[Path] = None) -> Optional["NliScorer"]:
    """Like ``get_scorer`` but returns ``None`` instead of raising, with a log line."""
    try:
        return get_scorer(model_dir)
    except NliUnavailable as exc:
        logger.info("nli verdict unavailable: %s", exc)
        return None
