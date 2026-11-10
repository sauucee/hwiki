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
import subprocess
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


WORKER_TIMEOUT_S = 120
ENV_WORKER = "HWIKI_NLI_WORKER"


def worker_enabled() -> bool:
    """True when verdicts should be scored in a short-lived worker.

    Default OFF so existing behaviour is unchanged; set ``HWIKI_NLI_WORKER=1`` to
    pay ~1 s of cold start per batch instead of holding ~315 MB permanently. On a
    memory-constrained host that is the better trade by a wide margin.
    """
    return os.environ.get(ENV_WORKER, "").strip().lower() in ("1", "true", "yes", "on")


def score_pairs_via_worker(
    pairs: Sequence[Tuple[str, str]],
    *,
    model_dir: Optional[Path] = None,
    timeout_s: int = WORKER_TIMEOUT_S,
) -> Optional[List[Tuple[float, float, float]]]:
    """Score ``pairs`` in a one-shot subprocess; ``None`` when it cannot run.

    Returns one ``(contradiction, entailment, neutral)`` tuple per input, in order.
    ``None`` means "no verdict available" — callers must degrade to no annotation,
    never to a guess. Any pair the worker could not score comes back as NaNs, so a
    caller can drop it without shifting the rest.
    """
    if not pairs:
        return []
    directory = Path(model_dir) if model_dir else nli_model_dir()
    if not (directory / "model.onnx").is_file():
        logger.info("nli worker unavailable: no model under %s", directory)
        return None
    worker = Path(__file__).with_name("nli_worker.py")
    if not worker.is_file():  # pragma: no cover - packaging error
        logger.info("nli worker script missing at %s", worker)
        return None

    payload = "".join(
        f"{(a or '').replace(chr(9), ' ')}\t{(b or '').replace(chr(9), ' ')}\n"
        for a, b in pairs)
    env = dict(os.environ)
    env[ENV_MODEL_DIR] = str(directory)
    try:
        proc = subprocess.run(
            [sys.executable, str(worker)],
            input=payload, capture_output=True, text=True,
            timeout=timeout_s, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info("nli worker failed to run: %s", exc)
        return None
    if proc.returncode != 0:
        logger.info("nli worker exited %s: %s", proc.returncode,
                    (proc.stderr or "").strip()[-300:])
        return None

    out: List[Tuple[float, float, float]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            values = tuple(float(x) for x in line.split())
        except ValueError:
            values = (float("nan"),) * 3
        if len(values) != 3:
            values = (float("nan"),) * 3
        out.append(values)  # type: ignore[arg-type]
    if len(out) != len(pairs):
        logger.info("nli worker returned %d rows for %d pairs", len(out), len(pairs))
        return None
    return out
