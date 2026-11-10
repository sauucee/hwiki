"""One-shot NLI worker: score a batch of pairs, print, exit.

Why a subprocess
----------------
``nli-deberta-small`` is a 172 MB int8 cross-encoder. Loading it in-process costs
~315 MB on top of the ~53 MB the ONNX Runtime import already takes — measured 372 MB
steady-state. On an 8 GB host that is a third of the free memory held permanently by
a signal that fires a handful of times per proposal.

The model has no cross-call state; it scores one (premise, hypothesis) pair at a
time. So the cheapest correct design is to load it, answer the batch, and exit.
Measured: 3 pairs in ~1.0 s including cold start, peak 371 MB that is released
immediately instead of held.

Protocol
--------
stdin : one ``premise<TAB>hypothesis`` per line (premises/hypotheses must not
        contain tabs; they are claims, not free text blobs)
stdout: one line per input, ``contradiction entailment neutral`` as 6-decimal
        floats, in input order
stderr: diagnostics only

Exit codes: 0 ok, 2 model unavailable (caller must treat as "no verdict").

Run directly to smoke-test:
    printf 'a claim\\tanother claim\\n' | python -m hwiki.nli_worker
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MAX_LENGTH = 256
LABELS = ("contradiction", "entailment", "neutral")


def _model_dir() -> Path:
    override = os.environ.get("HWIKI_NLI_DIR")
    if override:
        return Path(override).expanduser()
    return (Path.home() / "plugins-data" / "hwiki-extractor" / "models"
            / "nli-deberta-small")


def main(argv: list[str] | None = None) -> int:
    model_dir = _model_dir()
    model_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"
    if not model_path.is_file() or not tokenizer_path.is_file():
        print(f"nli_worker: no model under {model_dir}", file=sys.stderr)
        return 2

    # Hermes runs interpreters with -I, so the user site is invisible. The vendored
    # ONNX runtime lives beside the models; add it before importing.
    vendored = model_dir.parent / "site-packages"
    if vendored.is_dir() and str(vendored) not in sys.path:
        sys.path.insert(0, str(vendored))

    try:
        import onnxruntime  # type: ignore
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as exc:
        print(f"nli_worker: onnxruntime/tokenizers unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:  # corrupt download, unsupported op, OOM
        print(f"nli_worker: could not load the model: {exc}", file=sys.stderr)
        return 2
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    input_names = {i.name for i in session.get_inputs()}

    import math

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        premise = parts[0]
        hypothesis = parts[1] if len(parts) > 1 else ""
        try:
            encoded = tokenizer.encode(premise, hypothesis)
            feed = {"input_ids": [encoded.ids],
                    "attention_mask": [encoded.attention_mask]}
            if "token_type_ids" in input_names:
                feed["token_type_ids"] = [encoded.type_ids]
            logits = session.run(None, feed)[0][0]
            values = [float(v) for v in logits]
            peak = max(values)
            exps = [math.exp(v - peak) for v in values]
            total = sum(exps) or 1.0
            probs = [v / total for v in exps]
        except Exception as exc:  # one bad pair must not lose the batch
            print(f"nli_worker: pair failed: {exc}", file=sys.stderr)
            probs = [float("nan")] * 3
        print(" ".join(f"{p:.6f}" for p in probs), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry
    raise SystemExit(main())
