"""Pytest wrapper for the Phase 1 behavioural suite.

The graders are mutation-tested here because a grader that cannot fail is worse
than no suite at all. The behavioural runs go through the CLI so the same code
path produces the JSON artifacts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import behavior_cases as bc

ANSWERABILITY_ENV = bc.ANSWERABILITY_ENV

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent


def _run(args, timeout=600, answerability=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    if answerability is not None:
        env[ANSWERABILITY_ENV] = "1" if answerability else "0"
    return subprocess.run(
        [sys.executable, *args], cwd=TESTS, capture_output=True, text=True,
        timeout=timeout, env=env,
    )


def _onnx_available() -> bool:
    try:
        result = _run(["-c", (
            "import sys; sys.path.insert(0, 'tests')\n"
            "import _hwiki_path\n"
            "from hwiki.facts import Embedder\n"
            "from pathlib import Path\n"
            "import os\n"
            "e = Embedder(mode='auto', model_dir=Path(os.path.expanduser(\n"
            "    '~/.hermes/plugins-data/hwiki-extractor/models/minilm')))\n"
            "e.embed('probe')\n"
            "print('onnx' if e._onnx is not None else 'hash')\n"
        )], timeout=120)
    except Exception:
        return False
    return "onnx" in result.stdout


def test_graders_reject_bad_answers():
    """A grader that passes a wrong answer is a false green; prove each fails."""
    result = _run(["behavior_graders.py"], timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert " 0 failed" not in result.stdout
    assert "FAIL" not in result.stdout, result.stdout


def test_behavior_hash_baseline():
    """Hash tier is available everywhere, so this is the portable guard."""
    result = _run(["run_behavior.py", "--mode", "hash"], timeout=600)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "provenance/safety invariants: PASS" in result.stdout


@pytest.mark.skipif(not _onnx_available(), reason="ONNX tier unavailable")
def test_behavior_onnx_baseline():
    result = _run(["run_behavior.py", "--mode", "auto"], timeout=600)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "SKIPPED" not in result.stdout


@pytest.mark.skipif(not _onnx_available(), reason="ONNX tier unavailable")
def test_quality_target_met_with_answerability_enabled():
    """The 85% holdout target must be an assertion, not a hope.

    The answer-type gate is opt-in, so the target is only claimable with it on.
    The default-off run above is the "never regress" guard; this is the
    "actually good enough" guard. Both must hold.
    """
    target = bc.QUALITY_TARGET["onnx"]["holdout"]
    result = _run(["run_behavior.py", "--set", "holdout", "--mode", "auto"],
                  timeout=600, answerability=True)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "SKIPPED" not in result.stdout, result.stdout[-2000:]
    # run_behavior prints e.g. "[onnx/holdout] 10/11 = 91%"
    line = next((ln for ln in result.stdout.splitlines()
                 if ln.startswith("[onnx/holdout]")), "")
    assert line, result.stdout[-2000:]
    rate = int(line.split("=")[1].split("%")[0].strip()) / 100.0
    assert rate >= target, f"holdout {rate:.0%} below target {target:.0%}: {line}"


def test_answerability_gate_does_not_touch_the_default_path():
    """Off by default, and enabling it must not invent hits or leak secrets."""
    off = _run(["run_behavior.py", "--set", "holdout", "--mode", "hash"],
               timeout=600, answerability=False)
    assert off.returncode == 0, off.stdout[-3000:]
    assert "provenance/safety invariants: PASS" in off.stdout
    on = _run(["run_behavior.py", "--set", "holdout", "--mode", "hash"],
              timeout=600, answerability=True)
    assert on.returncode == 0, on.stdout[-3000:]
    assert "provenance/safety invariants: PASS" in on.stdout
