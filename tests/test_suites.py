"""Run the script-style suites under pytest so `pytest tests/` covers everything.

Each suite is also runnable on its own (`python tests/test_provider.py`) and needs
no Hermes install. Here each runs in a fresh subprocess, because the provider and
CLI suites deliberately manipulate module registration and HERMES_HOME.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent


@pytest.mark.parametrize("script", ["test_store_search.py", "test_provider.py", "test_cli_load.py"])
def test_suite_passes(script: str) -> None:
    proc = subprocess.run(
        [sys.executable, script], cwd=TESTS, capture_output=True, text=True, timeout=300,
    )
    tail = "\n".join(proc.stdout.strip().splitlines()[-25:])
    assert proc.returncode == 0, f"{script} failed:\n{tail}\n{proc.stderr[-2000:]}"
    assert " 0 failed" in proc.stdout, tail


# -- script mode must not silently skip tests -------------------------------
#
# A suite file that calls ``unittest.main()`` before its last class runs only the
# classes defined above that line when invoked directly: the file still prints OK,
# so the missing coverage is invisible. Five suites had drifted that way. This check
# compares what direct invocation discovers with what pytest discovers, so a block
# added in the wrong place fails here instead of quietly shrinking the suite.

SUITE_FILES = sorted(
    path.name for path in TESTS.glob("test_*.py")
    if re.search(r"^\s+unittest\.main\(", path.read_text(encoding="utf-8", errors="replace"),
                 re.M)
)


def _script_mode_ids(script: str) -> list:
    """(Class, test) pairs that ``python tests/<script>`` actually runs."""
    proc = subprocess.run(
        [sys.executable, script, "-v"], cwd=TESTS,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    output = proc.stdout + proc.stderr
    return [f"{cls}.{name}" for name, cls in re.findall(
        r"^(\w+) \(__main__\.(\w+)\.", output, re.M)]


def _pytest_ids(script: str) -> list:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", script],
        cwd=TESTS, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"collection failed for {script}:\n{proc.stdout[-2000:]}"
    ids = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split("::")
        if len(parts) == 3 and parts[0].endswith(script):
            ids.append(f"{parts[1]}.{parts[2]}")
    return ids


@pytest.mark.parametrize("script", SUITE_FILES)
def test_script_mode_runs_every_test_pytest_collects(script: str) -> None:
    script_ids = _script_mode_ids(script)
    pytest_ids = _pytest_ids(script)
    assert pytest_ids, f"pytest collected nothing from {script}"
    assert len(script_ids) == len(set(script_ids)), (
        f"{script} ran a test twice: "
        f"{sorted({i for i in script_ids if script_ids.count(i) > 1})}")
    assert set(script_ids) == set(pytest_ids), (
        f"{script}: direct invocation skipped "
        f"{sorted(set(pytest_ids) - set(script_ids))}")
