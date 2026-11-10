"""A lock that cannot be taken must be a refusal, not a traceback.

``lifecycle_lock`` opened the lock file with a bare ``os.open`` outside its ``try``.
An unwritable or unwalkable data directory — a root-owned file, a read-only mount,
a full disk — raised ``PermissionError``/``OSError`` straight out of ``forget``,
``supersede`` and ``compile``. Every other failure in the lifecycle is a structured
``{"ok": false, "error": ...}``, so callers that read the result had to catch
exceptions from some paths and not others.

The rule: acquiring the lock either succeeds or raises a ``LifecycleLockError``
that the tool layer turns into a normal structured refusal.

Run: python -m pytest tests/test_lifecycle_lock_errors.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.facts import Embedder, LifecycleLockError, ProposalStore

MEM = Path("Hermes/Memory/Auto")


def _store(tmp: Path) -> ProposalStore:
    return ProposalStore(tmp / "facts.sqlite", embedder=Embedder(mode="hash"))


def test_unwritable_lock_location_raises_a_typed_error(tmp_path: Path):
    """A directory we cannot create the lock file in fails as a typed refusal."""
    store = _store(tmp_path)
    # Point the store's database at a path whose parent cannot be created.
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    os.chmod(blocked, 0o500)  # read + execute, no write
    store.database = blocked / "facts.sqlite"
    try:
        with pytest.raises(LifecycleLockError):
            with store.lifecycle_lock():
                pass  # pragma: no cover - acquisition must fail first
    finally:
        os.chmod(blocked, 0o700)


def test_the_typed_error_is_an_os_error_subclass(tmp_path: Path):
    """Existing ``except OSError`` callers keep working."""
    assert issubclass(LifecycleLockError, OSError)


def test_a_normal_lock_still_works(tmp_path: Path):
    """The happy path is unchanged and remains reentrant."""
    store = _store(tmp_path)
    with store.lifecycle_lock():
        with store.lifecycle_lock():
            pass
    # And it is released: a second acquisition outside the nest is fine.
    with store.lifecycle_lock():
        pass
