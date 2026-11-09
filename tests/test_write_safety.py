"""Writes must fail closed: nothing outside the vault, nothing while unlocked.

The vault can be a shared directory and the log is an audit trail, so a write that
cannot be proven safe is refused instead of attempted. A symlink, a hardlink, a
symlinked parent directory, a replaced file or a lock that could not be acquired must
all leave the target untouched.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _hwiki_path  # noqa: F401,E402

from hwiki import catalog  # noqa: E402
from hwiki import store as store_module  # noqa: E402
from hwiki.store import WikiStore, _FileLock  # noqa: E402


class WriteContainmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "vault"
        (self.root / "memory").mkdir(parents=True)
        self.store = WikiStore(self.root, memory_dir="memory")
        self.outside = self.base / "outside"
        self.outside.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    # -- log -------------------------------------------------------------

    def test_append_log_refuses_a_hardlinked_log(self):
        """A hardlink to an outside file is the same inode: appending is writing there."""
        secret = self.outside / "log.md"
        secret.write_text(catalog._LOG_HEADER, encoding="utf-8")  # looks owned
        os.link(secret, self.root / "log.md")
        self.assertIsNone(catalog.append_log(self.store, "write", "escape"))
        self.assertEqual(secret.read_text(encoding="utf-8"), catalog._LOG_HEADER)

    def test_append_log_refuses_a_symlinked_parent_directory(self):
        (self.root / "shared").symlink_to(self.outside, target_is_directory=True)
        self.assertIsNone(catalog.append_log(
            self.store, "write", "escape", path=self.root / "shared" / "log.md"))
        self.assertFalse((self.outside / "log.md").exists())

    def test_append_log_refuses_a_log_replaced_after_the_ownership_check(self):
        """Ownership must be re-checked under the lock, not only before it."""
        log = self.root / "log.md"
        log.write_text(catalog._LOG_HEADER, encoding="utf-8")
        real_lock = catalog._FileLock

        class Swap(real_lock):  # type: ignore[misc,valid-type]
            def __enter__(self):
                super().__enter__()
                log.write_text("handwritten user content\n", encoding="utf-8")
                return self

        with patch.object(catalog, "_FileLock", Swap):
            self.assertIsNone(catalog.append_log(self.store, "event", "swap"))
        self.assertEqual(log.read_text(encoding="utf-8"), "handwritten user content\n")

    # -- memory notes ----------------------------------------------------

    def test_a_memory_note_symlink_does_not_write_outside(self):
        secret = self.outside / "note.md"
        secret.write_text("original\n", encoding="utf-8")
        (self.root / "memory" / "note.md").symlink_to(secret)
        with self.assertRaises(ValueError):
            self.store.append_note("injected", page="note")
        self.assertEqual(secret.read_text(encoding="utf-8"), "original\n")

    def test_a_memory_note_hardlink_does_not_write_outside(self):
        secret = self.outside / "note.md"
        secret.write_text("original\n", encoding="utf-8")
        os.link(secret, self.root / "memory" / "note.md")
        with self.assertRaises(ValueError):
            self.store.append_note("injected", page="note")
        self.assertEqual(secret.read_text(encoding="utf-8"), "original\n")

    def test_a_symlinked_memory_zone_does_not_write_outside(self):
        zone = self.root / "zone"
        zone.symlink_to(self.outside, target_is_directory=True)
        store = WikiStore(self.root, memory_dir="zone")
        with self.assertRaises(ValueError):
            store.append_note("injected", page="note")
        self.assertFalse((self.outside / "note.md").exists())

    def test_a_regular_note_still_appends(self):
        result = self.store.append_note("remember this", page="note")
        self.assertIn("note", result["path"])
        self.assertTrue(result["created"])
        again = self.store.append_note("and this", page="note")
        self.assertFalse(again["created"])


    def test_append_log_refuses_a_parent_swapped_after_validation(self):
        """The parent check and the open must be anchored, not two path lookups."""
        inside = self.root / "sub"
        inside.mkdir()
        real_lock = catalog._FileLock

        class Swap(real_lock):  # type: ignore[misc,valid-type]
            def __init__(self, target, **kwargs):
                self._target = target
                self._kwargs = kwargs

            def __enter__(self):
                inside.rename(self.root / "sub_saved")
                inside.symlink_to(self.outside, target_is_directory=True)
                self._real = real_lock(self._target, **self._kwargs)
                return self._real.__enter__()

            def __exit__(self, *exc):
                return self._real.__exit__(*exc)

        with patch.object(catalog, "_FileLock", Swap):
            self.assertIsNone(catalog.append_log(
                self.store, "event", "escape", path=inside / "log.md"))
        self.assertFalse((self.outside / "log.md").exists())
        self.assertFalse((self.outside / "log.md.lock").exists())

class LockFailClosedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "memory").mkdir()
        self.store = WikiStore(self.root, memory_dir="memory")
        self.log = self.root / "log.md"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_lock_timeout_raises_instead_of_continuing_unlocked(self):
        holder = _FileLock(self.log, timeout=1.0)
        holder.__enter__()
        try:
            with self.assertRaises(TimeoutError):
                with _FileLock(self.log, timeout=0.1):
                    self.fail("entered the critical section without the lock")
        finally:
            holder.__exit__(None, None, None)

    def test_append_log_writes_nothing_when_the_lock_times_out(self):
        self.log.write_text(catalog._LOG_HEADER, encoding="utf-8")
        before = self.log.read_bytes()
        holder = _FileLock(self.log, timeout=1.0)
        holder.__enter__()
        try:
            class Short(_FileLock):
                def __init__(self, target, **kwargs):
                    super().__init__(target, timeout=0.1, **kwargs)

            with patch.object(catalog, "_FileLock", Short):
                self.assertIsNone(catalog.append_log(self.store, "event", "timeout"))
        finally:
            holder.__exit__(None, None, None)
        self.assertEqual(self.log.read_bytes(), before)

    def test_write_index_does_not_clobber_a_file_replaced_after_the_check(self):
        """Ownership then replace is a race; the replacement must be verified, not assumed."""
        index = self.root / "index.md"
        index.write_text("---\nsource: hwiki-catalog\n---\nold\n", encoding="utf-8")
        original = catalog.build_index

        def swapped(*args, **kwargs):
            text = original(*args, **kwargs)
            index.write_text("precious hand-written navigation\n", encoding="utf-8")
            return text

        with patch.object(catalog, "build_index", swapped):
            self.assertIsNone(catalog.write_index(self.store))
        self.assertEqual(index.read_text(encoding="utf-8"),
                         "precious hand-written navigation\n")

    def test_the_marker_mentioned_in_a_comment_does_not_make_a_file_ours(self):
        index = self.root / "index.md"
        handwritten = ("---\ntitle: My Index\n"
                       "comment: source: hwiki-catalog is not my generator\n---\n"
                       "precious manual text\n")
        index.write_text(handwritten, encoding="utf-8")
        self.assertFalse(catalog.owns_artifact(index))
        self.assertIsNone(catalog.write_index(self.store))
        self.assertEqual(index.read_text(encoding="utf-8"), handwritten)

        log = self.root / "log.md"
        log.write_text(handwritten, encoding="utf-8")
        self.assertIsNone(catalog.append_log(self.store, "write", "nope"))
        self.assertEqual(log.read_text(encoding="utf-8"), handwritten)

    def test_a_real_source_field_still_marks_the_file_as_generated(self):
        index = self.root / "index.md"
        index.write_text("---\ntitle: Index\nsource: hwiki-catalog\n---\n# Index\n",
                         encoding="utf-8")
        self.assertTrue(catalog.owns_artifact(index))
        self.assertIsNotNone(catalog.write_index(self.store))


class LockFailClosedTests2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "memory").mkdir()
        self.store = WikiStore(self.root, memory_dir="memory")
        self.log = self.root / "log.md"

    def tearDown(self):
        self.tmp.cleanup()

    def test_without_advisory_locking_the_lock_refuses_to_enter(self):
        with patch.object(store_module, "fcntl", None):
            with self.assertRaises(RuntimeError):
                with _FileLock(self.log):
                    self.fail("entered the critical section without any locking")

    def test_a_lock_timeout_closes_the_descriptor_it_opened(self):
        holder = _FileLock(self.log, timeout=1.0)
        holder.__enter__()
        opened = []
        real_fdopen = os.fdopen

        def recording(fd, *args, **kwargs):
            handle = real_fdopen(fd, *args, **kwargs)
            opened.append(handle)
            return handle

        try:
            with patch.object(store_module.os, "fdopen", recording):
                with self.assertRaises(TimeoutError):
                    _FileLock(self.log, timeout=0.1).__enter__()
        finally:
            holder.__exit__(None, None, None)
        self.assertTrue(opened, "the lock never opened its descriptor")
        self.assertTrue(opened[-1].closed, "lock descriptor leaked on timeout")

if __name__ == "__main__":
    unittest.main(verbosity=2)
