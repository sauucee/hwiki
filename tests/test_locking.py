"""The write lock and the artifact write path, as an attacker sees them.

``log.md`` and its lockfile live in the vault, which may be a shared directory. A
symlink planted there must not redirect a write outside the vault, and the lock must
stay on one inode for its whole lifetime — otherwise a waiter on the old inode and a
newcomer on a fresh inode can both hold "the lock" at once.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

import _hwiki_path  # noqa: F401,E402

from hwiki import catalog  # noqa: E402
from hwiki.store import WikiStore, _FileLock  # noqa: E402

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None


class SymlinkWriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vault"
        self.root.mkdir()
        (self.root / "memory").mkdir()
        self.store = WikiStore(self.root, memory_dir="memory")
        self.outside = Path(self.tmp.name) / "outside"
        self.outside.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_log_refuses_a_log_symlinked_outside_the_vault(self):
        secret = self.outside / "log.md"
        secret.write_bytes(b"")
        (self.root / "log.md").symlink_to(secret)
        result = catalog.append_log(self.store, "write", "must not escape")
        self.assertIsNone(result)
        self.assertEqual(secret.read_bytes(), b"", "wrote through a symlink")

    def test_append_log_refuses_a_dangling_log_symlink(self):
        missing = self.outside / "not-there.md"
        (self.root / "log.md").symlink_to(missing)
        self.assertIsNone(catalog.append_log(self.store, "write", "must not create"))
        self.assertFalse(missing.exists(), "created a file outside the vault via symlink")

    def test_append_log_accepts_a_regular_in_vault_log(self):
        self.assertIsNotNone(catalog.append_log(self.store, "write", "fine"))
        self.assertTrue((self.root / "log.md").is_file())

    def test_write_index_refuses_a_symlink_instead_of_writing_through_it(self):
        secret = self.outside / "index.md"
        secret.write_bytes(b"")
        (self.root / "index.md").symlink_to(secret)
        self.assertIsNone(catalog.write_index(self.store))
        self.assertEqual(secret.read_bytes(), b"", "wrote through a symlink")


class LockInodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.target = Path(self.tmp.name) / "log.md"

    def tearDown(self):
        self.tmp.cleanup()

    @unittest.skipIf(fcntl is None, "advisory locks are POSIX only")
    def test_lockfile_inode_is_stable_across_release(self):
        with _FileLock(self.target):
            ino = self.target.with_suffix(".md.lock").stat().st_ino
        with _FileLock(self.target):
            again = self.target.with_suffix(".md.lock").stat().st_ino
        self.assertEqual(ino, again)

    @unittest.skipIf(fcntl is None, "advisory locks are POSIX only")
    def test_a_waiter_on_the_old_inode_blocks_a_newcomer_too(self):
        """The ABA case: holder releases, a waiter holds the old inode, newcomer arrives.

        With an unlink on release the newcomer creates a fresh inode and locks it
        immediately while the waiter holds the old one — two writers inside. A stable
        inode means the newcomer's flock blocks, so it must time out *unlocked*.
        """
        lockfile = self.target.with_suffix(".md.lock")
        holder = _FileLock(self.target)
        holder.__enter__()
        waiter_fd = os.open(lockfile, os.O_RDWR | os.O_CREAT, 0o644)
        inside = threading.Event()
        started = threading.Event()

        def wait_for_it():
            started.set()
            fcntl.flock(waiter_fd, fcntl.LOCK_EX)
            inside.set()

        thread = threading.Thread(target=wait_for_it, daemon=True)
        thread.start()
        started.wait(2)
        holder.__exit__(None, None, None)
        try:
            self.assertTrue(inside.wait(2), "waiter never got the lock")
            newcomer = _FileLock(self.target, timeout=0.3)
            started_at = time.time()
            with self.assertRaises(TimeoutError):
                newcomer.__enter__()
            self.assertGreaterEqual(time.time() - started_at, 0.25)
        finally:
            fcntl.flock(waiter_fd, fcntl.LOCK_UN)
            os.close(waiter_fd)
            thread.join(2)

    @unittest.skipIf(fcntl is None, "advisory locks are POSIX only")
    def test_a_symlinked_lockfile_is_not_followed(self):
        secret = Path(self.tmp.name) / "outside.lock"
        secret.write_bytes(b"")
        lockfile = self.target.with_suffix(".md.lock")
        lockfile.symlink_to(secret)
        with self.assertRaises(OSError):
            with _FileLock(self.target, timeout=0.2):
                pass
        self.assertEqual(secret.read_bytes(), b"")


class CatalogLinkIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vault"
        (self.root / "Projects").mkdir(parents=True)
        self.store = WikiStore(self.root, memory_dir="memory")

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_entry_resolves_to_the_page_it_names_not_the_root_artifact(self):
        (self.root / "index.md").write_text("# Index\n", encoding="utf-8")
        (self.root / "Projects" / "index.md").write_text("# Project index\n",
                                                          encoding="utf-8")
        index = catalog.build_index(self.store)
        entries = [line for line in index.splitlines()
                   if line.startswith("- ") and "[[" in line]
        self.assertTrue(entries, index)
        for line in entries:
            target = line.split("[[", 1)[1].split("]]", 1)[0].split("|", 1)[0]
            page = self.store.resolve_page(target)
            self.assertIsNotNone(page, f"catalog emitted unresolved link {target!r}")
            self.assertIn(page.path, ("Projects/index.md",), line)

    def test_a_colliding_curated_page_is_never_hidden_as_an_artifact(self):
        (self.root / "index.md").write_text("# Index\n", encoding="utf-8")
        (self.root / "Projects" / "index.md").write_text("# Project index\n",
                                                          encoding="utf-8")
        index = catalog.build_index(self.store)
        self.assertIn("Projects/index", index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
