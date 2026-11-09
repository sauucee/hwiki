"""The public read/status/graph contracts, as consumers see them.

These are the fields an agent or the CLI reads. Review required that identity fixes
land as *additive* fields rather than silently re-keying what already exists, so every
old field keeps its old shape and value; the path-exact views are new fields, and the
payload carries a contract version.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import _hwiki_path  # noqa: F401,E402

from hwiki import WikiMemoryProvider  # noqa: E402
from hwiki.store import WikiStore  # noqa: E402

MEM = "wiki/memory"


def _vault(root: Path) -> None:
    for folder in ("wiki/a", "wiki/b", MEM, "home"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "wiki" / "a" / "decisions.md").write_text("# A\n", encoding="utf-8")
    (root / "wiki" / "b" / "decisions.md").write_text("# B\n", encoding="utf-8")
    (root / "wiki" / "a" / "note.md").write_text(
        "# note a\n\nSee [[wiki/a/decisions]] and [[nowhere]].\n", encoding="utf-8")
    (root / "wiki" / "b" / "note.md").write_text(
        "# note b\n\nSee [[wiki/b/decisions]].\n", encoding="utf-8")
    (root / "wiki" / "orphan.md").write_text("# orphan\n", encoding="utf-8")


class GraphContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _vault(self.root)
        self.store = WikiStore(self.root, memory_dir=MEM)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dangling_links_keeps_slug_keys_by_default(self):
        dangling = self.store.dangling_links()
        self.assertEqual(list(dangling), ["note"], dangling)
        self.assertEqual(dangling["note"], ["nowhere"])

    def test_dangling_links_can_be_keyed_by_path_explicitly(self):
        dangling = self.store.dangling_links(key="path")
        self.assertEqual(list(dangling), ["wiki/a/note.md"], dangling)

    def test_dangling_links_rejects_an_unknown_key(self):
        with self.assertRaises(ValueError):
            self.store.dangling_links(key="identity")

    def test_link_graph_path_view_is_opt_in(self):
        _out, by_slug = self.store.link_graph()
        _out2, by_path = self.store.link_graph(key="path")
        self.assertIn("decisions", by_slug)
        self.assertIn("wiki/a/decisions.md", by_path)


class ProviderContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _vault(self.root)
        self.home = self.root / "home"
        self.provider = WikiMemoryProvider(config={
            "wiki_path": str(self.root), "backend": "fts5", "memory_dir": MEM,
            "index_path": str(self.home / "index.sqlite"), "reindex_interval_s": 0,
            "catalog_enabled": False, "_hermes_home": str(self.home),
        })
        self.provider.initialize("s", hermes_home=str(self.home), platform="cli")

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, page: str) -> dict:
        return json.loads(self.provider.handle_tool_call("wiki_memory_read", {"page": page}))

    def test_read_keeps_slug_backlinks_and_adds_paths(self):
        payload = self.read("wiki/a/decisions")
        self.assertIn("backlinks", payload)          # old field, old shape
        self.assertTrue(all("/" not in name for name in payload["backlinks"]),
                        payload["backlinks"])
        self.assertIn("backlink_paths", payload)     # additive, exact
        self.assertIn("wiki/a/note.md", payload["backlink_paths"])
        self.assertNotIn("wiki/b/note.md", payload["backlink_paths"])

    def test_status_keeps_slug_orphans_and_adds_paths(self):
        payload = json.loads(self.provider.handle_tool_call(
            "wiki_memory_status", {"lint": True}))
        lint = payload["lint"]
        self.assertEqual(lint["contract_version"], 2)
        self.assertIn("orphan_pages", lint)
        self.assertIn("orphan_page_paths", lint)
        self.assertTrue(all("/" not in name for name in lint["orphan_pages"]),
                        lint["orphan_pages"])
        self.assertIn("wiki/orphan.md", lint["orphan_page_paths"])
        self.assertEqual(lint["orphan_count"], len(lint["orphan_pages"]))
        self.assertEqual(lint["orphan_path_count"], len(lint["orphan_page_paths"]))
        self.assertNotIn("wiki/a/decisions.md", lint["orphan_page_paths"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
