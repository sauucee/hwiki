"""Tests for ``wiki_memory_file`` — filing a synthesised answer back.

Every test runs against a throwaway vault; the live Obsidian vault is never
touched. The answer pages are asserted to stay out of the accepted-fact
recall path while still being searchable.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _hwiki_path  # noqa: F401,E402

from hwiki import ANSWERS_SUBDIR, WikiMemoryProvider  # noqa: E402

SECRET_ANSWER = "Set the token with password=hunter2024 in the config file."


class FileAnswerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.wiki = self.root / "vault"
        self.home = self.root / "hermes_home"
        (self.wiki / "Hermes" / "Memory" / "Auto").mkdir(parents=True)
        (self.wiki / "concepts").mkdir(parents=True)
        (self.wiki / "concepts" / "seed.md").write_text(
            "# Seed page\n\nA page so the index is non-empty.\n", encoding="utf-8")
        self.home.mkdir()
        self._providers = []

    def tearDown(self):
        for provider in self._providers:
            try:
                provider.shutdown()
            except Exception:
                pass
        self.tmp.cleanup()

    def provider(self, read_only=False, **overrides):
        cfg = {
            "wiki_path": str(self.wiki),
            "backend": "fts5",
            "memory_dir": "Hermes/Memory/Auto",
            "index_path": str(self.home / "index.sqlite"),
            "reindex_interval_s": 0,
            "_hermes_home": str(self.home),
        }
        cfg.update(overrides)
        prov = WikiMemoryProvider(config=cfg)
        # agent_context is a runtime kwarg, not a config key.
        prov.initialize(session_id="test-session",
                        agent_context="cron" if read_only else "primary")
        self._providers.append(prov)
        return prov

    def file(self, provider, question="Why is the sky blue?", answer="Rayleigh scattering.",
             **kw):
        args = {"question": question, "answer": answer}
        args.update(kw)
        return json.loads(provider.handle_tool_call("wiki_memory_file", args))

    # -- happy path ------------------------------------------------------

    def test_files_answer_into_answers_subfolder(self):
        prov = self.provider()
        out = self.file(prov)
        self.assertTrue(out["success"], out)
        self.assertTrue(out["created_page"])
        self.assertFalse(out["duplicate"])
        path = self.wiki / out["path"]
        self.assertTrue(path.exists())
        self.assertEqual(path.parent.name, ANSWERS_SUBDIR)
        self.assertEqual(Path(out["path"]).parts[0:2],
                         ("Hermes", "Memory"))

    def test_page_has_answer_frontmatter_and_body(self):
        prov = self.provider()
        out = self.file(prov, tags="physics, optics")
        text = (self.wiki / out["path"]).read_text(encoding="utf-8")
        self.assertIn("type: answer", text)
        self.assertIn("status: filed", text)
        self.assertIn("# Why is the sky blue?", text)
        self.assertIn("Rayleigh scattering.", text)
        self.assertIn("#physics", text)
        self.assertIn("#optics", text)

    def test_filed_answer_is_searchable(self):
        prov = self.provider()
        self.file(prov, question="Why is the sky blue?",
                  answer="Rayleigh scattering of short wavelengths.")
        hits = json.loads(prov.handle_tool_call(
            "wiki_memory_search", {"query": "Rayleigh scattering"}))
        self.assertTrue(hits["success"])
        self.assertGreater(hits["count"], 0)
        paths = [h["path"] for h in hits["results"]]
        self.assertTrue(any(ANSWERS_SUBDIR in p for p in paths), paths)

    # -- idempotency -----------------------------------------------------

    def test_refiling_same_question_is_a_no_op(self):
        prov = self.provider()
        first = self.file(prov, answer="First answer.")
        path = self.wiki / first["path"]
        before = path.read_bytes()

        again = self.file(prov, answer="A completely different answer.")
        self.assertTrue(again["success"])
        self.assertTrue(again["duplicate"])
        self.assertFalse(again["created_page"])
        self.assertEqual(path.read_bytes(), before,
                         "an existing answer must not be overwritten")

    def test_case_and_whitespace_only_difference_is_a_duplicate(self):
        prov = self.provider()
        first = self.file(prov, question="Why is the sky blue?")
        again = self.file(prov, question="  why is   the SKY blue?  ")
        self.assertTrue(again["duplicate"])
        self.assertEqual(again["path"], first["path"])

    def test_slug_collision_files_a_sister_page(self):
        """Two different questions that slugify alike must both survive."""
        prov = self.provider()
        first = self.file(prov, question="What is C++?")
        second = self.file(prov, question="What is C#?")
        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        self.assertTrue(second["created_page"])
        self.assertNotEqual(first["path"], second["path"],
                            "a different question must never clobber the first")
        for out in (first, second):
            self.assertTrue((self.wiki / out["path"]).exists())

    # -- refusals --------------------------------------------------------

    def test_secret_answer_is_refused_and_nothing_written(self):
        prov = self.provider()
        out = self.file(prov, answer=SECRET_ANSWER)
        self.assertFalse(out["success"])
        answers = self.wiki / "Hermes" / "Memory" / "Auto" / ANSWERS_SUBDIR
        written = list(answers.glob("*.md")) if answers.is_dir() else []
        self.assertEqual(written, [], [p.name for p in written])

    def test_read_only_provider_refuses(self):
        prov = self.provider(read_only=True)
        out = self.file(prov)
        self.assertFalse(out["success"])

    def test_read_only_provider_does_not_advertise_the_tool(self):
        prov = self.provider(read_only=True)
        names = [s["name"] for s in prov.get_tool_schemas()]
        self.assertNotIn("wiki_memory_file", names)
        self.assertIn("wiki_memory_search", names)

    def test_missing_question_or_answer_is_rejected(self):
        prov = self.provider()
        self.assertFalse(json.loads(prov.handle_tool_call(
            "wiki_memory_file", {"answer": "orphan"}))["success"])
        self.assertFalse(json.loads(prov.handle_tool_call(
            "wiki_memory_file", {"question": "orphan?"}))["success"])

    # -- edge cases ------------------------------------------------------

    def test_unicode_question_files_cleanly(self):
        prov = self.provider()
        out = self.file(prov, question="Pourquoi le ciel est-il bleu ? ☁️",
                        answer="Diffusion de Rayleigh.")
        self.assertTrue(out["success"], out)
        text = (self.wiki / out["path"]).read_text(encoding="utf-8")
        self.assertIn("Diffusion de Rayleigh.", text)

    def test_long_question_title_is_bounded(self):
        prov = self.provider()
        out = self.file(prov, question="word " * 200 + "?")
        self.assertTrue(out["success"], out)
        text = (self.wiki / out["path"]).read_text(encoding="utf-8")
        h1 = [line for line in text.splitlines() if line.startswith("# ")][0]
        self.assertLessEqual(len(h1), 162)

    def test_answer_does_not_enter_fact_recall(self):
        """Filed answers are reference pages, not user-approved facts."""
        prov = self.provider()
        self.file(prov, question="Should replies be long?",
                  answer="No, keep them short.")
        from hwiki.facts import read_fact_pages
        facts = read_fact_pages(self.wiki, Path("Hermes/Memory/Auto"), max_facts=50)
        claims = [f["claim"] for f in facts]
        self.assertNotIn("Should replies be long?", claims)

    def test_no_temp_file_left_behind(self):
        prov = self.provider()
        self.file(prov)
        answers = self.wiki / "Hermes" / "Memory" / "Auto" / ANSWERS_SUBDIR
        self.assertEqual([p.name for p in answers.glob("*.tmp")], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
