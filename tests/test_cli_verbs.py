"""Tests for the ``file`` and ``confidence`` CLI verbs.

``file_answer`` moved out of the provider so the agent tool and the CLI share one
implementation; these tests pin the shared function directly and then drive the
verbs end to end through argv. Every test uses a throwaway vault and a throwaway
``HERMES_HOME`` — the live Obsidian vault is never touched.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import _hwiki_path  # noqa: F401,E402

from hwiki import ANSWERS_SUBDIR  # noqa: E402
from hwiki.facts import Embedder, ProposalStore, file_answer  # noqa: E402

SECRET = "Set the token with password=hunter2024 in the config."


class FileAnswerFunctionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = Path(self.tmp.name) / "Hermes" / "Memory" / "Auto"
        self.memory.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_page_in_answers_subdir(self):
        result = file_answer(self.memory, "where do we deploy?", "Fly.io.")
        self.assertTrue(result["success"])
        self.assertTrue(result["created_page"])
        page = Path(result["path"])
        self.assertEqual(page.parent.name, ANSWERS_SUBDIR)
        self.assertIn("Fly.io.", page.read_text(encoding="utf-8"))

    def test_empty_question_and_answer_are_refused(self):
        self.assertFalse(file_answer(self.memory, "   ", "text")["success"])
        self.assertFalse(file_answer(self.memory, "q", "  \n ")["success"])
        self.assertFalse((self.memory / ANSWERS_SUBDIR).exists())

    def test_secret_is_refused_and_nothing_is_written(self):
        result = file_answer(self.memory, "how do I auth?", SECRET)
        self.assertFalse(result["success"])
        self.assertIn("secret", result["error"].lower())
        self.assertFalse((self.memory / ANSWERS_SUBDIR).exists())

    def test_refiling_the_same_question_is_idempotent(self):
        first = file_answer(self.memory, "where do we deploy?", "Fly.io.")
        before = Path(first["path"]).read_text(encoding="utf-8")
        again = file_answer(self.memory, "where do we deploy?", "Hetzner now.")
        self.assertTrue(again["duplicate"])
        self.assertFalse(again["created_page"])
        self.assertEqual(Path(first["path"]).read_text(encoding="utf-8"), before)

    def test_whitespace_in_the_question_does_not_defeat_idempotency(self):
        first = file_answer(self.memory, "where   do we deploy?", "Fly.io.")
        again = file_answer(self.memory, "  where do we deploy?  ", "Fly.io.")
        self.assertTrue(again["duplicate"])
        self.assertEqual(first["path"], again["path"])

    def test_two_different_questions_with_one_slug_both_survive(self):
        first = file_answer(self.memory, "Where do we deploy?", "Fly.io.")
        second = file_answer(self.memory, "where do we deploy!", "Hetzner.")
        self.assertTrue(second["created_page"])
        self.assertNotEqual(first["path"], second["path"])
        self.assertIn("Fly.io.", Path(first["path"]).read_text(encoding="utf-8"))

    def test_tags_are_normalised_into_the_page(self):
        result = file_answer(self.memory, "deploy?", "Fly.io.", ["Ops Notes", "deploy"])
        text = Path(result["path"]).read_text(encoding="utf-8")
        self.assertIn("#ops-notes", text)
        self.assertIn("#deploy", text)

    def test_page_carries_frontmatter_for_lint(self):
        result = file_answer(self.memory, "deploy?", "Fly.io.")
        text = Path(result["path"]).read_text(encoding="utf-8")
        for field in ("title:", "type: answer", "status: filed", "source: hwiki"):
            self.assertIn(field, text)
        self.assertTrue(text.startswith("---\n"))


class _VaultCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.wiki = root / "vault"
        self.memory = self.wiki / "Hermes" / "Memory" / "Auto"
        self.memory.mkdir(parents=True)
        (self.wiki / "concepts").mkdir(parents=True)
        (self.wiki / "concepts" / "seed.md").write_text("# Seed\n", encoding="utf-8")
        self.home = root / "hermes_home"
        self.home.mkdir()
        self.config_path = self.home / "hwiki.json"
        self.config_path.write_text(json.dumps({
            "wiki_path": str(self.wiki),
            "memory_dir": "Hermes/Memory/Auto",
            "embeddings": {"mode": "hash"},
        }), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, argv, stdin=""):
        from hwiki import cli
        buf = io.StringIO()
        env = {"HERMES_HOME": str(self.home)}
        with mock.patch.dict(os.environ, env), \
                mock.patch("sys.stdin", io.StringIO(stdin)), \
                redirect_stdout(buf):
            import argparse
            parser = argparse.ArgumentParser(prog="hwiki")
            cli.register_cli(parser)
            args = parser.parse_args(argv)
            args.func(args)
        return buf.getvalue()


class FileVerbTests(_VaultCase):
    def test_file_verb_writes_a_page(self):
        out = self.run_cli(["file", "-q", "where do we deploy?"], stdin="Fly.io.\n")
        self.assertIn("Filed:", out)
        self.assertTrue(list((self.memory / ANSWERS_SUBDIR).glob("*.md")))

    def test_file_verb_prints_a_refusal_and_exits_cleanly(self):
        out = self.run_cli(["file", "-q", "auth?"], stdin=SECRET)
        self.assertIn("Refused:", out)
        self.assertIn("secret", out.lower())
        self.assertFalse(list((self.memory / ANSWERS_SUBDIR).glob("*.md")))

    def test_file_verb_json_mode_is_machine_readable(self):
        out = self.run_cli(["file", "-q", "deploy?", "--json"], stdin="Fly.io.\n")
        payload = json.loads(out)
        self.assertTrue(payload["success"])
        self.assertTrue(payload["created_page"])

    def test_file_verb_without_a_question_prints_usage(self):
        out = self.run_cli(["file"], stdin="orphan text\n")
        self.assertIn("Usage:", out)


class ConfidenceVerbTests(_VaultCase):
    def _seed(self):
        db = self.home / "plugins-data" / "hwiki" / "facts.sqlite"
        db.parent.mkdir(parents=True)
        store = ProposalStore(str(db), embedder=Embedder(mode="hash"))
        for claim in ("The staging server stays on Fly.io",
                      "Backups run nightly with restic to B2"):
            proposal = store.propose({"kind": "environment", "claim": claim,
                                      "evidence": "stated"},
                                     user_text="stated", session_id="s")
            store.accept(proposal["id"])
        return store

    def test_confidence_reports_the_signal_and_the_caveat(self):
        self._seed()
        out = self.run_cli(["confidence", "where does staging run"])
        self.assertIn("confident", out)
        self.assertIn("relatedness is not answerability", out)

    def test_confidence_json_has_the_documented_keys(self):
        self._seed()
        out = self.run_cli(["confidence", "where does staging run", "--json"])
        payload = json.loads(out)
        self.assertEqual(set(payload), {"best", "margin", "overlap", "candidates",
                                        "floor", "confident", "caveat"})

    def test_confidence_refuses_when_no_facts_db_exists(self):
        out = self.run_cli(["confidence", "anything"])
        self.assertIn("No facts database", out)
        self.assertFalse((self.home / "plugins-data" / "hwiki" / "facts.sqlite").exists())

    def test_confidence_without_a_query_prints_usage(self):
        self.assertIn("Usage:", self.run_cli(["confidence"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
