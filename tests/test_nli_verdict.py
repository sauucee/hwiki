"""Tests for the optional NLI tier (``hwiki.nli`` / ``hwiki.verdict``).

No test here needs the 165 MB model: the scorer is stubbed, so the suite stays fast
and runs on a machine where the bundle is absent. The single test that does touch
the real model skips itself when it is missing.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _hwiki_path  # noqa: F401,E402

from hwiki import facts, nli, verdict  # noqa: E402


class _Stub:
    """Stands in for NliScorer: returns preset (contradiction, entailment, neutral)."""

    def __init__(self, contradiction=0.0, entailment=0.0, neutral=0.0):
        self.triple = (contradiction, entailment, neutral)
        self.calls = []

    def probabilities(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        return self.triple

    def entailment(self, premise, hypothesis):
        return self.probabilities(premise, hypothesis)[1]

    def contradiction(self, premise, hypothesis):
        return self.probabilities(premise, hypothesis)[0]


class ModelDirTests(unittest.TestCase):
    def test_env_override_wins(self):
        with mock.patch.dict(os.environ, {"HWIKI_NLI_DIR": "/tmp/some-nli"}):
            self.assertEqual(nli.nli_model_dir(), Path("/tmp/some-nli"))

    def test_default_lives_under_plugins_data(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HWIKI_NLI_DIR", None)
            path = nli.nli_model_dir(home=Path("/home/x"))
        self.assertEqual(path, Path("/home/x") / "plugins-data" / "hwiki-extractor"
                          / "models" / "nli-deberta-small")


class ScorerLoadingTests(unittest.TestCase):
    def test_missing_model_raises_rather_than_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(nli.NliUnavailable):
                nli.NliScorer(Path(tmp))

    def test_try_get_scorer_returns_none_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HWIKI_NLI_DIR": tmp}):
                nli._SESSIONS.clear()
                self.assertIsNone(nli.try_get_scorer())

    def test_classify_pair_raises_when_no_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HWIKI_NLI_DIR": tmp}):
                nli._SESSIONS.clear()
                with self.assertRaises(nli.NliUnavailable):
                    verdict.classify_pair("a", "b")


class FrameOverlapTests(unittest.TestCase):
    def test_identical_claims_overlap_fully(self):
        self.assertAlmostEqual(verdict.frame_overlap("Staging runs on Fly.io",
                                                     "Staging runs on Fly.io"), 1.0)

    def test_unrelated_claims_do_not_overlap(self):
        self.assertEqual(verdict.frame_overlap("The user takes notes in Obsidian",
                                              "Backups run nightly with restic"), 0.0)

    def test_stopwords_are_ignored(self):
        # "the/is/of/on" carry no frame, so these two share nothing.
        self.assertEqual(verdict.frame_overlap("The value is on the server",
                                              "The result is of the kind"), 0.0)

    def test_empty_input_is_zero_not_a_crash(self):
        self.assertEqual(verdict.frame_overlap("", "anything"), 0.0)


class ClassifyTests(unittest.TestCase):
    def test_high_contradiction_is_a_contradiction(self):
        result = verdict.classify_pair("Deploys allowed on Fridays",
                                       "Deploys never allowed on Fridays",
                                       _Stub(contradiction=1.0))
        self.assertEqual(result["relation"], "contradiction")

    def test_high_entailment_is_a_restatement(self):
        result = verdict.classify_pair("User prefers terse updates",
                                       "User prefers short updates",
                                       _Stub(entailment=1.0))
        self.assertEqual(result["relation"], "restatement")

    def test_neutral_is_compatible(self):
        result = verdict.classify_pair("Staging is on Fly.io",
                                       "Staging runs Postgres",
                                       _Stub(contradiction=0.30, neutral=0.70))
        self.assertEqual(result["relation"], "compatible")

    def test_below_threshold_contradiction_is_not_flagged(self):
        result = verdict.classify_pair("Staging is on Fly.io",
                                       "Staging runs Postgres",
                                       _Stub(contradiction=0.89, neutral=0.11))
        self.assertEqual(result["relation"], "compatible")

    def test_result_carries_the_probabilities_and_the_frame(self):
        result = verdict.classify_pair("Staging runs on Fly.io", "Staging runs on Hetzner",
                                       _Stub(contradiction=1.0))
        self.assertEqual(set(result), {"relation", "contradiction", "entailment",
                                       "neutral", "frame_overlap"})

    def test_flags_contradiction_is_false_without_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HWIKI_NLI_DIR": tmp}):
                nli._SESSIONS.clear()
                self.assertFalse(verdict.flags_contradiction("a", "b"))


class EnableFlagTests(unittest.TestCase):
    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(verdict.ENV_ENABLE, None)
            self.assertFalse(verdict.enabled())

    def test_truthy_values(self):
        for value in ("1", "true", "yes", "on", "TRUE"):
            with mock.patch.dict(os.environ, {verdict.ENV_ENABLE: value}):
                self.assertTrue(verdict.enabled(), value)

    def test_falsey_values(self):
        for value in ("0", "false", "no", "off", ""):
            with mock.patch.dict(os.environ, {verdict.ENV_ENABLE: value}):
                self.assertFalse(verdict.enabled(), value)


class AnnotationWiringTests(unittest.TestCase):
    """The NLI tier annotates candidates; it must never change `possible_conflict`."""

    def setUp(self):
        self.embedder = facts.Embedder(mode="hash")

    def test_default_path_is_untouched(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(verdict.ENV_ENABLE, None)
            related, conflict, _dup = facts.related_and_conflict(
                "Staging runs on Fly.io",
                [("k", "Staging runs on Hetzner")], self.embedder)
        self.assertIs(conflict, False)
        self.assertNotIn("relation", related[0])

    def test_enabled_annotation_appears_and_conflict_is_unchanged(self):
        stub = _Stub(contradiction=1.0)
        with mock.patch.dict(os.environ, {verdict.ENV_ENABLE: "1"}), \
                mock.patch.object(verdict, "try_get_scorer", return_value=stub):
            related, conflict, _dup = facts.related_and_conflict(
                "Staging runs on Fly.io",
                [("k", "Staging runs on Hetzner")], self.embedder)
        self.assertIs(conflict, False, "annotation must not set possible_conflict")
        self.assertEqual(related[0]["relation"], "contradiction")
        self.assertEqual(related[0]["nli"]["contradiction"], 1.0)

    def test_enabled_but_model_absent_degrades_silently(self):
        with mock.patch.dict(os.environ, {verdict.ENV_ENABLE: "1"}), \
                mock.patch.object(verdict, "try_get_scorer", return_value=None):
            related, conflict, _dup = facts.related_and_conflict(
                "Staging runs on Fly.io",
                [("k", "Staging runs on Hetzner")], self.embedder)
        self.assertIs(conflict, False)
        self.assertNotIn("relation", related[0])

    def test_a_broken_scorer_does_not_break_propose(self):
        class Boom:
            def probabilities(self, *_a, **_k):
                raise RuntimeError("model exploded")

        with mock.patch.dict(os.environ, {verdict.ENV_ENABLE: "1"}), \
                mock.patch.object(verdict, "try_get_scorer", return_value=Boom()):
            related, conflict, _dup = facts.related_and_conflict(
                "Staging runs on Fly.io",
                [("k", "Staging runs on Hetzner")], self.embedder)
        self.assertIs(conflict, False)
        self.assertTrue(related)

    def test_no_candidates_means_no_scorer_call(self):
        stub = _Stub(contradiction=1.0)
        with mock.patch.dict(os.environ, {verdict.ENV_ENABLE: "1"}), \
                mock.patch.object(verdict, "try_get_scorer", return_value=stub):
            related, conflict, _dup = facts.related_and_conflict(
                "Staging runs on Fly.io",
                [("k", "The user prefers aisle seats")], self.embedder)
        self.assertEqual(related, [])
        self.assertIs(conflict, False)
        self.assertEqual(stub.calls, [])


class RealModelTests(unittest.TestCase):
    """Runs only where the bundle exists; the fast suite must not require it."""

    def setUp(self):
        if not nli.nli_model_dir().is_dir():
            self.skipTest("NLI model bundle not installed")

    def test_entailment_beats_neutral_on_a_paraphrase(self):
        scorer = nli.get_scorer()
        con, ent, neu = scorer.probabilities("User prefers terse status updates",
                                            "User prefers short status updates")
        self.assertGreater(ent, 0.9)
        self.assertLess(con, 0.1)

    def test_negation_is_a_contradiction(self):
        scorer = nli.get_scorer()
        con, ent, _neu = scorer.probabilities("Deploys are allowed on Fridays",
                                             "Deploys are never allowed on Fridays")
        self.assertGreater(con, 0.9)

    def test_scorer_is_shared_per_model_dir(self):
        self.assertIs(nli.get_scorer(), nli.get_scorer())


if __name__ == "__main__":
    unittest.main(verbosity=2)
