"""Regression lock for the two conformance matrices.

`eval_llm_wiki.py` and `eval_capabilities.py` are measurements, not gates: they print a
matrix. Without a test, a later refactor can quietly turn a demonstrated behaviour into a
shrug and nothing fails. This file turns the matrices into assertions.

Two rules, both deliberately blunt:

* **No probe may crash.** A probe that raises records ``NO`` in the matrix, which looks
  identical to an honest "not implemented". That is the failure mode most likely to hide
  a regression, so it fails here.
* **A demonstrated behaviour stays demonstrated, and an honest gap stays documented.** If
  a `NO` row ever becomes `YES` (for example someone adds induction), this test fails and
  forces the comparison document to be updated rather than silently outgrown.

Runs in the cheap tier: the matrix must stay honest about what the default path does, and
hash mode is the default for anyone who has not downloaded a model.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _hwiki_path  # noqa: F401,E402
from hwiki.facts import Embedder  # noqa: E402

# Elements the LLM Wiki pattern requires of a working implementation.
PATTERN_MUST_HOLD = (
    "Persistent wiki artifact (not query-time RAG)",
    "Compiled once, kept current",
    "Ingest integrates into existing pages",
    "Contradictions flagged at ingest",
    "Agent maintains the wiki (no hand-editing)",
    "Raw sources immutable",
    "Wiki layer distinct from raw sources",
    "Schema document (conventions + workflows)",
    "index.md content catalog (link + summary, grouped)",
    "index.md updated on every write",
    "log.md chronological, grep-parseable",
    "Query with page citations (no absolute paths)",
    "Answers filed back as wiki pages",
    "Lint: duplicate claims across pages",
    "Lint: dangling cross-references",
    "Lint: catalog drift",
    "Lint is read-only (never repairs)",
    "YAML frontmatter (Dataview-ready)",
    "Interlinked pages (Obsidian wikilinks)",
    "Incremental compounding (pages < claims)",
    "Lint: orphan pages (no inbound links)",
)

# Honcho documented surfaces hwiki genuinely delivers.
HONCHO_MUST_HOLD = (
    "No external API at runtime",
    "Cross-session memory",
    "Evidence (what an answer was built from)",
    "Structured outputs (typed JSON)",
    "Deletion semantics",
    "Context with a token budget",
)

# Capabilities that deliberately do not exist without a generative model. If one of these
# becomes YES, this test fails on purpose: the comparison document must be updated.
DOCUMENTED_GAPS = (
    "Derived conclusions (deduction)",
    "Pattern inference (induction)",
    "Dialectic / reasoned answers",
    "Save interaction history",
    "Webhooks / push notifications",
    "Streaming responses",
)


def _run(module, probes, emb):
    for probe in probes:
        try:
            probe(emb)
        except Exception as exc:  # noqa: BLE001 - surfacing the crash is the point
            module.row(probe.__name__, "NO", f"probe raised {type(exc).__name__}: {exc}")
    return {element: verdict for element, verdict, _evidence in module.ROWS}


class ConformanceKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import eval_capabilities
        import eval_llm_wiki

        cls.pattern_mod = eval_llm_wiki
        cls.honcho_mod = eval_capabilities
        emb = Embedder(mode="hash")
        emb.embed("warm")
        cls.pattern = _run(eval_llm_wiki, (
            eval_llm_wiki.probe_persistent_artifact,
            eval_llm_wiki.probe_compiled_not_rederived,
            eval_llm_wiki.probe_ingest_integrates,
            eval_llm_wiki.probe_contradiction_at_ingest,
            eval_llm_wiki.probe_llm_maintains,
            eval_llm_wiki.probe_raw_immutable,
            eval_llm_wiki.probe_wiki_layer_distinct,
            eval_llm_wiki.probe_schema_document,
            eval_llm_wiki.probe_index_catalog,
            eval_llm_wiki.probe_index_updated_on_write,
            eval_llm_wiki.probe_log_parseable,
            eval_llm_wiki.probe_query_citations,
            eval_llm_wiki.probe_answers_filed_back,
            eval_llm_wiki.probe_lint_duplicates,
            eval_llm_wiki.probe_lint_dangling,
            eval_llm_wiki.probe_lint_catalog_drift,
            eval_llm_wiki.probe_lint_orphans,
            eval_llm_wiki.probe_lint_readonly,
            eval_llm_wiki.probe_frontmatter,
            eval_llm_wiki.probe_wikilinks,
            eval_llm_wiki.probe_compounding,
            eval_llm_wiki.probe_session_summaries,
            eval_llm_wiki.probe_prose_synthesis,
            eval_llm_wiki.probe_crossref_maintenance,
            eval_llm_wiki.probe_images,
        ), emb)
        cls.honcho = _run(eval_capabilities, (
            eval_capabilities.probe_cross_session,
            eval_capabilities.probe_derived_conclusions,
            eval_capabilities.probe_induction,
            eval_capabilities.probe_consolidation,
            eval_capabilities.probe_conflict_detection,
            eval_capabilities.probe_dialectic,
            eval_capabilities.probe_answerability,
            eval_capabilities.probe_lineage,
            eval_capabilities.probe_peers_scopes,
            eval_capabilities.probe_inspectability,
            eval_capabilities.probe_offline,
            eval_capabilities.probe_interaction_history,
            eval_capabilities.probe_prompt_context,
            eval_capabilities.probe_workspace_scope,
            eval_capabilities.probe_document_import,
            eval_capabilities.probe_background_processing,
            eval_capabilities.probe_dream_knowledge_update,
            eval_capabilities.probe_representation_history,
            eval_capabilities.probe_peer_card,
            eval_capabilities.probe_summaries,
            eval_capabilities.probe_evidence,
            eval_capabilities.probe_queue_status,
            eval_capabilities.probe_webhooks,
            eval_capabilities.probe_structured_outputs,
            eval_capabilities.probe_streaming,
            eval_capabilities.probe_directional,
            eval_capabilities.probe_observe_modes,
            eval_capabilities.probe_filters,
            eval_capabilities.probe_deletion,
            eval_capabilities.probe_context_budget,
            eval_capabilities.probe_interface_surface,
        ), emb)

    def test_no_probe_crashed(self):
        """A crashing probe records NO, which is indistinguishable from an honest gap."""
        pattern_rows = {e: ev for e, _v, ev in self.pattern_mod.ROWS}
        honcho_rows = {e: ev for e, _v, ev in self.honcho_mod.ROWS}
        for name, evidence in pattern_rows.items():
            self.assertNotIn("probe raised", str(evidence), f"pattern probe {name} crashed")
        for name, evidence in honcho_rows.items():
            self.assertNotIn("probe raised", str(evidence), f"honcho probe {name} crashed")

    def test_pattern_elements_still_hold(self):
        for element in PATTERN_MUST_HOLD:
            self.assertEqual(self.pattern.get(element), "YES",
                             f"LLM Wiki element regressed: {element}")

    def test_honcho_surfaces_still_hold(self):
        for element in HONCHO_MUST_HOLD:
            self.assertEqual(self.honcho.get(element), "YES",
                             f"Honcho surface regressed: {element}")

    def test_generation_gaps_are_still_documented_gaps(self):
        for element in DOCUMENTED_GAPS:
            self.assertEqual(self.honcho.get(element), "NO",
                             f"{element} is now YES — update benchmarks/conformance-2026-11.md")
        # the pattern's remaining honest NOs
        self.assertEqual(self.pattern.get("Per-source summary pages"), "NO")
        self.assertEqual(self.pattern.get("Evolving prose synthesis"), "NO")


if __name__ == "__main__":
    unittest.main(verbosity=2)
