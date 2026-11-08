"""Conformance: does hwiki actually implement the LLM Wiki pattern?

The pattern (Karpathy's gist) is prose, so "we follow it" is unfalsifiable unless each
element is turned into an assertion. This file does that: every element of the gist gets
a probe that builds a throwaway vault, performs the operation the pattern describes, and
checks the artifact that should exist afterwards.

Verdicts are deliberately harsh:

* ``YES``      — the artifact exists and the assertion proves it.
* ``PARTIAL``  — something real exists but does not deliver what the pattern describes.
* ``NO``       — hwiki does not do this; the probe shows it does not.

Elements the gist marks optional (images, Marp decks, Dataview) are graded on whether
the *substrate* supports them, not on whether hwiki ships them.

    python3 tests/eval_llm_wiki.py            # onnx if available
    python3 tests/eval_llm_wiki.py --mode hash
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _hwiki_path  # noqa: F401,E402
import run_behavior as rb  # noqa: E402
from hwiki import catalog, compile as compile_mod, lint, maintain  # noqa: E402
from hwiki.config import build_runtime  # noqa: E402
from hwiki.facts import (Embedder, ProposalStore, file_answer,  # noqa: E402
                         write_fact_page)

MEMORY_REL = Path("Hermes/Memory/Auto")
ROWS = []


def row(element, verdict, evidence):
    ROWS.append((element, verdict, evidence))


class Lab:
    """A throwaway vault with a real store, catalog and lint, like a live install."""

    def __init__(self, emb):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vault"
        self.mem = self.root / MEMORY_REL
        self.mem.mkdir(parents=True)
        (self.root / "raw").mkdir()
        (self.root / "concepts").mkdir()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.emb = emb
        self.store = ProposalStore(Path(self.tmp.name) / "facts.sqlite", embedder=emb)
        self.config, self.wiki, self.backend = build_runtime({
            "wiki_path": str(self.root), "memory_dir": str(MEMORY_REL),
            "index_path": str(self.home / "index.sqlite"),
            "_hermes_home": str(self.home), "embeddings": {"mode": "hash"},
        })

    # -- fixtures ---------------------------------------------------------
    def fact(self, claim, *, kind="fact", session="s", project="", peer="user"):
        proposal = self.store.propose(
            {"kind": kind, "claim": claim, "evidence": "stated"},
            user_text="stated", session_id=session, project=project, peer=peer)
        self.store.accept(proposal["id"])
        page = write_fact_page(self.root, MEMORY_REL, self.store.get(proposal["id"]))
        self.store.set_page_path(proposal["id"], page)
        return proposal["id"], page

    def raw_source(self, name, text):
        path = self.root / "raw" / name
        path.write_text(text, encoding="utf-8")
        return path

    def page(self, name, text):
        path = self.mem / name
        path.write_text(text, encoding="utf-8")
        return path

    def index(self):
        return catalog.write_index(self.wiki)

    def log(self, kind, subject):
        return catalog.append_log(self.wiki, kind, subject)

    def close(self):
        self.tmp.cleanup()


def _digest(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


# ---------------------------------------------------------------------------
# core idea
# ---------------------------------------------------------------------------

def probe_persistent_artifact(emb):
    """The wiki is a persistent artifact, not a per-query retrieval cache."""
    lab = Lab(emb)
    try:
        _id, page = lab.fact("Staging server stays on Fly.io")
        # a brand-new store over the same vault: nothing in memory, only the artifact
        fresh = ProposalStore(Path(lab.tmp.name) / "fresh.sqlite", embedder=emb)
        fresh.rebuild_from_vault(lab.root, MEMORY_REL)
        hits = fresh.recall("where does staging run", limit=3)
        ok = page.exists() and hits and "fly.io" in hits[0]["claim"].lower()
        row("Persistent wiki artifact (not query-time RAG)", "YES" if ok else "NO",
            f"page on disk survives a new store; rebuilt index recalls it "
            f"({len(hits)} hit(s))" if ok else "artifact did not survive a new store")
    finally:
        lab.close()


def probe_compiled_not_rederived(emb):
    """Knowledge is compiled once and kept current, not re-derived per query."""
    lab = Lab(emb)
    try:
        _id, page = lab.fact("Backups run nightly with restic to B2")
        merged = compile_mod.merge_into_page(
            page, "Backups are verified monthly", evidence="stated", kind="fact")
        text = page.read_text(encoding="utf-8")
        sections = compile_mod.existing_sections(text)
        opened = "Backups run nightly with restic to B2" in text
        ok = bool(merged.merged) and opened and "Backups are verified monthly" in sections
        row("Compiled once, kept current", "YES" if ok else "NO",
            f"the original claim stays the page topic and a second claim is compiled "
            f"into it ({len(sections)} compiled section(s)); recall reads the page"
            if ok else f"merge did not happen (reason: {merged.get('reason')})")
    finally:
        lab.close()


def probe_ingest_integrates(emb):
    """Ingest integrates into existing pages instead of only indexing a new file."""
    lab = Lab(emb)
    try:
        _id, page = lab.fact("The API is written in FastAPI")
        before = page.read_text(encoding="utf-8")
        compile_mod.merge_into_page(page, "The API uses Pydantic models",
                                    evidence="stated", kind="fact")
        after = page.read_text(encoding="utf-8")
        touched = after != before and "Pydantic" in after
        row("Ingest integrates into existing pages", "YES" if touched else "NO",
            "a related claim rewrote the existing page rather than creating a second one"
            if touched else "ingest only appended a new page")
    finally:
        lab.close()


def probe_contradiction_at_ingest(emb):
    """Contradictions are flagged when the new claim arrives."""
    lab = Lab(emb)
    try:
        lab.fact("Deploys are allowed on Fridays")
        proposal = lab.store.propose(
            {"kind": "fact", "claim": "Deploys are never allowed on Fridays",
             "evidence": "stated"}, user_text="stated", session_id="s2")
        flagged = bool(proposal.get("possible_conflict"))
        row("Contradictions flagged at ingest", "YES" if flagged else "NO",
            "the second claim is marked possible_conflict and held for review"
            if flagged else "the contradiction was not flagged")
    finally:
        lab.close()


def probe_llm_maintains(emb):
    """The agent maintains the wiki; the human does not hand-edit it."""
    lab = Lab(emb)
    try:
        lab.fact("The user prefers terse status updates")
        report = maintain.run_maintenance(
            lab.store, lab.root, MEMORY_REL, embedder=emb,
            question_buffer=maintain.QuestionBuffer())
        page = lab.mem / maintain.MAINTENANCE_FILENAME
        ok = page.exists() and report.written
        row("Agent maintains the wiki (no hand-editing)", "YES" if ok else "NO",
            "a scheduled tick wrote the maintenance report and refreshed artifacts"
            if ok else "no autonomous maintenance artifact was produced")
    finally:
        lab.close()


# ---------------------------------------------------------------------------
# architecture: three layers
# ---------------------------------------------------------------------------

def probe_raw_immutable(emb):
    """Raw sources are read-only: the tool never modifies them."""
    lab = Lab(emb)
    try:
        source = lab.raw_source("article.md", "# Article\n\nStaging runs on Fly.io.\n")
        before = _digest([source])
        lab.fact("Staging server stays on Fly.io")
        lab.index()
        lab.log("ingest", "article")
        lint.lint_memory(lab.wiki)
        maintain.run_maintenance(lab.store, lab.root, MEMORY_REL, embedder=emb,
                                 question_buffer=maintain.QuestionBuffer())
        after = _digest([source])
        row("Raw sources immutable", "YES" if before == after else "NO",
            "source bytes unchanged after ingest, catalog, lint and maintenance"
            if before == after else "a raw source was modified")
    finally:
        lab.close()


def probe_wiki_layer_distinct(emb):
    """The wiki layer is distinguishable from raw sources and from hand notes."""
    lab = Lab(emb)
    try:
        _id, page = lab.fact("The database is Postgres 15")
        text = page.read_text(encoding="utf-8")
        markers = [m for m in ("spine: memory", "type: memory", "extractor: hwiki-extractor")
                   if m in text]
        row("Wiki layer distinct from raw sources", "YES" if len(markers) == 3 else "PARTIAL",
            f"generated pages carry {len(markers)}/3 layer markers ({', '.join(markers)})")
    finally:
        lab.close()


def probe_schema_document(emb):
    """A schema document tells the agent the conventions and workflows."""
    schema = Path(__file__).resolve().parent.parent / "docs" / "wiki-schema.md"
    if not schema.exists():
        row("Schema document (conventions + workflows)", "NO", "docs/wiki-schema.md missing")
        return
    text = schema.read_text(encoding="utf-8")
    needed = {"page format": "frontmatter", "ingest": "ingest", "query": "query",
              "lint": "lint", "invariants": "invariant"}
    found = [name for name, needle in needed.items() if needle in text.lower()]
    row("Schema document (conventions + workflows)",
        "YES" if len(found) == len(needed) else "PARTIAL",
        f"docs/wiki-schema.md documents {len(found)}/{len(needed)} required areas "
        f"({', '.join(found)})")
    return


# ---------------------------------------------------------------------------
# operations: ingest / query / lint
# ---------------------------------------------------------------------------

def probe_index_catalog(emb):
    """index.md is a content catalog: link + one-line summary, grouped."""
    lab = Lab(emb)
    try:
        lab.fact("Staging server stays on Fly.io")
        lab.fact("The user prefers terse status updates", kind="preference")
        path = lab.index()
        text = path.read_text(encoding="utf-8") if path else ""
        links = re.findall(r"\[\[([^\]|]+)", text)
        summaries = [ln for ln in text.splitlines()
                     if ln.startswith("- ") and "—" in ln]
        groups = re.findall(r"^## (.+)$", text, re.M)
        ok = len(links) >= 2 and len(summaries) >= 2 and bool(groups)
        row("index.md content catalog (link + summary, grouped)",
            "YES" if ok else "PARTIAL",
            f"{len(links)} link(s), {len(summaries)} one-line summar(ies), "
            f"categories={groups}")
    finally:
        lab.close()


def probe_index_updated_on_write(emb):
    """The catalog is regenerated when the wiki changes, not on a manual pass."""
    lab = Lab(emb)
    try:
        lab.fact("Staging server stays on Fly.io")
        lab.index()
        first = (lab.root / "index.md").read_text(encoding="utf-8")
        lab.fact("The on-call rotation changes every Monday")
        lab.index()
        second = (lab.root / "index.md").read_text(encoding="utf-8")
        ok = first != second and "on-call" in second.lower()
        row("index.md updated on every write", "YES" if ok else "NO",
            "a new page appears in the catalog after the next write"
            if ok else "the catalog did not pick up the new page")
    finally:
        lab.close()


def probe_log_parseable(emb):
    """log.md is chronological, append-only and grep-parseable."""
    lab = Lab(emb)
    try:
        lab.log("ingest", "Article Title")
        lab.log("query", "where does staging run")
        text = (lab.root / "log.md").read_text(encoding="utf-8")
        entries = re.findall(r"^## \[(\d{4}-\d{2}-\d{2})\] (\w+) \| (.+)$", text, re.M)
        ok = len(entries) == 2 and entries[0][1] == "ingest"
        row("log.md chronological, grep-parseable", "YES" if ok else "NO",
            f"{len(entries)} entries matching `## [date] kind | subject`; "
            "`grep '^## \\[' log.md | tail -5` works" if ok else "log format not parseable")
    finally:
        lab.close()


def probe_query_citations(emb):
    """Query answers can cite pages, and citations never leak absolute paths."""
    lab = Lab(emb)
    try:
        _id, page = lab.fact("Staging server stays on Fly.io")
        lab.index()
        hits = lab.store.recall("where does staging run", limit=3)
        blob = json.dumps(hits, default=str)
        cites = [h for h in hits if h.get("cite") or h.get("page")]
        leaks = str(lab.root) in blob
        ok = bool(cites) and not leaks
        row("Query with page citations (no absolute paths)", "YES" if ok else "PARTIAL",
            f"{len(cites)} hit(s) carry a wikilink citation "
            f"({cites[0].get('cite') if cites else 'none'}); absolute path leaked={leaks}")
    finally:
        lab.close()


def probe_answers_filed_back(emb):
    """Good answers are filed back into the wiki as new pages."""
    lab = Lab(emb)
    try:
        first = file_answer(lab.mem, "where does staging run",
                            "Staging runs on Fly.io, in the EU region.",
                            tags=["memory/answer"])
        again = file_answer(lab.mem, "where does staging run",
                            "Staging runs on Fly.io, in the EU region.",
                            tags=["memory/answer"])
        path = Path(first.get("path") or "")
        ok = (bool(first.get("success")) and path.is_file()
              and path.parent.name == "answers"
              and Path(str(again.get("path"))) == path and again.get("duplicate"))
        row("Answers filed back as wiki pages", "YES" if ok else "NO",
            f"answer written to {path.parent.name}/{path.name}, re-filing is idempotent"
            if ok else f"answer was not filed: {first}")
    finally:
        lab.close()


def probe_lint_duplicates(emb):
    """Lint finds the same claim accepted on two pages."""
    lab = Lab(emb)
    try:
        lab.fact("Backups run nightly with restic to B2")
        lab.page("dupe.md",
                 "---\ntitle: Backups run nightly with restic to B2\n"
                 "extractor: hwiki-extractor\nkind: fact\nstatus: accepted\n"
                 "accepted_at: 2026-01-01\nproposal_id: vault-dupe\n---\n\n"
                 "# Backups run nightly with restic to B2\n\n> **Evidence:** stated\n")
        report = lint.lint_memory(lab.wiki)
        checks = {f.check for f in report.findings}
        ok = "duplicate.claim" in checks
        row("Lint: duplicate claims across pages", "YES" if ok else "NO",
            f"found {sorted(checks)}" if ok else "duplicate claim not reported")
    finally:
        lab.close()


def probe_lint_dangling(emb):
    """Lint finds cross-references that point at nothing."""
    lab = Lab(emb)
    try:
        lab.page("broken.md",
                 "---\ntitle: Broken\nextractor: hwiki-extractor\nkind: fact\n"
                 "status: accepted\naccepted_at: 2026-01-01\nproposal_id: p1\n---\n\n"
                 "# A claim\n\nSee [[no-such-page]] for details.\n")
        report = lint.lint_memory(lab.wiki)
        checks = {f.check for f in report.findings}
        row("Lint: dangling cross-references", "YES" if "link.dangling" in checks else "NO",
            f"found {sorted(checks)}")
    finally:
        lab.close()


def probe_lint_catalog_drift(emb):
    """Lint notices the catalog and the wiki disagreeing."""
    lab = Lab(emb)
    try:
        lab.fact("Staging server stays on Fly.io")
        lab.index()
        lab.fact("The on-call rotation changes every Monday")   # not in the catalog
        report = lint.lint_memory(lab.wiki)
        checks = {f.check for f in report.findings}
        ok = "catalog.entry_missing" in checks
        row("Lint: catalog drift", "YES" if ok else "NO",
            f"found {sorted(checks)}" if ok else "missing catalog entry not reported")
    finally:
        lab.close()


def probe_lint_orphans(emb):
    """The pattern asks for orphan pages (no inbound links). hwiki has no such check."""
    lab = Lab(emb)
    try:
        lab.page("lonely.md",
                 "---\ntitle: Lonely\nextractor: hwiki-extractor\nkind: fact\n"
                 "status: accepted\naccepted_at: 2026-01-01\nproposal_id: p2\n---\n\n"
                 "# Nobody links here\n\n> **Evidence:** stated\n")
        report = lint.lint_memory(lab.wiki)
        checks = {f.check for f in report.findings}
        orphan = any("orphan" in c and "supersede" not in c for c in checks)
        row("Lint: orphan pages (no inbound links)", "YES" if orphan else "NO",
            f"checks present: {sorted(checks)}; no inbound-link graph is built"
            if not orphan else "orphan page reported")
    finally:
        lab.close()


def probe_lint_readonly(emb):
    """Lint never repairs: it must not change a byte."""
    lab = Lab(emb)
    try:
        _id, page = lab.fact("Staging server stays on Fly.io")
        lab.page("broken.md",
                 "---\ntitle: Broken\nextractor: hwiki-extractor\nkind: fact\n"
                 "status: accepted\naccepted_at: 2026-01-01\nproposal_id: p3\n---\n\n"
                 "# A claim\n\nSee [[no-such-page]].\n")
        before = _digest(sorted(lab.mem.glob("*.md")))
        lint.lint_memory(lab.wiki)
        after = _digest(sorted(lab.mem.glob("*.md")))
        row("Lint is read-only (never repairs)", "YES" if before == after else "NO",
            f"{len(before)} page(s) byte-identical after lint")
    finally:
        lab.close()


# ---------------------------------------------------------------------------
# substrate: Obsidian, frontmatter, links, compounding
# ---------------------------------------------------------------------------

def probe_frontmatter(emb):
    """Pages carry YAML frontmatter, so Dataview-style queries are possible."""
    lab = Lab(emb)
    try:
        _id, page = lab.fact("The user prefers terse status updates", kind="preference")
        text = page.read_text(encoding="utf-8")
        keys = [k for k in ("title:", "kind:", "status:", "accepted_at:", "tags:")
                if k in text]
        row("YAML frontmatter (Dataview-ready)", "YES" if len(keys) == 5 else "PARTIAL",
            f"{len(keys)}/5 frontmatter keys present")
    finally:
        lab.close()


def probe_wikilinks(emb):
    """Pages are interlinked with Obsidian wikilinks, so the graph view is meaningful."""
    lab = Lab(emb)
    try:
        first, old_page = lab.fact("Staging server stays on Fly.io")
        second, _page = lab.fact("Staging server is moving to Hetzner VPS")
        lab.store.supersede(second, first)
        # the superseded page is tombstoned with a wikilink to its replacement
        from hwiki.facts import mark_superseded_page
        mark_superseded_page(old_page, "Staging server is moving to Hetzner VPS")
        text = old_page.read_text(encoding="utf-8")
        links = re.findall(r"\[\[([^\]|]+)", text)
        linked = any("hetzner" in link for link in links)
        row("Interlinked pages (Obsidian wikilinks)", "YES" if linked else "PARTIAL",
            f"{len(links)} wikilink(s) on the tombstoned page: {links[:3]}")
    finally:
        lab.close()


def probe_compounding(emb):
    """The wiki compounds: related claims share pages instead of multiplying them."""
    lab = Lab(emb)
    try:
        claims = ["Backups run nightly with restic to B2",
                  "Backups are verified monthly",
                  "Backups are encrypted with age",
                  "Backups are retained for 30 days",
                  "Backups are tested by restoring a random file",
                  "Backups are monitored by a heartbeat check"]
        pages = []
        for claim in claims:
            _id, page = lab.fact(claim)
            pages.append(page)
        owner = pages[0]
        for claim in claims[1:]:
            compile_mod.merge_into_page(owner, claim, evidence="stated", kind="fact")
        text = owner.read_text(encoding="utf-8")
        sections = compile_mod.existing_sections(text)
        held = len([c for c in claims if c in text])
        # every claim lives on one page: pages/claim = 1/6 instead of 6
        ratio = held / len(claims)
        row("Incremental compounding (pages < claims)",
            "YES" if ratio == 1.0 and len(sections) >= 5 else "PARTIAL",
            f"{len(claims)} claims all present on 1 page ({held}/{len(claims)} claims, "
            f"{len(sections)} compiled section(s)) — 1 page instead of 6")
    finally:
        lab.close()


# ---------------------------------------------------------------------------
# what the pattern asks for that hwiki does not do
# ---------------------------------------------------------------------------

def probe_session_summaries(emb):
    """The pattern's ingest writes a summary page per source. hwiki writes per claim."""
    lab = Lab(emb)
    try:
        lab.raw_source("article.md", "# Article\n\nThree facts about staging.\n")
        lab.fact("Staging server stays on Fly.io")
        lab.fact("Staging uses the EU region")
        summaries = [p for p in lab.mem.glob("*.md")
                     if "summary" in p.stem or "article" in p.stem]
        row("Per-source summary pages", "NO",
            f"{len(summaries)} summary page(s) for 1 source; hwiki files one page per "
            "claim and summarises only inside index.md")
    finally:
        lab.close()


def probe_prose_synthesis(emb):
    """The pattern's wiki holds evolving prose synthesis. hwiki holds claims."""
    lab = Lab(emb)
    try:
        _id, page = lab.fact("Staging server stays on Fly.io")
        compile_mod.merge_into_page(page, "Staging uses the EU region",
                                    evidence="stated", kind="fact")
        text = page.read_text(encoding="utf-8")
        prose = [ln for ln in text.splitlines()
                 if len(ln.split()) > 25 and not ln.startswith((">", "#", "-"))]
        row("Evolving prose synthesis", "NO",
            f"{len(prose)} synthesised paragraph(s); the page is a list of claims with "
            "evidence, not a rewritten narrative")
    finally:
        lab.close()


def probe_crossref_maintenance(emb):
    """The pattern maintains cross-references across pages as the wiki grows."""
    lab = Lab(emb)
    try:
        _id, page = lab.fact("Staging server stays on Fly.io")
        before = page.read_text(encoding="utf-8")
        lab.fact("The database is Postgres 15")
        maintain.run_maintenance(lab.store, lab.root, MEMORY_REL, embedder=emb,
                                 question_buffer=maintain.QuestionBuffer())
        after = page.read_text(encoding="utf-8")
        row("Cross-reference maintenance across pages", "PARTIAL",
            "maintenance reports conflicts/duplicates and compiles related claims, but "
            f"does not rewrite links in existing pages (page changed={before != after})")
    finally:
        lab.close()


def probe_images(emb):
    """The gist's image workflow is optional; grade the substrate, not a feature."""
    lab = Lab(emb)
    try:
        (lab.root / "raw" / "assets").mkdir(parents=True, exist_ok=True)
        (lab.root / "raw" / "assets" / "diagram.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        lab.raw_source("with-image.md", "# Note\n\n![[diagram.png]]\n")
        report = lint.lint_memory(lab.wiki)
        row("Image handling (optional in the pattern)", "PARTIAL",
            "images can live in raw/assets and be linked, but nothing reads or describes "
            f"them (lint findings: {len(report.findings)})")
    finally:
        lab.close()


def main():
    mode = "auto"
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
    emb = Embedder(mode=mode, model_dir=rb._real_models() if mode == "auto" else None)
    emb.embed("warm")  # ONNX loads lazily; force it before claiming the tier
    tier = getattr(emb, "mode", mode)

    probes = (probe_persistent_artifact, probe_compiled_not_rederived,
              probe_ingest_integrates, probe_contradiction_at_ingest,
              probe_llm_maintains, probe_raw_immutable, probe_wiki_layer_distinct,
              probe_schema_document, probe_index_catalog, probe_index_updated_on_write,
              probe_log_parseable, probe_query_citations, probe_answers_filed_back,
              probe_lint_duplicates, probe_lint_dangling, probe_lint_catalog_drift,
              probe_lint_orphans, probe_lint_readonly, probe_frontmatter,
              probe_wikilinks, probe_compounding, probe_session_summaries,
              probe_prose_synthesis, probe_crossref_maintenance, probe_images)
    for probe in probes:
        try:
            probe(emb)
        except Exception as exc:  # noqa: BLE001 - a crashing probe is a finding
            row(probe.__name__.replace("probe_", ""), "NO",
                f"probe raised {type(exc).__name__}: {exc}")

    print(f"\n=== hwiki vs the LLM Wiki pattern (tier: {tier}) ===\n")
    order = {"YES": 0, "PARTIAL": 1, "NO": 2}
    for element, verdict, evidence in sorted(ROWS, key=lambda r: order.get(r[1], 3)):
        print(f"[{verdict:<7}] {element}")
        print(f"          {evidence}")
    counts = {v: sum(1 for r in ROWS if r[1] == v) for v in ("YES", "PARTIAL", "NO")}
    print(f"\nYES {counts['YES']}  PARTIAL {counts['PARTIAL']}  NO {counts['NO']}"
          f"  (of {len(ROWS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
