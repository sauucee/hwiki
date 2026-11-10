"""The maintenance tick must judge the catalog against the *configured* page set.

`_wiki_for_artifacts` used to build a `WikiStore` on its constructor defaults
(`exclude=()`), so the artifact sweep saw pages the live provider deliberately hides.
`catalog.entry_missing` then demanded `index.md` list excluded pages, which the
catalog is configured never to list — a stale-catalog finding that no regeneration
could clear, reported on every tick forever.

This pins the boundary: an excluded page must not make the catalog look stale, and a
genuinely unlisted page still must.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import _hwiki_path  # noqa: F401,E402  (registers the repo root as package `hwiki`)

from hwiki.catalog import write_index  # noqa: E402
from hwiki.maintain import _artifact_findings  # noqa: E402
from hwiki.store import WikiStore  # noqa: E402

MEMORY = "Hermes/Memory/Auto"
CONFIG = {
    "include": ["Hermes/**/*.md"],
    "exclude": ["Hermes/Templates/**", "Hermes/Inbox.md"],
    "untrusted_dirs": ["raw"],
}


def _vault() -> Path:
    root = Path(tempfile.mkdtemp(prefix="vault-artifacts-"))
    (root / "Hermes" / "Templates").mkdir(parents=True)
    (root / "Hermes" / "Templates" / "Handoff Template.md").write_text(
        "---\ntitle: Handoff Template\n---\n\n# Handoff\n\nbody\n", encoding="utf-8")
    (root / "Hermes" / "Inbox.md").write_text(
        "---\ntitle: Inbox\n---\n\n# Inbox\n\nscratch\n", encoding="utf-8")
    (root / MEMORY).mkdir(parents=True)
    (root / "Hermes" / "Home.md").write_text(
        "---\ntitle: Home\n---\n\n# Home\n\nbody\n", encoding="utf-8")
    return root


def _store(root: Path) -> WikiStore:
    return WikiStore(root, include=CONFIG["include"], exclude=CONFIG["exclude"],
                     untrusted_dirs=CONFIG["untrusted_dirs"], memory_dir=MEMORY)


def test_excluded_pages_do_not_make_the_catalog_stale():
    root = _vault()
    store = _store(root)
    write_index(store, memory_dir=MEMORY)

    findings = _artifact_findings(store, root, MEMORY, config=CONFIG)
    joined = " ".join(findings)
    assert "Handoff Template" not in joined, findings
    assert "Inbox" not in joined, findings


def test_a_genuinely_unlisted_page_still_reports_missing():
    root = _vault()
    store = _store(root)
    write_index(store, memory_dir=MEMORY)
    # Added *after* the catalog was written, and not excluded.
    (root / "Hermes" / "Unlisted.md").write_text(
        "---\ntitle: Unlisted\n---\n\n# Unlisted\n\nbody\n", encoding="utf-8")

    findings = _artifact_findings(store, root, MEMORY, config=CONFIG)
    assert any("catalog.entry_missing" in f and "Unlisted" in f for f in findings), findings


def test_the_report_cannot_create_its_own_dangling_links():
    """A finding that quotes `[[Page]]` must not re-emit a live wikilink.

    The report body is a page in the memory zone, so anything it links is a real
    link. Findings quote the wikilink a reader should fix; written back unescaped,
    one run's report produced the next run's `link.dangling` errors.
    """
    from hwiki.maintain import MaintenanceReport, render

    report = MaintenanceReport()
    report.stale_artifacts = ["catalog.stale: index.md — lists [[Inbox]] but nothing resolves it"]
    report.questions = ["is [[Handoff Template]] still used?"]

    text = render(report, memory_dir=None)
    assert "[[" not in text.replace("\\[\\[", ""), text
    assert "\\[\\[Inbox\\]\\]" in text, text
