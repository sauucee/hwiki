"""Deterministic wiki lint for the memory tree (LLM Wiki "lint" step).

Read-only by construction: this module never writes, never repairs and never
touches the store. It reports findings so a human (or the review flow) can decide.
That matters because the LLM Wiki pattern's lint step is where a wiki rots
silently otherwise — a page can lose its provenance, its supersession link or its
catalog entry and still be recalled as if nothing happened.

Design rules:

* **Never invent provenance.** A missing or unparseable field is reported as
  unknown, not back-filled from the page title or the file name.
* **Deterministic.** Ordered checks, stable ordering, no clock and no randomness,
  so two runs over the same vault produce byte-identical output.
* **Advisory, not authoritative.** Severities order attention; nothing here
  decides that a fact is wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ERROR = "error"
WARN = "warn"
INFO = "info"

_SEVERITY_ORDER = {ERROR: 0, WARN: 1, INFO: 2}

# A memory page is written by the provider and carries this marker. Pages without
# it are hand-written notes living in the memory zone: not our business to lint
# for schema, but still checked for secret material.
EXTRACTOR_MARKER = "hwiki-extractor"

REQUIRED_FIELDS = ("kind", "status")
MAX_PAGE_CHARS = 24_000

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_H1_RE = re.compile(r"^# (.+)$", re.M)
_LINK_RE = re.compile(r"\[\[([^\]|#]+)")


@dataclass(frozen=True)
class Finding:
    """One lint result. ``page`` is wiki-root-relative; ``detail`` is for humans."""

    check: str
    severity: str
    page: str
    detail: str

    @property
    def line(self) -> str:
        return f"[{self.severity}] {self.check}: {self.page} — {self.detail}"


@dataclass
class LintReport:
    findings: List[Finding] = field(default_factory=list)
    pages_checked: int = 0

    def add(self, check: str, severity: str, page: str, detail: str) -> None:
        self.findings.append(Finding(check, severity, page, detail))

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == WARN]

    def sorted_findings(self) -> List[Finding]:
        return sorted(self.findings,
                      key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), f.check, f.page))

    def counts(self) -> Dict[str, int]:
        out = {ERROR: 0, WARN: 0, INFO: 0}
        for finding in self.findings:
            out[finding.severity] = out.get(finding.severity, 0) + 1
        return out

    @property
    def ok(self) -> bool:
        """True when nothing needs action. Warnings and notes are still 'ok'."""
        return not self.errors


def _frontmatter(text: str) -> Dict[str, str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    meta: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta


def _wikilink_targets(text: str) -> List[str]:
    seen: List[str] = []
    for raw in _LINK_RE.findall(text):
        target = raw.strip()
        if target and target not in seen:
            seen.append(target)
    return seen


def _memory_pages(memory_dir: Path) -> List[Path]:
    if not memory_dir.is_dir():
        return []
    return sorted(p for p in memory_dir.rglob("*.md") if p.is_file())


def lint_memory(store, *, memory_dir: Optional[Path] = None,
                max_page_chars: int = MAX_PAGE_CHARS) -> LintReport:
    """Lint the memory zone of ``store``'s vault. Never writes.

    ``store`` is a :class:`hwiki.store.WikiStore`; only its read API is used.
    """
    from .store import scan_for_secrets  # local import keeps this module leaf-only

    report = LintReport()
    root = Path(getattr(store, "root", store))
    mem = Path(memory_dir) if memory_dir is not None else (root / store.memory_dir)
    pages = _memory_pages(mem)
    report.pages_checked = len(pages)
    if not pages:
        return report

    # Known wikilink targets: every page slug in the vault plus every title, so a
    # link to a page that exists but was never indexed is not called dangling.
    known = set()
    try:
        for page in store.iter_pages():
            known.add(page.slug.lower())
            if page.title:
                known.add(page.title.strip().lower())
    except Exception:  # pragma: no cover - a broken index must not stop lint
        known = set()

    rel_root = root
    accepted_claims: Dict[str, List[str]] = {}

    for path in pages:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            report.add("page.unreadable", ERROR, str(path), "could not read file")
            continue
        try:
            rel = path.relative_to(rel_root).as_posix()
        except ValueError:
            rel = path.name

        # Secrets are checked everywhere: hand-written notes in the memory zone
        # are the most likely place for one to land by accident.
        secrets = scan_for_secrets(text)
        if secrets:
            kinds = ", ".join(sorted({s.split(":", 1)[0] for s in secrets}))
            report.add("secret.content", ERROR, rel,
                       f"secret-shaped material present ({kinds}); never promote this page")

        meta = _frontmatter(text)
        is_extractor_page = meta.get("extractor") == EXTRACTOR_MARKER
        if not is_extractor_page:
            # Hand-written page: no provider schema to enforce.
            continue

        for required in REQUIRED_FIELDS:
            if not meta.get(required):
                report.add("frontmatter.incomplete", WARN, rel,
                           f"`{required}` missing; provenance is unknown, not guessed")
        if not meta.get("accepted_at"):
            report.add("frontmatter.incomplete", WARN, rel,
                       "`accepted_at` missing; the fact has no acceptance date")
        if not meta.get("proposal_id"):
            report.add("frontmatter.incomplete", INFO, rel,
                       "`proposal_id` missing; rebuild will derive a stable id")

        if not _H1_RE.search(text):
            report.add("claim.missing", ERROR, rel,
                       "no `# claim` heading; the page states no fact")

        status = (meta.get("status") or "").strip().lower()
        if status == "superseded":
            target = meta.get("superseded_by") or ""
            links = _wikilink_targets(target)
            if not links:
                report.add("supersede.incomplete", WARN, rel,
                           "status is superseded but `superseded_by` names no page")
            else:
                for link in links:
                    if known and link.lower() not in known:
                        report.add("supersede.orphan", WARN, rel,
                                   f"superseded_by points at missing page [[{link}]]")

        h1 = _H1_RE.search(text)
        if status in ("", "accepted", "auto") and h1:
            key = " ".join(h1.group(1).split()).lower()
            accepted_claims.setdefault(key, []).append(rel)

        if len(text) > max_page_chars:
            report.add("page.oversized", INFO, rel,
                       f"{len(text)} chars exceeds {max_page_chars}; consider splitting")

        for link in _wikilink_targets(text):
            if known and link.lower() not in known:
                report.add("link.dangling", ERROR, rel,
                           f"[[{link}]] does not resolve to any page")

    for claim, owners in sorted(accepted_claims.items()):
        if len(owners) > 1:
            report.add("duplicate.claim", WARN, owners[0],
                       f"same claim accepted on {len(owners)} pages: {', '.join(owners)}")

    _lint_catalog(report, root, pages, rel_root)
    _lint_orphans(report, root, pages, rel_root)
    _lint_answers(report, mem, rel_root, pages)
    return report


def _lint_orphans(report: LintReport, root: Path, pages: Iterable[Path],
                  rel_root: Path) -> None:
    """Pages that nothing navigates to.

    The pattern calls these orphans: no inbound links. In this wiki navigation is the
    catalog ("read index.md first"), so a catalog entry is an inbound route and counts.
    A page with neither a link nor an entry is reachable only by luck.

    Deliberately INFO, and deliberately quiet for generated artifacts: the maintenance
    report, catalogs and filed answers are entry points by design, and flagging them every
    run would train the reader to ignore the check.
    """
    pages = list(pages)
    inbound = {}
    for path in pages:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for target in _wikilink_targets(text):
            inbound.setdefault(target.strip().lower(), set()).add(path.stem)

    listed = set()
    index = root / "index.md"
    if index.exists():
        try:
            listed = {t.strip().lower() for t in
                      _wikilink_targets(index.read_text(encoding="utf-8", errors="replace"))}
        except OSError:
            listed = set()

    for path in pages:
        slug = path.stem.lower()
        if path.parent.name == "answers":
            continue  # answer.unreferenced already covers filed answers
        try:
            meta = _frontmatter(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if meta.get("status") == "generated" or meta.get("type") in ("maintenance", "log"):
            continue
        if slug in listed:
            continue
        backlinks = inbound.get(slug, set()) - {path.stem}
        if backlinks:
            continue
        try:
            rel = path.relative_to(rel_root).as_posix()
        except ValueError:
            rel = path.name
        report.add("link.orphan", INFO, rel,
                   "no inbound links and not listed in index.md; nothing navigates here")



def _lint_catalog(report: LintReport, root: Path, pages: Iterable[Path],
                  rel_root: Path) -> None:
    """index.md must not list pages that are gone, nor omit pages that exist."""
    index = root / "index.md"
    if not index.exists():
        return
    try:
        text = index.read_text(encoding="utf-8", errors="replace")
    except OSError:
        report.add("catalog.unreadable", WARN, "index.md", "could not read the catalog")
        return
    listed = {t.strip().lower() for t in _wikilink_targets(text)}
    if not listed:
        return
    present = {p.stem.lower() for p in pages}
    for slug in sorted(listed - present):
        report.add("catalog.stale", WARN, "index.md",
                   f"lists [[{slug}]] but no such memory page exists")
    for slug in sorted(present - listed):
        report.add("catalog.entry_missing", WARN, "index.md",
                   f"memory page `{slug}` is not listed; run `hermes hwiki catalog`")


def _lint_answers(report: LintReport, mem: Path, rel_root: Path,
                  pages: Iterable[Path]) -> None:
    """Filed answers are not facts; flag ones nothing references."""
    answers_dir = mem / "answers"
    if not answers_dir.is_dir():
        return
    corpus = []
    for path in pages:
        if answers_dir in path.parents:
            continue
        try:
            corpus.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    blob = "\n".join(corpus)
    for path in sorted(answers_dir.rglob("*.md")):
        try:
            rel = path.relative_to(rel_root).as_posix()
        except ValueError:
            rel = path.name
        if path.stem not in blob and f"[[{path.stem}]]" not in blob:
            report.add("answer.unreferenced", INFO, rel,
                       "no fact page references this filed answer")


def format_report(report: LintReport, *, show_info: bool = False) -> str:
    """Human-readable rendering. Empty string when there is nothing to say."""
    rows = [f for f in report.sorted_findings()
            if show_info or f.severity != INFO]
    if not rows:
        return ""
    return "\n".join(f.line for f in rows)


def to_dict(report: LintReport) -> dict:
    counts = report.counts()
    return {
        "pages_checked": report.pages_checked,
        "ok": report.ok,
        "counts": counts,
        "findings": [
            {"check": f.check, "severity": f.severity, "page": f.page,
             "detail": f.detail}
            for f in report.sorted_findings()
        ],
    }


# ---------------------------------------------------------------------------
# semantic suggestions (opt-in, advisory only)
# ---------------------------------------------------------------------------

# The roadmap is explicit: semantic lint "may suggest candidates only; it must
# never silently declare or resolve contradictions". So this pass never writes,
# never changes a status, and never says "these contradict" — it says "these are
# close enough that a human should look". Two thresholds control it:
#
#   SUGGEST_MIN  — below this the facts are merely in the same topic
#   SUGGEST_MAX  — above this they are handled by the duplicate/near-dup gate on
#                  the write path already, so reporting them here is noise
SUGGEST_MIN = 0.62
SUGGEST_MAX = 0.90


def suggest_review_pairs(store, *, min_score: float = SUGGEST_MIN,
                         max_score: float = SUGGEST_MAX,
                         limit: int = 25) -> LintReport:
    """Pairs of accepted facts close enough to deserve a human look.

    Read-only and advisory. Findings are always ``info`` severity: nothing here
    can make a vault fail lint, because similarity is not evidence of a problem.
    """
    from .facts import _tokens

    report = LintReport()
    rows = [r for r in store.list_proposals(status="accepted", limit=10000)
            if r.get("claim") and r.get("embedding")]
    report.pages_checked = len(rows)
    if len(rows) < 2:
        return report

    from .facts import _cosine_batch
    vectors = [r["embedding"] for r in rows]
    token_sets = [set(_tokens(str(r["claim"]))) for r in rows]
    pairs = []
    for i in range(len(rows)):
        cosines = _cosine_batch(vectors[i], vectors[i + 1:])
        for offset, cos in enumerate(cosines, start=1):
            j = i + offset
            overlap = len(token_sets[i] & token_sets[j]) / max(
                1, len(token_sets[i] | token_sets[j]))
            score = max(cos, 0.5 * cos + 0.5 * overlap)
            if min_score <= score < max_score:
                pairs.append((score, i, j))
    pairs.sort(key=lambda p: p[0], reverse=True)

    for score, i, j in pairs[:limit]:
        page = _claim_page(rows[i]) or _claim_page(rows[j]) or "facts"
        report.add("semantic.review_pair", INFO, page,
                   f"{rows[i]['claim'][:60]!r} ~ {rows[j]['claim'][:60]!r} "
                   f"(similarity {score:.2f}; compare, do not auto-resolve)")
    return report


def _claim_page(row: dict) -> str:
    page = row.get("page_path") or ""
    if not page:
        return ""
    from pathlib import Path as _Path
    return f"{_Path(str(page)).parent.name}/{_Path(str(page)).name}"
