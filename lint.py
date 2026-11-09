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


def _code_span_targets(text: str) -> List[str]:
    """Single-backtick spans in ``text`` (the catalog's listing for unlinkable paths)."""
    import re
    return [m.group(1).strip() for m in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", text)]


def _wikilink_targets(text: str) -> List[str]:
    """Link targets in page text, in order, de-duplicated.

    Code fences and inline code are literal, exactly as in ``WikiStore``'s
    ``Page.links``: a note that documents wikilink syntax must not have its samples
    reported as dangling links.
    """
    from .store import strip_code  # lazy: this module stays leaf-only at import time

    seen: List[str] = []
    for raw in _LINK_RE.findall(strip_code(text)):
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
    from .store import scan_for_secrets, slugify  # local import keeps this module leaf-only

    report = LintReport()
    root = Path(getattr(store, "root", store))
    mem = Path(memory_dir) if memory_dir is not None else (root / store.memory_dir)
    pages = _memory_pages(mem)
    report.pages_checked = len(pages)
    if not pages:
        return report

    # Page identity, in the same tiers ``WikiStore.resolve_page`` uses: an exact
    # root-relative path, a unique case-insensitive path, a filename slug, then a
    # title. A tier that yields matches ends the search, so a link is ambiguous only
    # when the store itself would be, and ``Docs/Plan.md`` never merges with
    # ``docs/plan.md``. ``indexed`` is what the catalog is expected to list: every
    # page except the catalog's own artifacts.
    identity = _Identity()
    indexed = set()
    sources: List[Path] = []
    try:
        from .catalog import is_catalog_artifact, linkable_target  # leaf-safe, local
        try:
            zone = mem.relative_to(root).as_posix().strip("/")
        except ValueError:  # a memory zone outside the vault: fall back to the config
            zone = str(getattr(store, "memory_dir", "") or "").strip("/")
        all_pages = list(store.iter_pages())
        # one resolver for lint and the store: a link the reader can follow must not be
        # reported dangling, and a link the store refuses (traversal, ambiguous
        # case-fold) must not be silently accepted
        resolved_cache: Dict[str, object] = {}

        def _resolve(reference: str):
            if reference not in resolved_cache:
                resolved_cache[reference] = store._resolve_in(reference, all_pages)
            return resolved_cache[reference]

        identity.resolver = _resolve
        for page in all_pages:
            identity.add(page)
            # every indexed page is a possible source of an inbound link, not just
            # the memory zone being reported on: a hand-written Hermes/ page that
            # links a note is navigation, and ignoring it flagged linked notes
            sources.append(Path(page.abspath))
            if (not is_catalog_artifact(page.path, memory_dir=zone)
                    and linkable_target(page.path)):
                # a path a wikilink cannot carry is listed by the catalog as a plain
                # path, so demanding a [[link]] for it would never be satisfiable
                indexed.add(page.path)
    except Exception:  # pragma: no cover - a broken index must not stop lint
        identity, indexed, sources = _Identity(), set(), []

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

        # Wikilinks are checked on every page in the zone: a hand-written note's
        # broken link is the most common kind, and schema checks below do not apply
        # to it (`continue`), which used to hide its dangling links entirely.
        for link in _wikilink_targets(text):
            if identity.known and identity.resolve(link) is None:
                report.add("link.dangling", ERROR, rel,
                           f"[[{link}]] does not resolve to any page")

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
                    if identity.known and not identity.candidates(link):
                        report.add("supersede.orphan", WARN, rel,
                                   f"superseded_by points at missing page [[{link}]]")

        h1 = _H1_RE.search(text)
        if status in ("", "accepted", "auto") and h1:
            key = " ".join(h1.group(1).split()).lower()
            accepted_claims.setdefault(key, []).append(rel)

        if len(text) > max_page_chars:
            report.add("page.oversized", INFO, rel,
                       f"{len(text)} chars exceeds {max_page_chars}; consider splitting")

    for claim, owners in sorted(accepted_claims.items()):
        if len(owners) > 1:
            report.add("duplicate.claim", WARN, owners[0],
                       f"same claim accepted on {len(owners)} pages: {', '.join(owners)}")

    _lint_catalog(report, root, identity=identity, indexed=indexed)
    _lint_orphans(report, root, pages, rel_root, identity=identity,
                  sources=sources)
    _lint_answers(report, mem, rel_root, pages)
    return report


class _Identity:
    """How a wikilink target maps to pages: the store's resolution order, per tier.

    Exact paths first (the only way to tell two pages with the same filename apart),
    then case-insensitive paths — accepted only when unique — then filename slugs,
    then titles. Each tier is consulted only when the previous one is empty, so lint
    never calls a link ambiguous when the store would resolve it deterministically.
    """

    TIERS = ("by_path", "by_stem", "folded_path", "by_slug", "by_title")

    def __init__(self) -> None:
        self.resolver = None  # injectable: lint_memory passes WikiStore._resolve_in
        self.by_path: Dict[str, set] = {}
        self.by_stem: Dict[str, set] = {}
        self.folded_path: Dict[str, set] = {}
        self.by_slug: Dict[str, set] = {}
        self.by_title: Dict[str, set] = {}

    @property
    def known(self) -> bool:
        """False when the vault could not be enumerated; then invent nothing."""
        return any(getattr(self, tier) for tier in self.TIERS)

    def add(self, page) -> None:
        from .store import slugify  # lazy: this module stays leaf-only at import time

        path = page.path
        # a ``.md`` reference whose folder is wrong still resolves by unique filename
        self.by_stem.setdefault(page.slug, set()).add(path)
        keys = [path]
        if path.lower().endswith(".md"):
            keys.append(path[:-3])
        for key in keys:
            self.by_path.setdefault(key, set()).add(path)
            self.folded_path.setdefault(key.lower(), set()).add(path)
        for key in {page.slug, slugify(page.slug)}:
            self.by_slug.setdefault(key, set()).add(path)
        if page.title:
            self.by_title.setdefault(slugify(page.title), set()).add(path)

    def resolve(self, target: str) -> Optional[str]:
        """The page path the store would open for ``target``, or ``None``.

        ``link.dangling`` asks this, not ``candidates``: a target the store refuses
        (traversal, an absolute path, an ambiguous case-folded path) is dangling even
        though some tier had a candidate set. ``lint_memory`` injects the store's own
        resolver, so the two cannot drift apart.
        """
        if self.resolver is not None:
            page = self.resolver(target)
            return None if page is None else page.path
        from .store import _is_traversal  # lazy: this module stays leaf-only at import time

        raw = (target or "").strip()
        if raw.startswith("[[") and raw.endswith("]]"):
            raw = raw[2:-2]
        if not raw or _is_traversal(raw):
            return None
        exact = self.by_path.get(raw)
        if exact:
            return sorted(exact)[0] if len(exact) > 1 else next(iter(exact))
        hits = self._tier_hits(raw)
        if hits is not None and len(hits) == 1:
            return next(iter(hits))
        return None

    def _tier_hits(self, target: str) -> Optional[set]:
        """Candidate paths for the first non-empty tier, or ``None`` when all are empty."""
        from .store import slugify  # lazy: this module stays leaf-only at import time

        target = target.split("|", 1)[0].split("#", 1)[0].strip()
        if target.lower().endswith(".md"):
            stem = target.rsplit("/", 1)[-1][:-3].split("\\")[-1]
            if stem in self.by_stem:
                return set(self.by_stem[stem])
        folded = self.folded_path.get(target.lower())
        if folded:
            return set(folded)
        for key in (target, slugify(target)):
            if key in self.by_slug:
                return set(self.by_slug[key])
        return self.by_title.get(slugify(target))

    def candidates(self, target: str) -> set:
        """The pages a link target names, as the store resolves them.

        Parity is by construction: when one page resolves, that page is the only
        candidate. Only an unresolvable target can report several — a case-folded path
        that matches two pages — so the catalog can call it ambiguous instead of stale
        without ever crediting a page the reader cannot open.
        """
        resolved = self.resolve(target)
        if resolved is not None:
            return {resolved}
        from .store import _is_traversal, slugify  # lazy: leaf-only at import time

        target = target.strip()
        if target.startswith("[[") and target.endswith("]]"):
            target = target[2:-2]
        if not target or _is_traversal(target):
            return set()
        folded = self.folded_path.get(target.lower())
        if folded and len(folded) > 1:
            return set(folded)
        for key in (target, slugify(target)):
            hits = self.by_slug.get(key)
            if hits and len(hits) > 1:
                return set(hits)
        return set()


def _lint_orphans(report: LintReport, root: Path, pages: Iterable[Path],
                  rel_root: Path, *, identity: Optional[_Identity] = None,
                  sources: Optional[Iterable[Path]] = None) -> None:
    """Pages that nothing navigates to.

    The pattern calls these orphans: no inbound links. In this wiki navigation is the
    catalog ("read index.md first"), so a catalog entry is an inbound route and counts.
    A page with neither a link nor an entry is reachable only by luck.

    Pages are compared by identity (their root-relative path), never by stem: two
    pages can share a filename, and collapsing a folder-qualified entry back to the
    stem made one page look catalogued because its twin was. An entry counts only
    when it names exactly one page; an inbound link that names several counts for all
    of them, because it does navigate to one of them.

    Deliberately INFO, and deliberately quiet for generated artifacts: the maintenance
    report, catalogs and filed answers are entry points by design, and flagging them every
    run would train the reader to ignore the check.
    """
    pages = list(pages)
    identity = identity or _Identity()

    def _rel(path: Path) -> str:
        try:
            return path.relative_to(rel_root).as_posix()
        except ValueError:
            return path.name

    # ``pages`` is what gets reported on (the memory zone); ``sources`` is every
    # indexed page, because navigation crosses zones.
    inbound = {}
    for path in (list(sources) if sources is not None else pages):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for target in _wikilink_targets(text):
            for candidate in identity.candidates(target):
                inbound.setdefault(candidate, set()).add(_rel(path))

    listed = set()
    index = root / "index.md"
    if index.exists():
        try:
            index_text = index.read_text(encoding="utf-8", errors="replace")
        except OSError:
            index_text = ""
        for target in _wikilink_targets(index_text):
            candidates = identity.candidates(target)
            if len(candidates) == 1:
                listed |= candidates
        # the catalog lists unlinkable paths in backticks; that is navigation too
        from .catalog import linkable_target
        for span in _code_span_targets(index_text):
            if linkable_target(span):
                continue  # a plain code span only counts where a link cannot
            listed |= identity.candidates(span)

    for path in pages:
        if path.parent.name == "answers":
            continue  # answer.unreferenced already covers filed answers
        try:
            meta = _frontmatter(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if meta.get("status") == "generated" or meta.get("type") in ("maintenance", "log"):
            continue
        rel = _rel(path)
        if rel in listed:
            continue
        backlinks = inbound.get(rel, set()) - {rel}
        if backlinks:
            continue
        report.add("link.orphan", INFO, rel,
                   "no inbound links and not listed in index.md; nothing navigates here")



def _catalog_artifact_names() -> set:
    """Basenames of the generated catalog files, for callers that only have a name.

    Location decides whether a file is an artifact (``catalog.is_catalog_artifact``):
    ``index.md``/``log.md`` at the root, the maintenance report at the top of the
    memory zone. A curated ``Projects/X/maintenance.md`` is a page, not the report.
    """
    try:
        from .catalog import INDEX_FILENAME, LOG_FILENAME, MAINTENANCE_FILENAME
    except Exception:  # pragma: no cover - bundled, defensive only
        INDEX_FILENAME, LOG_FILENAME, MAINTENANCE_FILENAME = (
            "index.md", "log.md", "maintenance.md")
    return {INDEX_FILENAME.lower(), LOG_FILENAME.lower(), MAINTENANCE_FILENAME.lower()}


def _lint_catalog(report: LintReport, root: Path, *,
                  identity: Optional[_Identity] = None,
                  indexed: Optional[Iterable[str]] = None) -> None:
    """index.md must not list pages that are gone, nor omit pages that exist.

    The catalog covers the **whole vault**: ``build_index`` emits one link per page,
    curated pages and memory notes alike. A target is resolved like a wikilink: exact
    path (with or without ``.md``), then a unique case-insensitive match, then a
    unique slug or title.

    Coverage is what the reader can open, not what a set of candidates suggests: an
    entry covers exactly the page ``resolve_page`` returns for it. A case-folded target
    that matches two pages resolves to neither, so it covers neither and is reported as
    ambiguous; a bare stem that two pages share resolves to the first match, so it
    covers that page and the other is reported missing. Judging either direction against
    the memory zone alone reported every curated page as gone on every run.
    """
    index = root / "index.md"
    if not index.exists():
        return
    try:
        text = index.read_text(encoding="utf-8", errors="replace")
    except OSError:
        report.add("catalog.unreadable", WARN, "index.md", "could not read the catalog")
        return
    listed = [t.strip() for t in _wikilink_targets(text)]
    if not listed:
        return
    if identity is None or indexed is None or not identity.known:
        return  # the vault could not be enumerated; invent nothing
    covered = set()
    for target in listed:
        resolved = identity.resolve(target)
        if resolved is not None:
            covered.add(resolved)
            continue
        candidates = identity.candidates(target)
        if len(candidates) > 1:
            report.add("catalog.ambiguous", WARN, "index.md",
                       f"lists [[{target}]] which names {len(candidates)} pages "
                       f"({', '.join(sorted(candidates))}); use a folder-qualified link")
        else:
            report.add("catalog.stale", WARN, "index.md",
                       f"lists [[{target}]] but no page in the vault resolves it")
    for path in sorted(indexed):
        if path not in covered:
            report.add("catalog.entry_missing", WARN, "index.md",
                       f"page `{path}` is not listed; run `hermes hwiki catalog`")


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
