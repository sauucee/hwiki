"""Compile: merge a newly accepted fact into the page that already owns it.

The LLM-wiki pattern's core claim is that the wiki is a *persistent, compounding
artifact* — a new source does not only add a page, it **updates** the pages it
touches. Until now hwiki's write path could only append: every accepted fact
became its own note, so two related facts ended up as two near-identical pages
with no relationship a human could see, and a refinement of an old claim sat
beside the old claim rather than inside it.

This module adds the missing operation. It runs strictly AFTER the review gate
(the user already accepted the fact) and does not touch extraction, dedup or
conflict detection — those stay in ``facts.py`` exactly as Phase 4 hardened
them. Compile only decides *where an accepted fact is filed* and *how that page
reads afterwards*.

Four guarantees, in priority order:

1. **Never lose a fact.** Every accepted claim is still retrievable afterwards:
   either it owns a page, or it is an explicit section of an owning page with
   its own heading and frontmatter provenance.
2. **Never silently contradict.** A claim flagged as conflicting is never merged
   as a refinement; it keeps its own page so the contradiction stays visible
   and resolvable by a human — which is what the review gate is for.
3. **Never clobber hand edits.** Merged sections are delimited by HTML comment
   markers. Everything outside those markers belongs to the user (or to
   hwiki's original page body) and is preserved byte for byte.
4. **Never raise.** A compile failure degrades to "keep the separate page",
   which is exactly the pre-compile behaviour, so a bug here can only lose
   consolidation — never a fact.

The operation is deterministic and idempotent: compiling the same claim twice
yields the same page, and re-compiling an already-merged claim is a no-op.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Claims this similar to an existing page's claim are treated as the SAME topic
# and merged as a refinement. Deliberately conservative: a wrong merge is worse
# than a missed one, because the missed case simply keeps pre-compile behaviour.
MERGE_SIMILARITY = 0.62

SECTION_START = "<!-- hwiki:compiled:start -->"
SECTION_END = "<!-- hwiki:compiled:end -->"
_SECTION_RE = re.compile(
    re.escape(SECTION_START) + r"(.*?)" + re.escape(SECTION_END), re.DOTALL
)
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.DOTALL)
_MAX_CLAIM = 160


class CompileResult(dict):
    """Result dict with attribute access, so callers can read ``.merged``."""

    @property
    def merged(self) -> bool:
        return bool(self.get("merged"))

    @property
    def page(self) -> Optional[str]:
        return self.get("page")

    @property
    def reason(self) -> str:
        return str(self.get("reason") or "")


def _one_line(text: str, limit: int = 200) -> str:
    return " ".join((text or "").split())[:limit]


def compiled_section_block(claim: str, *, evidence: str = "", kind: str = "fact",
                           accepted_iso: str = "", proposal_id: str = "",
                           peer: str = "user") -> str:
    """Build the delimited section that carries a merged claim."""
    lines = [
        SECTION_START,
        "",
        f"### {_one_line(claim, _MAX_CLAIM)}",
        "",
    ]
    if evidence:
        lines += [f"> **Evidence:** {_one_line(evidence, 400)}", ""]
    meta: List[str] = []
    if kind:
        meta.append(f"kind: {kind}")
    if peer:
        meta.append(f"peer: {peer}")
    if accepted_iso:
        meta.append(f"accepted: {accepted_iso}")
    if proposal_id:
        meta.append(f"proposal: {proposal_id}")
    if meta:
        lines += [f"*{' · '.join(meta)} · merged by `hwiki` compile.*", ""]
    lines.append(SECTION_END)
    return "\n".join(lines)


def existing_sections(text: str) -> List[str]:
    """Claim headings already merged into a page (used for idempotency)."""
    out: List[str] = []
    for block in _SECTION_RE.findall(text or ""):
        for line in block.splitlines():
            if line.startswith("### "):
                out.append(line[4:].strip())
                break
    return out


def append_compiled_section(text: str, block: str) -> str:
    """Append ``block`` to ``text``, keeping every existing compiled section."""
    text = (text or "").rstrip("\n")
    if not text:
        return block + "\n"
    return text + "\n\n" + block + "\n"


def _bump_compiled_count(text: str) -> str:
    """Maintain ``compiled_sections:`` and ``updated:`` inside frontmatter.

    Only the frontmatter block is touched; a page without frontmatter is
    returned unchanged, so hand-written pages are never restructured.
    """
    count = len(_SECTION_RE.findall(text or ""))
    if count <= 0:
        return text
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return text
    header = match.group(1)
    rest = text[match.end():]
    if re.search(r"^compiled_sections:", header, re.M):
        header = re.sub(r"^compiled_sections:.*$", f"compiled_sections: {count}",
                        header, count=1, flags=re.M)
    else:
        header = header.rstrip("\n") + f"\ncompiled_sections: {count}"
    today = date.today().isoformat()
    if re.search(r"^updated:", header, re.M):
        header = re.sub(r"^updated:.*$", f"updated: {today}", header,
                        count=1, flags=re.M)
    return f"---\n{header}\n---\n{rest}"


def choose_owner(candidates: Sequence[Dict[str, Any]],
                 *, threshold: float = MERGE_SIMILARITY) -> Optional[Dict[str, Any]]:
    """Pick the page that should own a new claim, or ``None``.

    ``candidates`` are dicts with at least ``score``, ``path`` and ``claim``.
    A candidate is eligible only when it is a refinement of the same topic:
    similar enough to merge, not conflict-flagged, not the fact itself.
    """
    eligible: List[Dict[str, Any]] = []
    for candidate in candidates or ():
        if candidate.get("conflict") or candidate.get("is_self"):
            continue
        if not candidate.get("path"):
            continue
        try:
            score = float(candidate.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
        if score < threshold:
            continue
        eligible.append(candidate)
    if not eligible:
        return None
    # Highest similarity wins; ties break toward the page with fewer merged
    # claims so one page cannot become a dumping ground.
    eligible.sort(key=lambda c: (-float(c.get("score") or 0.0),
                                 int(c.get("merged_count") or 0)))
    return eligible[0]


def merge_into_page(page_path: Path, claim: str, *, evidence: str = "",
                    kind: str = "fact", accepted_iso: str = "",
                    proposal_id: str = "", peer: str = "user",
                    already: Iterable[str] = ()) -> CompileResult:
    """Write the compiled section into ``page_path``.

    Idempotent: a claim already present is not re-added. Returns a
    :class:`CompileResult`; never raises.
    """
    page_path = Path(page_path)
    try:
        text = page_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return CompileResult(merged=False, page=None,
                             reason=f"owner page unreadable: {exc}")

    normalized = _one_line(claim, _MAX_CLAIM).lower()
    seen = {s.lower() for s in existing_sections(text)}
    seen.update(_one_line(a, _MAX_CLAIM).lower() for a in (already or ()))
    if normalized in seen:
        return CompileResult(merged=False, page=str(page_path),
                             reason="already compiled")

    block = compiled_section_block(claim, evidence=evidence, kind=kind,
                                   accepted_iso=accepted_iso,
                                   proposal_id=proposal_id, peer=peer)
    updated = _bump_compiled_count(append_compiled_section(text, block))

    tmp = page_path.with_name(page_path.name + ".tmp")
    try:
        tmp.write_text(updated, encoding="utf-8")
        tmp.replace(page_path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        return CompileResult(merged=False, page=None, reason=f"write failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("hwiki compile: unexpected merge failure: %s", exc)
        return CompileResult(merged=False, page=None, reason=f"error: {exc}")

    return CompileResult(merged=True, page=str(page_path), reason="merged",
                         sections=len(existing_sections(updated)))
