"""Read-only ingest planning (LLM Wiki "propose a change set" step).

Given source material the caller explicitly supplies, work out **what would
happen** if each candidate claim were proposed: which page it would land on, what
it would conflict with, whether it is a near-duplicate, and which step of the
lifecycle it would enter. Nothing is written — no proposal rows, no pages, no
catalog, no log.

That read-only property is the whole point. The roadmap's Phase 3 forbids letting
an ingestion path touch the vault before a human has reviewed it, and forbids
reading ``raw/`` at all. A planner that wrote its own proposals would quietly be
an ingestion pipeline.

Two honest limits, stated up front:

* **This is not an extractor.** It does not call a model. Candidate claims are
  split naively on sentence/line boundaries, which is good enough for a caller who
  supplies tidy material (and for the tests), and poor for raw prose. A caller with
  a real extractor should pass the claims it already has.
* **A plan is a prediction, not a promise.** It reuses the same scoring the accept
  path uses, but the review gate can still reject, and a human can reword.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .compile import MERGE_SIMILARITY, choose_owner
from .facts import related_and_conflict

# Actions a candidate can map to. Deliberately the same vocabulary the accept path
# uses, so a plan reads like the lifecycle it predicts.
NEW_PAGE = "new_page"
MERGE = "merge_into_owner"
DUPLICATE = "near_duplicate"
CONFLICT = "conflict"

MAX_CLAIM_CHARS = 400

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_NOISE = re.compile(r"^(?:#|>|---|\*\*|\[|\|)")


@dataclass
class PlanEntry:
    """What would happen to one candidate claim."""

    claim: str
    action: str
    owner_slug: str = ""
    owner_claim: str = ""
    score: float = 0.0
    conflicts_with: str = ""
    duplicate_of: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def writes_anywhere(self) -> bool:
        """A plan never writes; the property exists so tests can assert it."""
        return False

    def line(self) -> str:
        where = ""
        if self.owner_slug:
            where = f" -> [[{self.owner_slug}]] ({self.score:.2f})"
        detail = "; ".join(self.notes)
        return (f"  [{self.action}]{where}: {self.claim[:96]}"
                + (f"\n      {detail}" if detail else ""))


@dataclass
class Plan:
    entries: List[PlanEntry] = field(default_factory=list)
    source: str = ""
    skipped: int = 0
    tier: str = ""

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for entry in self.entries:
            out[entry.action] = out.get(entry.action, 0) + 1
        return out

    @property
    def writes_performed(self) -> int:
        return 0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "tier": self.tier,
            "writes_performed": 0,
            "skip_unsafe": True,
            "skipped_candidates": self.skipped,
            "counts": self.counts(),
            "entries": [
                {
                    "claim": e.claim,
                    "action": e.action,
                    "owner_slug": e.owner_slug,
                    "owner_claim": e.owner_claim,
                    "score": e.score,
                    "conflicts_with": e.conflicts_with,
                    "duplicate_of": e.duplicate_of,
                    "notes": e.notes,
                }
                for e in self.entries
            ],
        }


def candidate_claims(text: str, *, limit: int = 40) -> List[str]:
    """Split supplied material into plausible claims.

    Naive by design (no model). Bullets and sentences are kept, markdown noise and
    anything too short to be a claim are dropped.
    """
    out: List[str] = []
    for raw in _SENTENCE_SPLIT.split(text or ""):
        line = _BULLET.sub("", raw.strip())
        if not line or _NOISE.match(line):
            continue
        if len(line) < 8 or len(line) > MAX_CLAIM_CHARS:
            continue
        if not re.search(r"[A-Za-z]", line):
            continue
        if line not in out:
            out.append(line)
        if len(out) >= limit:
            break
    return out


def build_plan(store, claims: Sequence[str], *, source: str = "",
               memory_rel: Optional[str] = None) -> Plan:
    """Predict the lifecycle for each claim. Reads the store; writes nothing."""
    from .facts import _secret_formed  # local: keep module import surface small

    plan = Plan(source=source,
                tier="onnx-minilm" if getattr(store.embedder, "_onnx", None) is not None
                else "hash")
    existing = []
    # list_proposals is paged (default 20); a plan must see every accepted fact or
    # it would invent an owner that already exists.
    for row in store.list_proposals(status="accepted", limit=10000):
        if row.get("page_path"):
            existing.append({
                "key": row["id"],
                "text": row["claim"],
                "path": row["page_path"],
                "claim": row["claim"],
            })

    for claim in claims:
        if _secret_formed(claim):
            # A secret-shaped candidate is reported, never planned for writing.
            plan.skipped += 1
            plan.entries.append(PlanEntry(
                claim=claim[:80] + ("…" if len(claim) > 80 else ""),
                action="refused_secret",
                notes=["secret-shaped; would be refused at propose time"]))
            continue

        related, conflict, dup_key = related_and_conflict(
            claim, [(r["key"], r["text"]) for r in existing], store.embedder)

        by_key = {r["key"]: r for r in existing}
        candidates = []
        for rel in related:
            row = by_key.get(rel["key"])
            if not row:
                continue
            candidates.append({
                "score": rel["score"],
                "path": row["path"],
                "claim": row["claim"],
                "conflict": conflict,
            })
        owner = choose_owner(candidates, threshold=MERGE_SIMILARITY)

        entry = PlanEntry(claim=claim, action=NEW_PAGE)
        if dup_key:
            entry.action = DUPLICATE
            entry.duplicate_of = by_key.get(dup_key, {}).get("claim", "")[:96]
            entry.notes.append("would be rejected as a near-duplicate at propose time")
        elif conflict:
            entry.action = CONFLICT
            entry.conflicts_with = related[0]["text"][:96] if related else ""
            entry.notes.append("negation/contradiction with an existing fact; "
                               "needs explicit review, never auto-accepted")
        elif owner:
            entry.action = MERGE
            from pathlib import Path as _Path
            entry.owner_slug = _Path(str(owner.get("path"))).stem
            entry.owner_claim = str(owner.get("claim", ""))[:96]
            entry.score = float(owner.get("score", 0.0))
            entry.notes.append(f"would become a compiled section of the owning page "
                               f"(threshold {MERGE_SIMILARITY})")
        else:
            entry.score = float(related[0]["score"]) if related else 0.0
            entry.notes.append("no page owns this topic yet")
        plan.entries.append(entry)
    return plan


def format_plan(plan: Plan) -> str:
    if not plan.entries:
        return "  (no candidate claims found in the supplied material)"
    counts = plan.counts()
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    lines = [f"  {len(plan.entries)} candidate(s): {summary}",
             "  writes performed: 0 (read-only plan)", ""]
    lines.extend(entry.line() for entry in plan.entries)
    if plan.skipped:
        lines += ["", f"  {plan.skipped} candidate(s) refused for secret material."]
    return "\n".join(lines)
