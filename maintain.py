"""Autonomous maintenance: work the memory does without being asked.

Honcho's autonomy comes from a deriver queue that runs in the background. hwiki has
no daemon and, by design, no external model at runtime — so "autonomous" here means a
**bounded, deterministic tick** that does the maintenance a human would otherwise be
asked to trigger, and writes only artifacts that are safe to regenerate.

What a tick does
----------------
1. ``heal``       — rebuild the index from the vault when the DB looks short or the
                    vault gained pages (the vault is authoritative)
2. ``conflicts``  — sweep *accepted* facts against each other for contradictions the
                    propose-time check never saw (two facts accepted in different
                    sessions may now disagree)
3. ``duplicates`` — same sweep, looking for facts that say the same thing twice
4. ``questions``  — write down what recall could not answer, so the next conversation
                    can ask instead of silently forgetting
5. ``artifacts``  — refresh ``index.md`` / ``log.md`` when lint says they are stale

What a tick deliberately does NOT do
------------------------------------
* It never accepts, supersedes, merges or deletes a fact. The review gate stays the
  only way memory changes meaning; a tick queues work for a human, it does not decide.
* It never guesses provenance. If a page is unreadable, the tick reports it.
* It never runs the model unless the opt-in NLI flags are set.

Everything it writes goes to one derived page, ``memory_dir/maintenance.md``, plus
meta bookkeeping. That page is regenerable, is excluded from fact recall (it lives in
the memory zone but carries no ``# claim``), and is safe to delete.

Budgeting
---------
Each tick takes a wall-clock budget (default 2 s). Jobs are ordered cheapest-first and
the tick stops when the budget is spent, reporting what it skipped. A tick therefore
cannot stall a conversation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

MAINTENANCE_FILENAME = "maintenance.md"
META_LAST_RUN = "maintenance_last_run"
META_OPEN_QUESTIONS = "open_questions"
# The pairs found by the last tick, so the prefetch path can raise one without a sweep.
META_CONFLICTS = "maintenance_conflicts"
META_DUPLICATES = "maintenance_duplicates"
DEFAULT_INTERVAL_HOURS = 24
DEFAULT_BUDGET_MS = 2000

# A fact pair must share this much vocabulary before the sweep considers it. The
# cheap prefilter keeps the sweep O(pairs-that-could-possibly-relate) instead of O(n^2)
# on a vault with thousands of facts.
MIN_PAIR_OVERLAP = 0.20
MAX_FACTS_SWEPT = 400
MAX_PAIRS_CHECKED = 2000
MAX_LISTED = 25

_STOP = frozenset("""
a an the and or but if then than so is are was were be been being am
of in on at to for from by with without into onto over under about
it its this that these those there here we you they i he she
do does did done have has had can could will would should may might must
not no never always every all any some more most less least very
user users they them their our your my
""".split())


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@dataclass
class JobResult:
    name: str
    ran: bool = False
    skipped: str = ""
    notes: List[str] = field(default_factory=list)
    count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"job": self.name, "ran": self.ran, "skipped": self.skipped,
                "count": self.count, "notes": self.notes}


@dataclass
class MaintenanceReport:
    elapsed_ms: int = 0
    budget_ms: int = DEFAULT_BUDGET_MS
    ran_at: str = ""
    jobs: List[JobResult] = field(default_factory=list)
    conflicts: List[Tuple[str, str, float]] = field(default_factory=list)
    duplicates: List[Tuple[str, str, float]] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    stale_artifacts: List[str] = field(default_factory=list)
    written: bool = False
    path: str = ""

    @property
    def work_items(self) -> int:
        return len(self.conflicts) + len(self.duplicates) + len(self.questions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elapsed_ms": self.elapsed_ms,
            "budget_ms": self.budget_ms,
            "ran_at": self.ran_at,
            "jobs": [j.to_dict() for j in self.jobs],
            "work_items": self.work_items,
            "conflicts": [{"a": a, "b": b, "score": s} for a, b, s in self.conflicts],
            "duplicates": [{"a": a, "b": b, "score": s} for a, b, s in self.duplicates],
            "questions": self.questions,
            "stale_artifacts": self.stale_artifacts,
            "written": self.written,
            "path": self.path,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _content_tokens(text: str) -> set:
    out = set()
    for raw in str(text or "").lower().split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch in ".-_")
        if len(token) > 1 and token not in _STOP:
            out.add(token)
    return out


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _claim_sort_key(fact: Dict[str, Any]) -> str:
    return str(fact.get("accepted_at") or "")


def render(report: MaintenanceReport, *, memory_dir: Optional[Path] = None) -> str:
    """The derived page. Regenerated every tick; never contains a `# claim` heading."""
    stamp = report.ran_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "---",
        "title: Memory maintenance",
        "type: maintenance",
        "status: generated",
        "source: hwiki",
        f"ran: {stamp}",
        "---",
        "",
        "# Memory maintenance",
        "",
        "> Generated by `hermes hwiki maintain`. Regenerated on every run — edits here "
        "are lost. Nothing on this page changes what memory contains: it lists work "
        "for a human to approve.",
        "",
        f"Ran in {report.elapsed_ms} ms of a {report.budget_ms} ms budget · "
        f"{report.work_items} item(s) found.",
        "",
    ]

    skipped = [j for j in report.jobs if not j.ran]
    if skipped:
        lines.append("**Skipped this run:** " + "; ".join(
            f"`{j.name}` ({j.skipped or 'budget'})" for j in skipped))
        lines.append("")

    lines.append("## Possible contradictions")
    lines.append("")
    if report.conflicts:
        lines.append("Two accepted facts may disagree. Neither has been changed — "
                     "resolve by hand or let the fact be superseded. These pairs are "
                     "also stored in meta, so a conversation about one of them is told "
                     "about the disagreement instead of silently trusting either side.")
        lines.append("")
        for a, b, score in report.conflicts:
            lines.append(f"- **{a}**")
            lines.append(f"  vs **{b}**  _(score {score:.2f})_")
    else:
        lines.append("None found.")
    lines.append("")

    lines.append("## Possible duplicates")
    lines.append("")
    if report.duplicates:
        for a, b, score in report.duplicates:
            lines.append(f"- **{a}**")
            lines.append(f"  ~ **{b}**  _(score {score:.2f})_")
    else:
        lines.append("None found.")
    lines.append("")

    lines.append("## Open questions")
    lines.append("")
    if report.questions:
        lines.append("Questions recall could not answer from memory. Worth asking, "
                     "rather than assuming or re-deriving.")
        lines.append("")
        for question in report.questions:
            lines.append(f"- {question}")
    else:
        lines.append("None recorded.")
    lines.append("")

    if report.stale_artifacts:
        lines.append("## Stale artifacts")
        lines.append("")
        for item in report.stale_artifacts:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# open-question capture (in-memory, flushed by a tick)
# ---------------------------------------------------------------------------

_QUESTION_WORDS = frozenset(
    "what when where who why how which whose whom".split())


def looks_like_question(query: str) -> bool:
    text = str(query or "").strip()
    if len(text) < 12:
        return False
    lowered = text.lower()
    if "?" in text:
        return True
    return any(token in _QUESTION_WORDS for token in
               re.split(r"[^a-z]+", lowered)[:4])


def normalise_question(query: str) -> str:
    return " ".join(str(query or "").strip().rstrip("?").split()).lower()


class QuestionBuffer:
    """Bounded, deduplicated list of questions recall could not answer.

    Held in memory: recording a miss must not write to SQLite, because recall is a
    read path and a write there would show up in every conversation's latency. A tick
    flushes the buffer to meta, so the cost lands on maintenance instead.
    """

    def __init__(self, limit: int = 100):
        self.limit = limit
        self._counts: Dict[str, Tuple[str, int]] = {}

    def note(self, query: str) -> None:
        if not looks_like_question(query):
            return
        key = normalise_question(query)
        if not key:
            return
        text = " ".join(str(query).strip().split())
        previous = self._counts.get(key)
        self._counts[key] = (previous[0] if previous else text,
                             (previous[1] if previous else 0) + 1)
        if len(self._counts) > self.limit:
            # drop the least-asked question, keeping the recent hot ones
            worst = min(self._counts, key=lambda k: (self._counts[k][1], k))
            self._counts.pop(worst, None)

    def drain(self) -> List[str]:
        """Newest-first by ask count, then alphabetically; clears the buffer."""
        items = sorted(self._counts.values(), key=lambda p: (-p[1], p[0]))
        self._counts.clear()
        return [text for text, _count in items]

    def merge_into(self, store, limit: int = MAX_LISTED) -> List[str]:
        """Fold the buffer into store meta and return the merged open questions."""
        import json

        fresh = self.drain()
        try:
            existing = json.loads(store.get_meta(META_OPEN_QUESTIONS) or "[]")
        except Exception:
            existing = []
        merged: List[Tuple[str, int]] = []
        seen: Dict[str, int] = {}
        for index, text in enumerate(list(existing) + fresh):
            key = normalise_question(str(text))
            if not key:
                continue
            if key in seen:
                merged[seen[key]] = (merged[seen[key]][0], merged[seen[key]][1] + 1)
            else:
                seen[key] = len(merged)
                merged.append((" ".join(str(text).split()), 1 if index < len(existing) else 2))
        merged.sort(key=lambda p: (-p[1], p[0]))
        merged = merged[:limit]
        try:
            store.set_meta(META_OPEN_QUESTIONS,
                           json.dumps([text for text, _ in merged], ensure_ascii=False))
        except Exception:
            pass
        return [text for text, _ in merged]


# ---------------------------------------------------------------------------
# the tick
# ---------------------------------------------------------------------------

def _accepted_claims(store, limit: int = MAX_FACTS_SWEPT) -> List[str]:
    rows = store.list_proposals(status="accepted", limit=limit)
    claims = []
    for row in rows:
        claim = str(row.get("claim") or "").strip()
        if claim and (not row.get("superseded_by")):
            claims.append(claim)
    return claims


def run_maintenance(store, vault_root, memory_rel, *, embedder=None,
                    budget_ms: int = DEFAULT_BUDGET_MS, write: bool = True,
                    question_buffer: Optional[QuestionBuffer] = None,
                    now: Optional[datetime] = None) -> MaintenanceReport:
    """One bounded maintenance tick. Deterministic; writes only derived artifacts."""
    started = time.monotonic()
    deadline = started + max(50, budget_ms) / 1000.0
    report = MaintenanceReport(
        budget_ms=budget_ms,
        ran_at=(now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
    )
    vault_root = Path(vault_root)
    memory_rel = Path(memory_rel)

    def remaining() -> bool:
        return time.monotonic() < deadline

    # -- 1. heal ------------------------------------------------------------
    heal = JobResult("heal")
    if not remaining():
        heal.skipped = "budget"
    else:
        try:
            accepted = len(store.list_proposals(status="accepted", limit=10000))
            on_disk = len(list((vault_root / memory_rel).glob("*.md"))) \
                if (vault_root / memory_rel).is_dir() else 0
            if on_disk > accepted:
                result = store.rebuild_from_vault(vault_root, memory_rel)
                heal.ran = True
                heal.count = int(result.get("imported", 0))
                if heal.count:
                    heal.notes.append(f"recovered {heal.count} fact page(s) missing from the index")
            else:
                heal.ran = True
                heal.notes.append("index matches the vault")
        except Exception as exc:
            heal.skipped = f"error: {exc}"
    report.jobs.append(heal)

    # -- 2/3. conflicts + duplicates ---------------------------------------
    conflicts = JobResult("conflicts")
    duplicates = JobResult("duplicates")
    if not remaining():
        conflicts.skipped = duplicates.skipped = "budget"
    else:
        try:
            claims = _accepted_claims(store)
            tokens = [_content_tokens(c) for c in claims]
            pairs = 0
            for i in range(len(claims)):
                if pairs >= MAX_PAIRS_CHECKED or not remaining():
                    break
                for j in range(i + 1, len(claims)):
                    if _overlap(tokens[i], tokens[j]) < MIN_PAIR_OVERLAP:
                        continue
                    pairs += 1
                    if pairs > MAX_PAIRS_CHECKED:
                        break
                    verdict = _pair_relation(claims[i], claims[j], embedder,
                                             tokens[i], tokens[j])
                    if verdict == "contradiction":
                        report.conflicts.append((claims[i], claims[j],
                                                 round(_pair_score(tokens[i], tokens[j]), 2)))
                        if len(report.conflicts) >= MAX_LISTED:
                            break
                    elif verdict == "duplicate":
                        report.duplicates.append((claims[i], claims[j],
                                                  round(_pair_score(tokens[i], tokens[j]), 2)))
                        if len(report.duplicates) >= MAX_LISTED:
                            break
            conflicts.ran = duplicates.ran = True
            conflicts.count = len(report.conflicts)
            duplicates.count = len(report.duplicates)
            if not remaining():
                conflicts.notes.append("stopped at the budget")
        except Exception as exc:
            conflicts.skipped = duplicates.skipped = f"error: {exc}"
    report.jobs.append(conflicts)
    report.jobs.append(duplicates)

    # -- 4. open questions -------------------------------------------------
    questions = JobResult("questions")
    if question_buffer is None:
        questions.skipped = "no buffer (nothing asked this process)"
    else:
        try:
            report.questions = question_buffer.merge_into(store)
            questions.ran = True
            questions.count = len(report.questions)
        except Exception as exc:
            questions.skipped = f"error: {exc}"
    report.jobs.append(questions)

    # -- 5. artifacts ------------------------------------------------------
    artifacts = JobResult("artifacts")
    try:
        stale = _artifact_findings(store, vault_root, memory_rel)
        report.stale_artifacts = stale
        if stale:
            try:
                from .catalog import write_index
            except ImportError:  # pragma: no cover - loaded as a top-level module
                try:
                    from catalog import write_index  # type: ignore
                except ImportError:
                    write_index = None
            if write_index is not None and remaining():
                write_index(store)
                artifacts.ran = True
                artifacts.notes.append(f"regenerated index.md ({len(stale)} stale entr(ies))")
            else:
                artifacts.skipped = "index writer unavailable"
        else:
            artifacts.ran = True
            artifacts.notes.append("catalogs current")
    except Exception as exc:
        artifacts.skipped = f"error: {exc}"
    report.jobs.append(artifacts)

    report.elapsed_ms = int((time.monotonic() - started) * 1000)

    # -- persist the report ------------------------------------------------
    if write:
        target = vault_root / memory_rel / MAINTENANCE_FILENAME
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(render(report), encoding="utf-8")
            tmp.replace(target)
            report.written = True
            report.path = str(target)
            store.set_meta(META_LAST_RUN, report.ran_at)
            # Keep the pair list where a turn can reach it cheaply: the prefetch path
            # must not pay for a sweep. Replaced wholesale each tick, so a conflict
            # that a human resolved stops being raised.
            for key, pairs in ((META_CONFLICTS, report.conflicts),
                               (META_DUPLICATES, report.duplicates)):
                store.set_meta(key, json.dumps(
                    [{"a": a, "b": b, "score": s, "at": report.ran_at}
                     for a, b, s in pairs][:MAX_LISTED]))
        except Exception:
            report.written = False
    return report


def _pair_score(a: set, b: set) -> float:
    return _overlap(a, b)


DUPLICATE_SCORE = 0.90


def _pair_relation(a: str, b: str, embedder=None, toks_a=None, toks_b=None) -> str:
    """'contradiction' | 'duplicate' | ''.

    Deterministic signals run first and always; the model is consulted only when it
    is explicitly enabled, because loading it costs ~165 MB of RSS and a background
    tick must not spend that on a vault whose owner never opted in.
    """
    try:  # local import: facts imports this module, so this must not run at import time
        from .facts import _negation_signal
    except ImportError:  # pragma: no cover - loaded as a top-level module
        try:
            from facts import _negation_signal  # type: ignore
        except ImportError:
            _negation_signal = None

    if _negation_signal is not None:
        try:
            if _negation_signal(a) != _negation_signal(b):
                return "contradiction"
        except Exception:
            pass

    if embedder is not None:
        try:
            from .facts import _cosine_bytes
        except ImportError:  # pragma: no cover
            _cosine_bytes = None
        if _cosine_bytes is not None:
            try:
                cos = _cosine_bytes(embedder.embed(a), embedder.embed(b))
                overlap = _overlap(toks_a or _content_tokens(a), toks_b or _content_tokens(b))
                if max(cos, 0.5 * cos + 0.5 * overlap) >= DUPLICATE_SCORE:
                    return "duplicate"
            except Exception:
                pass

    try:
        from . import verdict as verdict_mod
    except ImportError:  # pragma: no cover - loaded as a top-level module
        try:
            import verdict as verdict_mod  # type: ignore
        except ImportError:
            return ""
    try:
        if not (verdict_mod.enabled() or verdict_mod.dup_enabled()):
            return ""
        scorer = verdict_mod.try_get_scorer()
    except Exception:
        return ""
    if scorer is None:
        return ""
    try:
        relation = verdict_mod.classify_pair(a, b, scorer)
    except Exception:
        return ""
    if relation["relation"] == "contradiction":
        return "contradiction"
    if relation["relation"] == "restatement":
        return "duplicate"
    return ""


def _artifact_findings(store, vault_root: Path, memory_rel: Path) -> List[str]:
    findings: List[str] = []
    try:
        from .lint import lint_memory
    except ImportError:  # pragma: no cover - loaded as a top-level module
        try:
            from lint import lint_memory  # type: ignore
        except ImportError:
            return findings
    try:
        report = lint_memory(store)
    except Exception:
        return findings
    for finding in getattr(report, "findings", []):
        if str(getattr(finding, "code", "")) in ("catalog.stale", "catalog.entry_missing"):
            findings.append(f"{finding.code}: {finding.subject}")
    return findings[:MAX_LISTED]


def interval_hours(config: Optional[Dict[str, Any]] = None) -> float:
    section = (config or {}).get("maintenance") or {}
    try:
        return float(section.get("interval_hours", DEFAULT_INTERVAL_HOURS))
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_HOURS


def due(store, config: Optional[Dict[str, Any]] = None,
        now: Optional[datetime] = None) -> bool:
    """True when the last tick is older than the configured interval."""
    hours = interval_hours(config)
    if hours <= 0:
        return False
    last = ""
    try:
        last = str(store.get_meta(META_LAST_RUN) or "")
    except Exception:
        return False
    if not last:
        return True
    try:
        then = datetime.fromisoformat(last.replace("Z", "+00:00"))
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    current = now or datetime.now(timezone.utc)
    return (current - then).total_seconds() >= hours * 3600
