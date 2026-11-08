"""Graders for the offline behavioural suite.

Deliberately dependency-free: they take plain hit dicts, so the rules that decide
pass/fail can be mutation-tested without a store, a model, or ONNX.

Design rule (learned the hard way in Phase 6): grade the *claim* that is being
presented as current, never a substring somewhere in a formatted block. A
superseded value legitimately appears in lineage (``replaced``), so joining a
whole block and searching it would fail a correct answer *and* pass a wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class Grade:
    passed: bool
    reason: str
    detail: str = ""


def _claims(hits: Sequence[Dict[str, Any]], top: int) -> List[str]:
    """Current claims of the top-N hits, lowercased.

    Only the claim text counts. ``replaced`` is lineage metadata that names the
    superseded value on purpose and must never be treated as a current answer.
    """
    return [str(h.get("claim") or "").lower() for h in hits[:top]]


def _contains_any(texts: Iterable[str], needles: Iterable[str]) -> Optional[str]:
    for needle in needles:
        low = needle.lower()
        for text in texts:
            if low in text:
                return needle
    return None


@dataclass
class Expect:
    """What a case asserts. All declared clauses must hold to pass."""

    # ANY of these substrings must appear in the claim of a hit within `top`.
    needs_any: Sequence[str] = ()
    # ALL of these must appear (in one claim each, not necessarily the same one).
    needs_all: Sequence[str] = ()
    # None of these may appear in ANY current claim: the stale/forbidden value.
    forbid_current: Sequence[str] = ()
    # The correct answer should be ranked first.
    top1: bool = False
    # Query must return nothing at all (unknown / not in memory).
    refuse: bool = False
    # Coverage: at least this many distinct claims.
    min_distinct: int = 0
    top: int = 5


def grade(hits: Sequence[Dict[str, Any]], expect: Expect) -> Grade:
    claims = _claims(hits, expect.top)
    joined = " | ".join(claims)

    if expect.refuse:
        if not hits:
            return Grade(True, "refused (no hits)")
        return Grade(False, f"should have refused but returned {claims[0][:70]!r}")

    if expect.min_distinct:
        distinct = len({c for c in claims if c})
        if distinct < expect.min_distinct:
            return Grade(False, f"covered {distinct} facts, need {expect.min_distinct}")

    if expect.forbid_current:
        bad = _contains_any(claims, expect.forbid_current)
        if bad:
            return Grade(False, f"presented superseded value {bad!r} as current")

    if not hits:
        return Grade(False, "no hits (expected a relevant fact)")

    if expect.needs_any:
        found = _contains_any(claims, expect.needs_any)
        if not found:
            return Grade(False,
                         f"none of {list(expect.needs_any)} in current claims")

    for needle in expect.needs_all:
        if _contains_any(claims, [needle]) is None:
            return Grade(False, f"missing required {needle!r}")

    if expect.top1:
        first = claims[0] if claims else ""
        if expect.needs_any and _contains_any([first], expect.needs_any) is None:
            return Grade(False, f"correct fact not ranked first: {first[:70]!r}")

    return Grade(True, "ok", joined[:90])


# ---------------------------------------------------------------------------
# Mutation self-checks: prove the graders actually fail bad answers.
#
# Each returns (ok, message). "ok" means the grader behaved CORRECTLY on the
# mutated input — including correctly rejecting a plausible-looking wrong answer.
# ---------------------------------------------------------------------------

def _h(claim: str, **kw) -> Dict[str, Any]:
    base = {"id": "x", "kind": "preference", "claim": claim,
            "evidence": "", "replaced": "", "score": 0.9, "source": "fact"}
    base.update(kw)
    return base


def mutation_checks() -> List[tuple[str, bool, str]]:
    out: List[tuple[str, bool, str]] = []

    good = [_h("User's main editor is VS Code")]
    out.append(("correct answer passes",
                grade(good, Expect(needs_any=["vs code"])).passed,
                "expected pass"))

    out.append(("wrong answer fails",
                not grade([_h("User's main editor is Sublime")],
                          Expect(needs_any=["vs code"])).passed,
                "expected fail"))

    # The key regression: the expected word in lineage must NOT count.
    lineage = [_h("Staging stays on Fly.io", replaced="Staging moves to Hetzner")]
    out.append(("expected word only in lineage fails",
                not grade(lineage, Expect(needs_any=["hetzner"])).passed,
                "lineage must not satisfy needs_any"))

    # Stale value named in lineage must not trip forbid_current.
    out.append(("stale value in lineage is allowed",
                grade(lineage,
                      Expect(needs_any=["fly.io"],
                             forbid_current=["moves to hetzner"])).passed,
                "lineage mentioning the old value is not a stale answer"))

    # ...but a stale value presented AS the claim must fail.
    out.append(("stale value as current claim fails",
                not grade([_h("Staging moves to Hetzner")],
                          Expect(needs_any=["fly.io"],
                                 forbid_current=["hetzner"])).passed,
                "stale claim must fail"))

    out.append(("substring in unrelated claim does not pass top1",
                not grade([_h("User likes dark mode"), _h("User's editor is VS Code")],
                          Expect(needs_any=["vs code"], top1=True)).passed,
                "top1 must inspect the first claim"))

    out.append(("refusal with no hits passes",
                grade([], Expect(refuse=True)).passed, "expected pass"))
    out.append(("refusal with a hit fails",
                not grade([_h("anything")], Expect(refuse=True)).passed,
                "expected fail"))

    out.append(("coverage shortfall fails",
                not grade([_h("a"), _h("b")], Expect(min_distinct=4)).passed,
                "expected fail"))
    out.append(("coverage met passes",
                grade([_h("a"), _h("b"), _h("c"), _h("d")],
                      Expect(min_distinct=4)).passed, "expected pass"))

    out.append(("empty result fails a needs_any case",
                not grade([], Expect(needs_any=["x"])).passed, "expected fail"))

    return out


def report_mutations() -> bool:
    checks = mutation_checks()
    width = max(len(name) for name, _, _ in checks)
    ok = True
    for name, passed, note in checks:
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<{width}}  ({note})")
    return ok


if __name__ == "__main__":
    print("grader mutation checks")
    raise SystemExit(0 if report_mutations() else 1)
