"""Conservative answer-type compatibility check (opt-in abstention aid).

Why this exists
---------------
Recall answers "is this fact *related* to the query?". Users often ask "is this
fact *the answer*?". Those differ: a question about a flight number scores 0.78
against a fact about trip dates, because both concern "the Lisbon trip".

Measured first, twice, before writing this module (see
``benchmarks/phase2-answerability.md``):

* attribute-focus similarity did NOT separate the two (a correct answer scored
  -0.003 while the wrong one scored +0.228);
* head-term anchoring abstained on **21 of 21** correct answers — useless.

So this is deliberately NOT a general answerability model. It is a small,
conservative, explainable check on **four concrete answer types** where a
mismatch is meaningful and cheap to detect. It is OFF by default and only ever
*removes* hits, never adds them.

What it does NOT do
-------------------
It cannot judge whether prose answers a question. Questions that ask for a
verdict, a reason, a comparison or an opinion are never filtered. A real
answerability layer needs a model (Phase 4); this narrows one specific failure
mode and nothing more.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Set

DATE = "date"
AMOUNT = "amount"
IDENTIFIER = "identifier"
QUANTITY = "quantity"

# Only these are enforced. PLACE/PERSON are detectable in principle but are too
# easy to get wrong (a claim can name a place in passing), so a false abstain
# there would silently delete a correct answer. Not enforced on purpose.
ENFORCED = (DATE, AMOUNT, IDENTIFIER, QUANTITY)

_MONTHS = (r"january|february|march|april|may|june|july|august|september|"
           r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
_DAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"

_DATE_RE = re.compile(
    rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}\s+(?:{_MONTHS})|(?:{_MONTHS})\b|"
    rf"(?:{_DAYS})\b|\b(?:19|20)\d{{2}}\b|today|tomorrow|yesterday|tonight)\b",
    re.I)

_AMOUNT_RE = re.compile(
    r"[$€£¥]\s?\d|\b\d+(?:[.,]\d+)?\s?(?:usd|eur|gbp|dollars?|euros?|pounds?|"
    r"cents?|per\s+(?:month|year|hour|week))\b|\busd|eur|gbp\b",
    re.I)

_IDENTIFIER_RE = re.compile(
    r"\b(?:[a-z]+[-_]?\d+|\d+[-_]?[a-z]+|[a-f0-9]{8,}|\bv\d+(?:\.\d+)*)\b", re.I)

# Ticket/flight style ids such as "BA 2491" or "SEV1". Deliberately
# case-SENSITIVE and upper-case-only: a bare number is NOT an identifier here,
# and neither is "May 2027". Letting dates count as identifiers would make an
# identifier question accept a date claim, which is exactly the failure this
# gate exists to catch.
_IDENTIFIER_UPPER_RE = re.compile(r"\b[A-Z]{1,4}[-_ ]?\d{2,5}\b")

_QUANTITY_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:gb|mb|tb|kb|ms|s|sec|seconds?|minutes?|hours?|days?|"
    r"weeks?|months?|years?|%|percent|items?|users?|pages?|runners?)\b", re.I)

# Words in the QUESTION that request each type. Kept narrow on purpose.
_ASKED = {
    DATE: re.compile(r"\bwhen\b|\bwhat (?:date|time|day|month|year)\b|"
                     r"\bwhich (?:date|day|month|year)\b|\bhow (?:long|soon)\b", re.I),
    AMOUNT: re.compile(r"\bhow much\b|\b(?:salary|wage|bonus|price|cost|fee|budget|"
                       r"revenue|invoice|payment|pay)\b", re.I),
    IDENTIFIER: re.compile(r"\b(?:number|id|identifier|code|version|serial|reference|"
                           r"ticket|hash|port)\b", re.I),
    QUANTITY: re.compile(r"\bhow many\b|\bhow much (?:ram|memory|storage|disk|space)\b|"
                         r"\b(?:how large|what size)\b", re.I),
}


def requested_types(question: str) -> Set[str]:
    """Answer types the question explicitly asks for (enforced set only)."""
    asked: List[str] = []
    for kind in ENFORCED:
        if _ASKED[kind].search(question or ""):
            asked.append(kind)
    # "how much" alone is ambiguous between AMOUNT and QUANTITY; if both fire,
    # accept either by keeping both (see _satisfied).
    return set(asked)


def provided_types(text: str) -> Set[str]:
    """Answer types actually present in a claim/evidence string."""
    text = text or ""
    have: Set[str] = set()
    if _DATE_RE.search(text):
        have.add(DATE)
    if _AMOUNT_RE.search(text):
        have.add(AMOUNT)
    if _IDENTIFIER_RE.search(text) or _IDENTIFIER_UPPER_RE.search(text):
        have.add(IDENTIFIER)
    if _QUANTITY_RE.search(text):
        have.add(QUANTITY)
    return have


def is_answerable(question: str, claim: str, evidence: str = "") -> bool:
    """True when the claim is not a confident answer-type mismatch.

    Conservative by construction: if the question asks for no recognised type,
    or the claim provides no recognised type at all, this returns True. It only
    abstains when the question asks for a specific type and the claim shows
    concrete evidence of a *different* type.
    """
    asked = requested_types(question)
    if not asked:
        return True
    have = provided_types(claim) | provided_types(evidence or "")
    if not have:
        # Nothing typed at all: do not guess. Prose may still answer.
        return True
    return bool(asked & have)


def filter_hits(question: str, hits: Sequence[dict]) -> List[dict]:
    """Drop hits that are a confident answer-type mismatch."""
    if not requested_types(question):
        return list(hits)
    kept = [h for h in hits
            if is_answerable(question, str(h.get("claim") or ""),
                             str(h.get("evidence") or ""))]
    return kept
