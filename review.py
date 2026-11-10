"""Risk split for the review gate.

Every proposed fact currently waits for an explicit human accept. That is safe and it
is also why the store stays empty: nothing gets approved, so the assistant falls back
to conversation history.

This module decides which facts may skip the queue. The rule is deliberately lopsided
— **when in doubt, review** — because the two mistakes are not equal:

* auto-accepting something trivial that turns out slightly wrong: you correct it.
* auto-accepting a decision, a contradiction, or a credential: the assistant now
  believes something you never agreed to, and the evidence of how it got there is
  a page you did not read.

So the allow-list is short and the deny-list is not.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

AUTO_ACCEPT = "auto_accept"
REVIEW_REQUIRED = "review_required"

#: Kinds cheap enough to get wrong: they describe the user or the machine, not a
#: course of action. Anything absent from this tuple requires review.
AUTO_OK_KINDS = ("preference", "environment")

#: Words that mean a claim is about money, access, or identity. A false positive here
#: costs one extra approval; a false negative costs trust.
_SENSITIVE = re.compile(
    r"\b(api[\s_-]?key|secret|token|password|passphrase|credential|"
    r"card|iban|bank|account|routing|ssn|passport|licen[cs]e|"
    r"private[\s_-]?key|auth|login|2fa|otp)\b",
    re.IGNORECASE,
)


def classify_risk(proposal: Dict[str, Any], *, conflict: bool,
                  duplicate: bool, related: int = 0) -> Tuple[str, str]:
    """Return ``(verdict, reason)`` for one proposal.

    ``conflict``, ``duplicate`` and ``related`` are the signals the propose path
    already computes; passing them in keeps this function pure and testable, and
    keeps the decision in one place rather than spread through the write path.

    ``related`` is the important one. The conflict detector only recognises
    *negation* ("does not prefer X"), so a slot-value contradiction such as
    "prefers dark mode" vs "prefers light mode" is flagged as neither a conflict nor
    a duplicate — measured, see ``benchmarks/phase4-floor-shape.md``. Auto-accepting
    that would silently overwrite a belief the user approved. So a claim that sits
    close to anything already accepted goes to review regardless of how harmless its
    kind looks: the gate only opens for facts that stand on their own.
    """
    kind = str(proposal.get("kind") or "").strip().lower()
    claim = " ".join(str(proposal.get("claim") or "").split())

    if not claim:
        return REVIEW_REQUIRED, "empty claim"
    if conflict:
        return REVIEW_REQUIRED, "flagged as a conflict with an accepted fact"
    if duplicate:
        return REVIEW_REQUIRED, "near-duplicate of an accepted fact"
    if related:
        return REVIEW_REQUIRED, (
            f"close to {related} accepted fact(s), so a contradiction could hide here")
    if kind not in AUTO_OK_KINDS:
        return REVIEW_REQUIRED, f"kind {kind!r} needs review"
    if _SENSITIVE.search(claim):
        return REVIEW_REQUIRED, "mentions credentials, accounts or money"
    return AUTO_ACCEPT, f"{kind}, no conflict, no duplicate, unrelated to stored facts"


def is_auto_acceptable(config: Dict[str, Any],
                       proposal: Dict[str, Any]) -> bool:
    """True only when the operator opted in AND the proposal is low risk.

    Deliberately requires ``auto_accept: true`` in config: silently switching the
    gate off for everyone who upgrades is exactly the kind of behaviour a memory
    system should not have.
    """
    if not (config or {}).get("auto_accept"):
        return False
    verdict, _reason = classify_risk(proposal, conflict=False, duplicate=False)
    return verdict == AUTO_ACCEPT
