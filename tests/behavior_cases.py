"""Frozen fixtures for the offline behavioural suite (Phase 1).

**Preregistered before tuning.** The dev set may be used to diagnose and fix
behaviour; the holdout set must not influence implementation choices. If a
holdout case fails, fix the cause *generically* and record that the holdout was
consulted — do not special-case it.

Dev and holdout use disjoint domains and disjoint wording so that a change tuned
on dev is not silently rewarded on holdout. Dev is deliberately larger; holdout
is small and read-once.

Scope note: this suite grades **retrieval identity, currency and rank**, plus
store-level provenance/safety invariants. It does not grade generated prose,
because no answer model is invoked offline. Answer-generation grading belongs to
Phase 4 when a real local model is in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from behavior_graders import Expect

# Two different numbers, deliberately kept apart.
#
# QUALITY_TARGET is the aspirational bar the product should reach on this suite.
# It IS met when the opt-in answer-type gate is enabled (Phase 2): onnx holdout
# 10/11 = 91% vs the 85% target. With the gate off (the default retrieval path)
# the holdout sits at the regression floor instead. Both are asserted.
# Do not lower it to make a run go green.
QUALITY_TARGET = {
    "onnx": {"dev": 0.90, "holdout": 0.85},
    "hash": {"dev": 0.70, "holdout": 0.65},
}

# The gate is opt-in, so the target above is only claimable with it switched on.
# This flag name must match the env var read by ProposalStore.__init__.
ANSWERABILITY_ENV = "HWIKI_ANSWERABILITY"

# REGRESSION_FLOOR is measured once from the baseline run and then acts as a
# "never get worse" guard. Each value is the measured rate rounded DOWN to two
# decimals (8/11 = 0.727 -> 0.72), so the baseline run itself always passes and
# any real drop trips the suite. Matching a floor only means "no regression";
# it never means the quality target was reached.
REGRESSION_FLOOR = {
    "onnx": {"dev": 1.00, "holdout": 0.72},
    "hash": {"dev": 0.75, "holdout": 0.54},
}


@dataclass
class FactSpec:
    kind: str
    claim: str
    evidence: str = "stated by the user"
    project: str = ""
    peer: str = "user"


@dataclass
class Case:
    id: str
    category: str
    query: str
    expect: Expect
    project: str = ""
    peer: str = ""


@dataclass
class Story:
    id: str
    set: str                      # "dev" | "holdout"
    facts: List[FactSpec]
    cases: List[Case]
    # (new_index, old_index) — old is superseded by new
    supersedes: List[Tuple[int, int]] = field(default_factory=list)
    # claims that must raise ProposalRejected at propose time
    expect_reject: List[str] = field(default_factory=list)
    # (a_index, b_index) — both remain accepted, neither is superseded
    unresolved_conflicts: List[Tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DEV
# ---------------------------------------------------------------------------

EDITOR_STYLE = Story(
    id="editor-style",
    set="dev",
    facts=[
        FactSpec("preference", "User's main editor is VS Code"),
        FactSpec("preference", "User wants short chat replies with no progress pings"),
        FactSpec("preference", "User prefers dark mode in every tool"),
        FactSpec("preference", "User takes notes in Obsidian daily"),
    ],
    cases=[
        Case("editor-direct", "direct",
             "what editor does the user use",
             Expect(needs_any=["vs code"], top1=True)),
        Case("editor-paraphrase", "paraphrase",
             "which program do they write code in",
             Expect(needs_any=["vs code"])),
        Case("replies-direct", "direct",
             "how should chat replies be formatted",
             Expect(needs_any=["short"])),
        Case("dark-paraphrase", "paraphrase",
             "do they like light or dark interfaces",
             Expect(needs_any=["dark mode"])),
        Case("notes-paraphrase", "paraphrase",
             "what tool collects their daily journal",
             Expect(needs_any=["obsidian"])),
        Case("unknown-food", "refusal",
             "what is the user's favourite food",
             Expect(refuse=True)),
    ],
)

INFRA_MIGRATION = Story(
    id="infra-migration",
    set="dev",
    facts=[
        # 0 current, 1 is the stale plan it replaces
        FactSpec("environment", "Staging server stays on Fly.io", project="web"),
        FactSpec("environment", "Staging server is moving to Hetzner VPS", project="web"),
        FactSpec("environment", "Production runs on a Hetzner VPS", project="web"),
        FactSpec("environment", "Staging database runs Postgres 15", project="web"),
        FactSpec("environment", "Backups run nightly with restic to B2", project="ops"),
    ],
    supersedes=[(0, 1)],
    cases=[
        Case("staging-current", "reversal",
             "where does the staging server run now",
             Expect(needs_any=["fly.io"], forbid_current=["moving to hetzner"])),
        Case("staging-paraphrase", "paraphrase",
             "who hosts the test environment",
             # Forbid the *stale claim*, not the word "hetzner": production
             # legitimately runs on Hetzner and may match this query. Forbidding
             # the bare host name made a correct answer look like a stale one.
             Expect(needs_any=["fly.io"], forbid_current=["moving to hetzner"])),
        Case("staging-not-migrating", "negation",
             "is the staging server moving to another provider",
             Expect(needs_any=["fly.io"], forbid_current=["moving to hetzner"])),
        Case("prod-hetzner", "direct",
             "where does production run",
             Expect(needs_any=["hetzner"])),
        Case("db-postgres", "direct",
             "which database does staging use",
             Expect(needs_any=["postgres"])),
        Case("backup-tool", "direct",
             "what tool do backups use",
             Expect(needs_any=["restic"])),
        Case("unknown-blood", "refusal",
             "what is the user's blood type",
             Expect(refuse=True)),
    ],
)

SAFETY_BOUNDARIES = Story(
    id="safety-boundaries",
    set="dev",
    facts=[
        FactSpec("blocker", "Never paste API keys or secrets into chat"),
        FactSpec("preference", "User prefers terse status updates"),
    ],
    expect_reject=[
        "Set the deploy token with password=hunter2024 in the config",
    ],
    cases=[
        Case("secret-rule", "direct",
             "what is the rule about secrets in chat",
             Expect(needs_any=["api keys"])),
        Case("terse", "direct",
             "how much detail does the user want in updates",
             Expect(needs_any=["terse"])),
        Case("unknown-pets", "refusal",
             "what pets does the user own",
             Expect(refuse=True)),
    ],
)

# ---------------------------------------------------------------------------
# HOLDOUT — frozen, read once, never used to tune
# ---------------------------------------------------------------------------

TRAVEL_PLAN = Story(
    id="travel-plan",
    set="holdout",
    facts=[
        FactSpec("milestone", "The Lisbon trip now starts in May 2027", project="travel"),
        FactSpec("milestone", "The Lisbon trip was originally booked for April 2027", project="travel"),
        FactSpec("decision", "The Lisbon hotel is booked on a refundable rate", project="travel"),
        FactSpec("preference", "User prefers aisle seats on long flights", project="travel"),
    ],
    supersedes=[(0, 1)],
    cases=[
        Case("trip-when", "reversal",
             "when does the Lisbon trip start",
             Expect(needs_any=["may"], forbid_current=["april"])),
        Case("trip-when-paraphrase", "paraphrase",
             "what month is the Portugal holiday",
             Expect(needs_any=["may"], forbid_current=["april"])),
        Case("hotel-terms", "direct",
             "what are the hotel booking terms",
             Expect(needs_any=["refundable"])),
        Case("seat-pref", "paraphrase",
             "where do they like to sit on a plane",
             Expect(needs_any=["aisle"])),
        Case("unknown-flight", "refusal",
             "what is the flight number for the Lisbon trip",
             Expect(refuse=True)),
    ],
)

TEAM_OPS = Story(
    id="team-ops",
    set="holdout",
    facts=[
        FactSpec("environment", "The on-call rotation changes every Monday", project="ops"),
        FactSpec("blocker", "No production deploys on Fridays", project="ops"),
        FactSpec("decision", "Incidents are graded from SEV1 to SEV4", project="ops"),
        FactSpec("environment", "Runbooks live in the ops vault folder", project="ops"),
    ],
    cases=[
        Case("rotation", "direct",
             "how often does on-call rotate",
             Expect(needs_any=["monday"])),
        Case("deploy-freeze", "direct",
             "when are production deploys forbidden",
             Expect(needs_any=["friday"])),
        Case("severity", "paraphrase",
             "how are incidents classified",
             Expect(needs_any=["sev1"])),
        Case("runbooks", "paraphrase",
             "where are the operational procedures kept",
             Expect(needs_any=["runbook"])),
        Case("overview", "broad",
             "summarise the ops process",
             Expect(min_distinct=3)),
        Case("unknown-salary", "refusal",
             "what is the on-call salary bonus",
             Expect(refuse=True)),
    ],
)

DEV_STORIES = [EDITOR_STYLE, INFRA_MIGRATION, SAFETY_BOUNDARIES]
HOLDOUT_STORIES = [TRAVEL_PLAN, TEAM_OPS]
ALL_STORIES = DEV_STORIES + HOLDOUT_STORIES


def stories_for(set_name: str) -> List[Story]:
    if set_name == "dev":
        return DEV_STORIES
    if set_name == "holdout":
        return HOLDOUT_STORIES
    return list(ALL_STORIES)
