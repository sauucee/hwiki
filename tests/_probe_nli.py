"""Does a local NLI model answer the question thresholds could not? (Phase 4)

Two different questions get confused under "answerability". This probe separates
them and measures each, because the first framing does not work:

  (A) does this fact ANSWER this question?      -> NLI cannot do this
      A question is not a hypothesis, so entailment(question, claim) is
      ~0 for correct answers and the contradiction head fires on every pair.
      Measured: answered n=20 median entailment 0.000, p90 0.005; refusal
      median 0.000. Separation +0.00 — i.e. no signal at all. Rejected as a
      formulation, not as a model.

  (B) what is the relation between two FACTS?   -> this is exactly MNLI's task
      entailment  == same claim restated   (duplicate/merge candidate)
      contradiction == incompatible claim   (conflict candidate)
      neutral     == compatible refinement  (keep both)

(B) is what the token heuristics kept getting wrong: a slot-value divergence rule
was rejected because it flagged "runs on Fly.io" against "runs Postgres" — two
different slots, both true. A model that reads the predicate should not.

Run:  python3 tests/_probe_nli.py [--pairs]
"""

from __future__ import annotations

import argparse
import sys

import _hwiki_path  # noqa: F401,E402

from hwiki.nli import NliScorer, nli_model_dir  # noqa: E402

# (a, b, expected relation, why it is in the set)
PAIRS = [
    # --- genuine contradictions -------------------------------------------------
    ("Deploys are allowed on Fridays", "Deploys are never allowed on Fridays",
     "contradiction", "explicit negation"),
    ("Staging runs on Fly.io", "Staging runs on Hetzner",
     "contradiction", "swapped value, one slot"),
    ("The database is Postgres 15", "The database is MySQL 8",
     "contradiction", "swapped engine and version"),
    ("Production runs on a Hetzner VPS", "Production runs on Fly.io",
     "contradiction", "swapped host"),
    ("The Lisbon trip starts in May 2027", "The Lisbon trip starts in June 2026",
     "contradiction", "swapped date"),
    ("The on-call rotation changes every Monday", "The on-call rotation changes every Friday",
     "contradiction", "swapped day"),
    ("Incidents are graded from SEV1 to SEV4", "Incidents are graded from P1 to P4",
     "contradiction", "swapped scheme"),
    # --- paraphrases: same claim, so a duplicate/merge --------------------------
    ("User prefers terse status updates", "User prefers short status updates",
     "entailment", "synonym pair that similarity missed"),
    ("User's main editor is VS Code", "They write code in VS Code",
     "entailment", "reworded claim"),
    ("Backups run nightly with restic to B2", "Nightly backups go to B2 with restic",
     "entailment", "reordered claim"),
    ("Staging server stays on Fly.io", "We are not migrating staging off Fly.io",
     "entailment", "negation of a change, not of the fact"),
    # --- compatible refinements: BOTH true, must stay neutral -------------------
    ("The staging server runs on Fly.io", "The staging server runs Postgres",
     "neutral", "different slot — the pair that killed the token heuristic"),
    ("Staging server stays on Fly.io", "Staging server stays on Fly.io and runs the queue worker",
     "neutral", "an addition"),
    ("Staging is on Fly.io", "Staging is on Fly.io in the EU region",
     "neutral", "a refinement of the same value"),
    ("User prefers dark mode in every tool", "User prefers dark mode in every editor",
     "neutral", "near-identical, narrower scope"),
    ("Backups run nightly with restic to B2", "Backups are verified monthly",
     "neutral", "same topic, different claim"),
    ("The user takes notes in Obsidian", "The user reviews notes on Sundays",
     "neutral", "same topic, different claim"),
    ("Runbooks live in the ops vault folder", "Runbooks are reviewed each quarter",
     "neutral", "same topic, different claim"),
    ("The user prefers aisle seats on long flights", "The user prefers window seats on short flights",
     "neutral", "different scope, not a contradiction of the same fact"),
    # --- unrelated: must not become any relation -------------------------------
    ("The user takes notes in Obsidian", "Backups run nightly with restic to B2",
     "unrelated", "different topics"),
    ("Incidents are graded from SEV1 to SEV4", "The Lisbon hotel is booked on a refundable rate",
     "unrelated", "different topics"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="store_true", help="(default) measure fact-pair relations")
    ap.add_argument("--json", action="store_true")
    ap.parse_args()

    model_dir = nli_model_dir()
    if not model_dir.is_dir():
        print(f"  no NLI model at {model_dir}", file=sys.stderr)
        return 2
    scorer = NliScorer(model_dir)

    rows = []
    for a, b, expected, why in PAIRS:
        con, ent, neu = scorer.probabilities(a, b)
        got = max((("contradiction", con), ("entailment", ent), ("neutral", neu)),
                  key=lambda p: p[1])[0]
        rows.append((a, b, expected, got, con, ent, neu, why))

    ok = sum(1 for r in rows if r[3] == r[2] or
             (r[2] == "unrelated" and r[3] == "neutral" and r[6] > 0.6))
    print(f"  fact-pair relations: {ok}/{len(rows)} as expected\n")
    for a, b, expected, got, con, ent, neu, why in rows:
        mark = "ok  " if (got == expected or
                          (expected == "unrelated" and got == "neutral" and neu > 0.6)) else "MISS"
        print(f"  {mark} want={expected:12s} got={got:13s} "
              f"c={con:.2f} e={ent:.2f} n={neu:.2f}  {why}")
        print(f"        {a[:56]!r} / {b[:56]!r}")

    # The decisive comparison: the same pairs through the shipped token rules.
    from hwiki.facts import _negation_signal

    print("\n  what the shipped (token) rules say about the same pairs:")
    for a, b, expected, got, con, ent, neu, why in rows:
        neg = _negation_signal(a) != _negation_signal(b)
        verdict = "contradiction" if neg else "not flagged"
        agree = "ok  " if ((expected == "contradiction") == neg) else "MISS"
        print(f"    {agree} {verdict:14s} want={expected:12s}  {why}")

    if args.json:
        import json
        print(json.dumps([{"a": a, "b": b, "expected": e, "got": g,
                           "contradiction": c, "entailment": en, "neutral": n, "why": w}
                          for a, b, e, g, c, en, n, w in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
