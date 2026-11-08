# Wiki memory schema — v1

**Status:** stable · **Applies to:** hwiki v2.1 · **Source of truth:** the Obsidian
vault, not the SQLite index.

This is the agent-facing contract for what a memory page *is*, what may be
believed about it, and what a caller must and must not do with it. It exists
because the LLM Wiki pattern is only useful if a fact can be traced, challenged and
reversed — a page that merely *looks* like knowledge is worse than no page.

---

## 1. Reading a page

A provider-written page has this shape:

```markdown
---
title: Staging server stays on Fly.io
spine: memory
type: memory
status: accepted              # accepted | superseded
extractor: hwiki-extractor    # marker: written by the provider, not by hand
proposal_id: 1f0c…            # stable id, survives rebuild
kind: environment             # preference | decision | fact | environment | …
peer: user                    # user | assistant
project: web
accepted_at: 2026-11-08
supersedes: "[[fact-staging-server-is-moving-to-hetzner-vps-…]]"   # optional
superseded_by: "[[…]]"                                             # optional
tags: [memory, hwiki]
---

# Staging server stays on Fly.io

> **Evidence:** stated by the user in session 2026-11-08

<!-- hwiki:compiled:start -->
### Staging also runs the queue worker
…merged claim…
<!-- hwiki:compiled:end -->
```

**`extractor: hwiki-extractor` is the trust marker.** Pages without it are
hand-written notes that happen to live in the memory zone. They are never imported
by a rebuild and never graded for schema — they are not facts, and the system will
not claim they are.

## 2. Source trust

There is exactly one source class: **the user, or the assistant, stating something
in a session** (recorded in `peer`). Nothing else is ingested. There is no web
scraping, no external API and no inferred source trust score.

Consequences a caller must respect:

* `peer` is the only authorship signal. `peer: assistant` means the assistant
  proposed it about itself; it is not a user preference.
* `evidence` is a short quote or paraphrase, **not** a transcript. Raw conversation
  is never stored.
* A fact is not a citation of a document. Do not render it as one.

## 3. Approval

Nothing enters recall without passing the review gate. The lifecycle is:

```
propose ──accept──▶ accepted ──supersede──▶ superseded
   │                    │
   └──reject──▶ rejected└──page deleted──▶ deleted_by_edit
```

* `proposed` rows are **not** recallable. They are a queue for the human.
* `accepted` is the only state that recall reads.
* `rejected` is terminal for that proposal id.
* `superseded` stays on disk as a tombstone with a wikilink forward. It is never
  returned by recall and never resurrected by a rebuild.

Compile (merging a new fact into the page that already owns the topic) runs
**strictly after** approval and never changes any of the above states.

## 4. Provenance

| field | meaning | if missing |
|---|---|---|
| `proposal_id` | stable id, survives rebuild | rebuild derives a stable id from the path; lint reports `info` |
| `kind` | what type of fact | **unknown** — lint warns; the value is never guessed |
| `peer` | who said it | treated as `user` (documented default, stated here) |
| `project` | scope | unscoped (`""`), matching every project |
| `accepted_at` | when it was approved | **unknown** — lint warns; never back-filled from file mtime |

**Rule: missing provenance is reported as unknown, never invented.** Lint
(`hermes hwiki lint`) is the mechanism.

Lineage is carried in `supersedes` / `superseded_by` wikilinks. A claim may
legitimately still contain the old value in a `↻ replaced:` annotation — that is
history, not a current answer, and graders must not mistake one for the other.

## 5. Index and log

* `index.md` — generated catalog in the **wiki root** (never inside `memory_dir`).
  Summaries are deterministic string truncation; no model is called.
* `log.md` — append-only timeline of writes.

Both are in `demote_paths`: they list titles, so they must never outrank the page
that actually answers a query. Locked by `tests/test_catalog_recall.py`.

Neither is authoritative. If `index.md` disagrees with the tree, the tree wins and
lint says so (`catalog.entry_missing` / `catalog.stale`).

## 6. Query citations

Every recall hit carries:

```python
{"id": "1f0c…",                # stable source id = proposal_id
 "page": "<page slug>", "cite": "[[<page slug>]]", "accepted": "YYYY-MM-DD",
 "claim": "...", "evidence": "...", "kind": "...", "peer": "...", "score": 0.71}
```

`id` is the **stable approved source id**: it survives a rebuild because it is
written into the page frontmatter as `proposal_id`. Cite it when a durable
reference is needed; cite `[[page]]` when a human needs to click through.

A caller that reports a fact **must** be able to name its page. Only the slug is
exposed — never the absolute path, which may contain a home directory. A fact with
no page yet has `cite: ""`; treat that as uncitable and say so rather than
inventing a link.

Two limits, stated plainly:

* The citation marks *where the claim came from*, not that the claim answers the
  question.
* Retrieval answers "is this related?". It does not answer "is this the answer?".
  The optional answer-type gate (`HWIKI_ANSWERABILITY=1`) narrows one specific
  mismatch (date/amount/identifier/quantity) and nothing else.

## 7. Review

Review is a human step, not an automatic one. An agent may:

* propose facts,
* flag conflicts, duplicates and stale pages,
* file a synthesised answer under `answers/` (never as an accepted fact).

An agent may **not** accept its own proposal, mark something superseded to make a
test pass, or hand-edit a page to change a fact without the change coming back
through `sync_from_vault` (which re-validates and re-embeds).

## 8. Lint checks (v1)
`hermes hwiki lint [--json] [--info]` — read-only, deterministic, never repairs.

| check | severity | meaning |
|---|---|---|
| `secret.content` | error | secret-shaped material on a memory page (checked even on hand-written notes) |
| `claim.missing` | error | provider page with no `# claim` heading |
| `link.dangling` | error | wikilink that resolves to nothing |
| `page.unreadable` | error | file could not be read |
| `frontmatter.incomplete` | warn | `kind`/`status`/`accepted_at` missing — provenance unknown |
| `supersede.orphan` | warn | `superseded_by` points at a missing page |
| `supersede.incomplete` | warn | superseded but names no replacement |
| `duplicate.claim` | warn | identical claim accepted on more than one page |
| `catalog.stale` | warn | `index.md` lists a page that is gone |
| `catalog.entry_missing` | warn | a memory page is not in `index.md` |
| `page.oversized` | info | page beyond 24 000 chars; consider splitting |
| `answer.unreferenced` | info | filed answer no fact points at |

## 9. Ingest planning (read-only)

`hermes hwiki plan --from <file>` (or stdin) answers "what *would* happen to these
claims?" without doing it. For each candidate it predicts one of
`near_duplicate`, `conflict`, `merge_into_owner`, `new_page`, or
`refused_secret`, naming the owning page and the score where relevant.

**It writes nothing.** No proposal rows, no pages, no catalog, no log, and the
facts database is not modified — asserted by test, not by inspection. If the facts
database does not exist the command refuses rather than creating one as a side
effect.

It is **not an extractor**: candidate claims are split naively on sentence and
bullet boundaries without a model. A caller with a real extractor should pass the
claims it already has. A plan is a prediction, not a promise — the review gate can
still reject, and a human can reword.

## 10. Recall confidence (advisory, not a verdict)

`store.recall_confidence(query)` returns `best`, `margin`, `overlap`,
`candidates`, `floor`, `confident`, and a fixed `caveat`. It is a **pure
observation**: it never changes what `recall` returns.

`confident` means only "this retrieval is not empty and not ambiguous". It does
**not** mean the fact answers the question. Two deterministic attempts to derive
answerability were measured and failed; a real verdict needs a model. Callers that
surface `confident` must surface `caveat` with it.

Available from the command line for a caller that wants to decide before speaking:

```
hermes hwiki confidence "where does staging run" [--json]
```

It prints `confident`, `best` vs `floor`, `margin`, `overlap`, the candidate count
and the caveat. `--json` emits exactly the seven documented keys.

The same signal is a **read-only tool**, `wiki_memory_confidence`, advertised in every
context including cron and subagent runs (it writes nothing, so the read-only gate
does not cover it). It returns the seven keys plus an `advice` string instructing the
caller to abstain and ask rather than guess. The tool is the point: an agent that
cannot see the signal cannot choose to abstain.

## 11. Filing answers back

`hermes hwiki file -q "<question>" [--from answer.md] [--tags a,b] [--json]`
(answering on stdin otherwise) writes into `<memory_dir>/answers/`. It shares one
implementation with the `wiki_memory_file` tool (`facts.file_answer`), so the page
format cannot drift between the two entry points.

Guarantees: secret-scanned and refused; idempotent per question; a slug collision
between two *different* questions gives the newcomer a `-<digest>` sister file
rather than overwriting; answers stay out of accepted-fact recall and out of the
review queue.

## 12. Optional NLI tier (advisory annotation)

`hwiki/nli.py` loads a bundled MNLI cross-encoder (DeBERTa-v3-small, int8 ONNX) and
`hwiki/verdict.py` turns it into a fact-pair relation. It is **opt-in**
(`HWIKI_NLI_CONFLICT=1`) and it is an **annotation**, not a verdict:

* `possible_conflict` stays negation-only, so the flag keeps its high precision and
the default path is byte-for-byte unchanged.
* With the tier on, each `related` candidate gains `relation`
(`contradiction` / `restatement` / `compatible`) plus the raw probabilities.
* Measured: 7/7 genuine contradictions found where the token rules find 1/7, at the
cost of 3/14 compatible pairs wrongly read as contradictory. Both numbers travel
with the feature.
* `HWIKI_NLI_DUP=1` additionally treats a **mutually** entailing pair as the same fact.
One-directional entailment is a refinement, not a duplicate, and merging it drops
detail (measured: 1 wrong merge in 4 before requiring both directions, 0 after).
* It cannot answer "does this fact answer this question?" — a question is not a
hypothesis, and entailment(question, claim) measured zero signal. Do not rebuild that.

Failure mode is explicit: a missing bundle raises `NliUnavailable`, callers keep the
previous behaviour, and nothing is guessed.

### Lint checks

`hermes hwiki lint [--semantic] [--info]` reports, in one read-only pass: missing
frontmatter, a page with no `# claim`, supersession links that point nowhere, dangling
`[[wikilinks]]`, the same claim accepted on two pages, catalog drift in both directions,
orphan pages (`link.orphan`, INFO: no inbound link and no `index.md` entry), unreferenced
filed answers, and secret-shaped content.

An orphan is INFO rather than a warning: it is not rot, it is a page nothing navigates to.
Generated artifacts and filed answers are exempt — they are entry points by design.

## 13. Autonomous maintenance

`hermes hwiki maintain` (and the scheduled tick behind the turn hook) sweeps memory
for work that a human should look at:

| job | writes |
|---|---|
| `heal` | proposal rows recovered from extractor-written vault pages |
| `conflicts` | nothing — reported on the maintenance page |
| `duplicates` | nothing — reported |
| `questions` | `meta.open_questions` |
| `artifacts` | `index.md` when lint reports staleness |

Its authority is deliberately narrow:

* It **never** accepts, supersedes, merges or deletes a fact. Invariant 3 is unchanged:
  the review gate remains the only way memory changes meaning.
* It never invents provenance. Pages that are not extractor-written are skipped by
  `heal`, not guessed at.
* It writes exactly one derived page, `memory_dir/maintenance.md`. It carries no
  `# claim` heading, so it is never a fact — and its concrete path is **hard-excluded**
  by `config.excluded_paths()` (not just `demote_paths`), because a demoted page still
  came back for an ops query and the report reads like authority.
* It is bounded: cheapest jobs first, then a wall-clock budget, then it stops and says
  what it skipped.
* Unanswered questions are buffered **in memory** during recall (so recall stays a
  pure read) and flushed to meta on the next turn.
* A recorded question whose vocabulary contains the current query is surfaced in the
  prefetch block as *"Memory gaps this looks like (ask, do not assume)"*. It is
  presented as a question to ask, never as a fact to assert — surfacing a gap must not
  become inventing an answer.
* Contradiction pairs are stored in `meta.maintenance_conflicts` and surfaced the same
  way, with both claims quoted and neither ranked. A pair that a human resolves
  disappears on the next tick, because the list is replaced wholesale.
* The read path never re-runs a sweep to answer a prefetch: it reads two meta keys.

## 14. Invariants this schema guarantees

1. The vault is authoritative; the SQLite index is disposable and rebuildable.
2. `store.py` never writes outside `<root>/<memory_dir>`.
3. Nothing is recallable without passing the review gate.
4. No external API is called on the retrieval path.
5. `raw/` is excluded from recall, index and read.
6. Secret material is refused on write and reported by lint on read.
7. A superseded claim is never presented as current.
8. Missing provenance is unknown, never invented.
9. Compiled sections remain recallable as their own claims.

Changing any of 1–9 is a schema **v2** change, not a patch.
