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
{"page": "<page slug>", "cite": "[[<page slug>]]", "accepted": "YYYY-MM-DD",
 "claim": "...", "evidence": "...", "kind": "...", "peer": "...", "score": 0.71}
```

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

## 9. Invariants this schema guarantees

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
