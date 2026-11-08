# Configuration

Configuration lives in `$HERMES_HOME/hwiki.json` (`$HERMES_HOME` is `~/.hermes`, or
`~/.hermes/profiles/<name>` for a profile). Only `wiki_path` is required.
`hermes memory setup` writes this file for you.

## All options

| Key | Default | Description |
|---|---|---|
| `wiki_path` | | **Required.** Wiki root. `$HERMES_HOME` and `~` expand. |
| `backend` | `fts5` | `fts5` or `qmd` (`qmd` builds a hybrid with FTS5 fallback). |
| `memory_dir` | `wiki/memory` | Wiki-relative folder the agent may write to. The only writable path. |
| `untrusted_dirs` | `["raw"]` | Never indexed, recalled or readable through memory tools. |
| `include` | `["**/*.md"]` | Root-relative globs to index. |
| `exclude` | `[]` | Root-relative globs to skip (`.git`, `node_modules`, `.obsidian` and similar are always skipped). |
| `index_path` | `$HERMES_HOME/hwiki-index.sqlite` | FTS5 index location. Disposable. |
| `prefetch_limit` | `4` | Excerpts auto-injected per turn. |
| `prefetch_max_chars` | `1400` | Hard cap on injected characters per turn. |
| `auto_recall` | `true` | `false` = tools only, no automatic injection. |
| `min_query_chars` | `8` | Skip recall for trivially short messages. |
| `reindex_interval_s` | `60` | Minimum seconds between incremental reindexes. |
| `max_hits_per_page` | `2` | Per-page chunk cap when ranking results. |
| `demote_paths` | `["log.md", "index.md", "wiki/memory/builtin-mirror.md"]` | Pages ranked after primary results. Suffix-matched, so a relocated copy stays demoted. |
| `mirror_builtin_memory` | `true` | Mirror built-in `memory` tool additions into the wiki. |
| `mirror_page` | `builtin-mirror` | Memory page receiving those mirrors. |
| `catalog_enabled` | `true` | Maintain `index.md` (page catalog) and `log.md` (append-only timeline) in the wiki root. Disabled automatically in read-only agent contexts. |
| `index_filename` | `index.md` | Name of the generated catalog. Written to the wiki root, never into `memory_dir`. |
| `log_filename` | `log.md` | Name of the generated event log. |
| `compile_enabled` | `true` | Merge an accepted fact into the page that already owns the topic, as a delimited `###` section, instead of creating a near-duplicate page. Runs strictly after the review gate; any failure leaves the standalone page intact. Conflicts are never merged. Set `false` to reproduce v2.0 behaviour. Disabled in read-only contexts. |
| `qmd_binary` | `qmd` | Path or name of the qmd executable. |
| `qmd_collection` | `hwiki` | qmd collection name. |
| `qmd_mode` | `search` | `search` (BM25), `vsearch` (vector), `query` (hybrid + rerank). |
| `qmd_timeout_s` | `6.0` | Per-query qmd budget before falling back to FTS5. |

### Environment overrides

Environment variables win over the file: `HWIKI_PATH`, `HWIKI_BACKEND`,
`HWIKI_MEMORY_DIR`, `HWIKI_QMD_BINARY`, `HWIKI_QMD_MODE`,
`HWIKI_ANSWERABILITY`, `HWIKI_NLI_CONFLICT`, `HWIKI_NLI_DUP`, and `HWIKI_NLI_DIR`.

### Answer-type gate (`HWIKI_ANSWERABILITY`, off by default)

Recall answers *"is this fact related?"*. Sometimes the user asked *"is this fact
the answer?"* — e.g. a question about a flight number scoring 0.78 against a fact
about trip dates. `HWIKI_ANSWERABILITY=1` enables a narrow abstention gate over
four answer types: **date**, **amount**, **identifier**, **quantity**.

It abstains only when the question asks for one of those and the claim supplies a
*different* one. It is conservative by design and only ever **removes** hits:

- a question asking for no recognised type is never filtered;
- a claim supplying no recognised type at all is never filtered (prose can still
  be the answer);
- a bare number does **not** count as an identifier — otherwise "May 2027" would
  satisfy a flight-number question;
- place and person are detected but never enforced, because a claim can name one
  in passing and a false abstain deletes a correct answer.

Measured effect on the frozen behavioural suite: onnx holdout 73% → 91%, dev
unchanged at 100%, and the 40-case recall corpus unchanged at 38/40. It is a
heuristic for one failure mode, **not** general answerability: verdicts, reasons
and comparisons are untouched. See `benchmarks/phase2-answerability.md` in the
project bundle for the measurements and the two rejected hypotheses.

Also settable per-store (`ProposalStore(db, answerability=True)`) and per-call
(`recall(query, answerability=True)`); an explicit argument beats the env var.

### NLI second opinion (`HWIKI_NLI_CONFLICT`, off by default)

An optional local MNLI cross-encoder (DeBERTa-v3-small, int8 ONNX, ~165 MB) that
reads the *relation between two facts*. `HWIKI_NLI_CONFLICT=1` annotates each
`related` candidate with `relation` (`contradiction` / `restatement` /
`compatible`) and the raw probabilities. `HWIKI_NLI_DIR` points at the bundle
(default `<hermes_home>/plugins-data/hwiki-extractor/models/nli-deberta-small`).

It deliberately **does not** set `possible_conflict`. Measured on a labelled
21-pair set: the shipped negation rule catches 1 of 7 genuine contradictions with
no false positives; the model catches 7 of 7 but also calls 3 of 14 compatible
pairs contradictory (different slots, same subject, different scope). So the flag
stays precise and the model adds recall under its own label, for a human to read.

It also cannot answer the original question — "does this fact *answer* this
question?" — because a question is not a hypothesis: entailment(question, claim)
measured 0.000 median for correct answers and 0.000 for refusals. That formulation
is rejected and recorded so it is not retried.

### Paraphrase duplicates (`HWIKI_NLI_DUP`, off by default)

Similarity alone misses paraphrases: "User prefers terse status updates" against
"… short status updates" blends to 0.887, just under the 0.90 duplicate bar.
`HWIKI_NLI_DUP=1` asks the model, and treats the pair as the same fact when **each
claim entails the other**.

Mutual entailment matters. One-directional entailment is not paraphrase — a specific
claim entails a general one ("Staging is on Fly.io in the EU region" entails "Staging
is on Fly.io" at 0.979) and merging that would silently drop the detail. Measured:
one-directional produced one wrong merge in four; mutual entailment produced none.

    paraphrases deduplicated     default 1/4   HWIKI_NLI_DUP=1  2/4
    wrong merges                 0/4                         0/4

The model is only consulted for candidates in the 0.75–0.90 similarity band, and not
at all when similarity has already decided.

No network at runtime; the model is on disk. If the bundle is missing every entry
point raises `NliUnavailable` and callers keep the previous behaviour. See
`benchmarks/phase4-nli-verdict.md`.

### Maintenance (`maintenance`, on by default)

```json
{ "maintenance": { "interval_hours": 24, "budget_ms": 2000 } }
```

One bounded tick per interval, run from the turn hook and also available as
`hermes hwiki maintain`. It reports contradictions and duplicates, records questions
recall could not answer, and refreshes stale catalogs. It never changes what memory
contains, and it writes only `memory_dir/maintenance.md` plus meta.

`interval_hours: 0` disables the scheduled tick (the CLI verb still works with
`--force`). `budget_ms` is a wall-clock cap: jobs are ordered cheapest first and the
tick stops when it is spent, reporting what it skipped, so a tick cannot stall a turn.

The generated report is hard-excluded from search via `config.excluded_paths()`, which
also carries the user's own `exclude` list. A user page named `maintenance.md` elsewhere
in the vault is unaffected — only the concrete report path is excluded.

The contradiction and duplicate sweeps use deterministic signals by default. With
`HWIKI_NLI_CONFLICT=1` the model is also consulted — on a 14-fact vault that found a
swapped-value contradiction the token rules miss.

### Prompt budget (`max_context_tokens`)

```json
{ "max_context_tokens": 1200 }
```

Caps the whole injected recall block, not a hit count. Estimated at 4 characters per
token, whole blocks only — a block is either in or out, never cut mid-sentence — and at
least one block always survives. Priority: recalled facts, then gaps/conflicts/duplicates,
then builtin memory, then wiki text. `0` disables the cap.

### Query-embedding cache (`embeddings.cache_size`)

```json
{ "embeddings": { "mode": "auto", "cache_size": 128 } }
```

Exact-match, bounded LRU over the query text. Measured on the ONNX tier: recall of a
repeated query **7.6 ms cold → 0.92 ms warm**. The lexical (`hash`) tier gains ~nothing —
it was already ~2 ms — so the cache is about the model tier only. `0` disables it.

It is deliberately exact-match: a fuzzy key would answer a different question with the
previous vector, which is the kind of silence this project refuses.

### Choosing a tier by cost

| tier | RSS | recall p50 | disk |
|---|---|---|---|
| hash (no model) | +1.4 MB | 3.2 ms | 0 |
| onnx (MiniLM) | +89 MB | 22.7 ms | 24 MB |
| onnx + NLI | +390 MB | 22.7 ms | 205 MB |

Measured by `tests/bench_footprint.py`. The NLI bundle is only mapped when a pair enters
the model band, so the +390 MB is the cost of using it, not of allowing it.

### Upgrading from the pre-1.0 `wiki` provider

Early versions used the provider name `wiki`, the config file `wiki-memory.json`
and `WIKI_MEMORY_*` environment variables. Those are still read as fallbacks, so
an existing setup keeps working. To switch over:

1. Install hwiki as above and set `memory.provider: hwiki`.
2. Rename `wiki-memory.json` to `hwiki.json` (optional; the old name is read when
   the new one is absent).
3. The index rebuilds automatically under its new name.

## Profiles

Each Hermes profile has its own `$HERMES_HOME`, so install and configure per
profile to scope memory. Profiles can point at different wikis or share one.

## Using an Obsidian vault

Point `wiki_path` at the vault root. Recommended layout inside the vault:

- Keep curated notes anywhere under the indexed globs.
- Set `memory_dir` to a folder you review regularly (for example `wiki/memory`).
- Keep clipped articles, PDFs and other third-party material in an untrusted
  folder (`raw/` by default).

The `.obsidian/` settings folder is never indexed. Obsidian's sync, git plugins and
backups work as normal because hwiki only reads your notes and appends to the
memory folder.

## qmd backend (optional)

[qmd](https://github.com/tobi/qmd) adds semantic and hybrid retrieval. It needs
Node.js 22 or newer, and its vector modes download local GGUF models (roughly 0.3 GB
for `vsearch`, 1.6 GB for `query`).

```bash
npm install -g @tobilu/qmd
hermes config set memory.provider hwiki
# then in hwiki.json: "backend": "qmd", "qmd_mode": "search" | "vsearch" | "query"
hermes hwiki index
```

Selecting `qmd` always builds a hybrid: FTS5 runs first and qmd results are fused
in with Reciprocal Rank Fusion only when qmd answers within `qmd_timeout_s`.
If qmd is missing, slow or broken, recall degrades to FTS5 instead of failing.

Use qmd when the wiki is large and its vocabulary is inconsistent, or when you need
conceptual matches that share no keywords with the query. On a curated wiki, FTS5
alone usually answers correctly.

## Docker

The provider runs inside the Hermes gateway process, so it only needs the wiki on
a writable, persistent volume.

```yaml
services:
  hermes:
    environment:
      HERMES_HOME: /hermes-home
      HWIKI_PATH: /wiki
    volumes:
      - ./my-wiki:/wiki:rw          # must be writable: the agent appends to memory_dir
      - hermes-home:/hermes-home    # config and index survive restarts
volumes:
  hermes-home:
```

Every official `python:*` image ships SQLite with FTS5. Check yours with:

```bash
python3 -c "import sqlite3; sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)'); print('FTS5 OK')"
```

For the qmd backend in a container:

- Use a base image with Node.js 22 or newer, and assert it in the build
  (`node --version && qmd --version`); an older Node only warns during install.
- Put the model cache (`XDG_CACHE_HOME`) on a named volume with at least 3 GB free,
  otherwise models re-download on every recreate or fail on a small tmpfs.
- Allow around 4 GB of memory for `vsearch` and `query` modes.

## Troubleshooting

**`hermes memory status` does not list hwiki.** The plugin must be at
`$HERMES_HOME/plugins/hwiki/`. Run `hermes config path` to see which
`$HERMES_HOME` is active.

**`hermes hwiki` is not a command.** CLI commands register only for the active
provider. Set `memory.provider: hwiki` and restart.

**Provider loads but is unavailable.** `wiki_path` is unset or not a directory.
`hermes hwiki config` shows what resolved.

**No recall.** Check `auto_recall`, and that the message is at least
`min_query_chars` long. Test retrieval with `hermes hwiki search "..."`.

**New edits are not found.** Agent writes reindex immediately; external edits are
picked up within `reindex_interval_s`. Force with `hermes hwiki index`.

**`database is locked`.** Two Hermes processes share one `index_path`. Give each
profile its own; the index is disposable.

**qmd is slow on the first query.** Vector modes load their models on first use.
Run a warm qmd daemon, raise `qmd_timeout_s`, or use `qmd_mode: search`. Timeouts
are not fatal: you get FTS5 results.
