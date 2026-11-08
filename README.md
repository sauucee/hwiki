# hwiki: a markdown wiki as Hermes Agent memory

**Long-term memory you can browse and edit easily.** hwiki turns a folder of markdown
files into [Hermes Agent](https://github.com/NousResearch/hermes-agent) memory.
The agent recalls from it before every turn and files new facts into it. You read,
edit and reorganise the same files in editors and tools like Obsidian, and track every
change in git.

The idea and structure is based on [Karpathy's LLM wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

No opaque databases you can't easily browse, no cloud service, no extra dependencies: Python's
standard library and a local SQLite FTS5 index. 

You think querying the markdown memory is slow? The provider answers in milliseconds.

## v2.0 — combined memory provider (wiki + review-gated facts)

hwiki now owns the full memory loop. The standalone `hwiki-extractor` plugin
merged into this provider (v2.0):

- **Write path (review-gated)**: per-turn extraction proposes durable facts
  (`memory_propose`); only user-accepted facts become one Obsidian note each
  in the memory zone. `memory_supersede` / `memory_forget` / `memory_accept`
  manage the lifecycle; every change lands in the audit log and regenerates
  the fact-graph Canvas.
- **Dynamic per-turn injection**: `prefetch` fuses approved-fact recall
  (local ONNX MiniLM vectors, gracefully degrading to hashed n-grams),
  builtin-memory lines, and wiki recall into one bounded block — every turn,
  not just session start.
- **Learns from you**: accept/reject decisions feed back into extraction
  prompts per fact kind; usage counters rank facts that actually get used;
  never-used old facts resurface as "still true?" review candidates.
- **Self-healing + hand-editable**: the vault is the source of truth — edits
  sync back, deletions revoke, and a lost fact database rebuilds from pages.
- **Builtin-memory compliant**: never writes Hermes MEMORY.md/USER.md;
  cross-checks for duplicates; recall can search them read-only.

Config (all optional) lives under a `"facts"` key in `hwiki.json`:

```json
{
  "facts": {
    "injection": {"enabled": true, "max_chars": 3500, "max_facts": 25},
    "review": {"cadence_turns": 5, "decay_days": 90, "proposal_ttl_days": 14},
    "diary": {"enabled": true},
    "embeddings": {"mode": "auto"}
  }
}
```

The optional ONNX embedding tier (~25 MB, one-time, fully offline afterwards)
is set up by `hwiki-extractor`'s `tests/setup_embeddings.sh` recipe or any
equivalent; without it recall degrades to word-matching, never failing.

## Navigation catalog: `index.md` + `log.md` (v2.1)

The [LLM wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
keeps two special files beside the pages. hwiki maintains both automatically:

- **`index.md`** — a content catalog at the wiki root: every page as a
  `[[wikilink]]` with a one-line summary and its `updated` date, grouped by
  folder, with memory notes listed separately. Regenerated on startup and
  after every write. Summaries are deterministic (frontmatter
  `summary`/`description`, else the first prose sentence) — no LLM call, so
  the catalog cannot silently drift when a model call fails.
- **`log.md`** — append-only timeline. Every entry is
  `## [YYYY-MM-DD] kind | subject`, so the history is greppable:
  `grep "^## \[" log.md | tail -5`.

Both are **derived artifacts**: delete them and nothing is lost. They are
written to the wiki root, never into `memory_dir` — that zone belongs to
accepted facts and must not be polluted with generated catalogs. They are
also held out of the ranking (`demote_paths`), because retrieval here is
FTS5 + local embeddings: an index-first navigation step would let a page that
merely *lists* every title outrank the page holding the answer.

Browse them in Obsidian, or from the CLI:

```bash
hermes hwiki catalog   # regenerate index.md
hermes hwiki log       # show the last 10 events (--limit N)
hermes hwiki file      # file a synthesised answer into Hermes/Memory/Auto/answers/
```

`wiki_memory_status` reports both files under a `catalog` key, and log entries
are written for `compile` and `file` events as well as plain writes.

## Compile: one page per topic (v2.1)

The older behaviour appended every accepted fact to its own page, so a user
who refined a preference three times ended up with three near-duplicate pages.
`compile` fixes that: when a fact is accepted and an existing page already
owns the topic, the new claim is merged into that page as a delimited section
instead of becoming a page of its own.

Compile runs **strictly after the review gate** — the fact is already accepted
and already written to its own page before any merging is attempted. Four
guarantees, each covered by a test:

1. **Never lose a fact.** Any failure leaves the standalone page in place.
2. **Never silently contradict.** A conflict-flagged proposal is never folded
   into the claim it disagrees with; contradictions keep their own page.
3. **Never clobber hand edits.** Only the region between the
   `<!-- hwiki:compiled:start/end -->` delimiters is touched; frontmatter gains
   `compiled_sections` and a fresh `updated`. Everything else survives byte for
   byte, so editing a page in Obsidian is safe.
4. **Never raise.** A compile error cannot break an accepted write.

Merged claims stay recallable: `read_fact_pages` emits each compiled section as
its own fact, so a merge is invisible to retrieval. Merging is idempotent —
re-running it on the same claim is a no-op. Disable with `compile_enabled:
false` in `hwiki.json`; the provider then reproduces the v2.0 behaviour exactly.

Owner selection uses the similarity already computed when the proposal was
scored (`related[].score`), so it costs no extra embedding call. The bar is
0.62 — deliberately low, because the point is to catch *refinements of the same
topic*, and the conflict check plus the idempotency check do the real safety
work. Ties break toward the page with fewer merged sections.

## Filing an answer back (v2.1)

`wiki_memory_file` turns a synthesised answer into a page, so an exploration
compounds instead of disappearing into chat history:

```json
{"question": "How do I rotate the deploy key?",
 "answer": "Revoke, regenerate, redeploy, verify.", "tags": "ops, keys"}
```

Filed answers land in `answers/` **inside** `memory_dir` — the only tree this
provider may write — but segregated from accepted facts, so they never enter
fact recall or the review queue. They are ordinary pages for search purposes.
The text is secret-scanned before anything touches disk, and filing is
idempotent: the same question returns the existing page rather than
overwriting it. Two *different* questions that slugify alike get separate
files (`-<sha1>` suffix) so the first is never clobbered.

## Navigation catalog: `index.md` + `log.md` (v2.1)

## Recall quality (1.1.0)

- **Stopword hygiene**: personal pronouns/possessives ("user", "my", "their"…) are
  stripped from queries — "what does the user prefer" now compiles to `"prefer"*`
  instead of matching every memory page's title.
- **Recency-aware ranking**: `pages.updated` (frontmatter date, falling back to
  indexed mtime) applies a gentle multiplier — `1.0 + 0.25·e^(−age_days/45)` — so
  freshly updated pages win ties while old pages are demoted, never hidden.
- Ranking regressions live in `tests/test_ranking.py`.

## Why hwiki

| Typical agent memory | hwiki |
|---|---|
| Opaque rows or vectors | Plain markdown pages you can read |
| Fixing outdated and wrong memories | Quickly skim over pages and fix with any editor |
| No history | `git log`, `git diff`, `git blame` |
| A pile of facts | Pages, headings, `[[wikilinks]]`, frontmatter, backlinks |
| Data leaves the machine (hosted options) | Local files, local index, zero network calls |
| Backups and restores | Simple and flexible; just sync with tools like Obsidian and Nextcloud, git history gives you the ability to revert any changes |

### Work on the same knowledge base as your agent

The wiki is a shared artifact, and you and the agent are co-authors:

- **You curate in your favourite editor.** Write pages, link them, restructure folders. hwiki
  re-indexes changes automatically, so the agent sees your edits within a minute.
- **The agent recalls what you wrote.** Before each turn the most relevant
  heading-sized excerpts are injected, with page and section, so answers are
  grounded in your notes and citable.
- **The agent writes to a staging area, never over your pages.** New facts are stored as
  dated bullets in `wiki/memory/`. You review them in your favourite markdown editor, promote what
  matters into proper pages, and delete the rest. One `git diff` shows everything
  the agent added.
- **The graph works for both of you.** `[[wikilinks]]` and backlinks render in
  tools like Obsidian's graph view and let the agent follow links instead of re-searching.

## Install

```bash
hermes plugins install https://github.com/ottosulin/hwiki
hermes memory setup            # choose "hwiki", point it at your wiki folder
hermes gateway restart         # or restart the CLI
hermes hwiki status            # verify: root, backend, page count
```

Or configure it non-interactively:

```bash
hermes config set memory.provider hwiki
cat > ~/.hermes/hwiki.json <<'JSON'
{ "wiki_path": "~/wiki" }
JSON
```

Any folder of markdown works, including for example an existing Obsidian vault.

## How it works

```
my-wiki/
├── wiki/              indexed and recalled, human-curated
│   ├── concepts/ ...
│   └── memory/        the ONLY place the agent writes (dated bullets)
└── raw/               never indexed, never recalled, never readable by memory tools
```

- **Recall:** heading-scoped chunks from SQLite FTS5 (BM25). Sub-10 ms on a
  2,000-page wiki. Optional [qmd](https://github.com/tobi/qmd) hybrid semantic
  search, with automatic fallback to FTS5 if qmd is missing or slow.
- **Writes:** append-only to `memory_dir`, de-duplicated, with a secret guard that
  refuses API keys, tokens and private keys.
- **Untrusted sources:** put third-party documents in `raw/`. They never reach
  auto-injected context, which is where prompt-injection attacks are most
  dangerous. Summarise them into `wiki/` yourself.
- **Background agents are read-only:** in cron, subagent and flush contexts the
  write tool is not offered, so automated runs cannot pollute your wiki.

## About agent's access to the wiki

The write tool is intentionally simple, and currently there is no replace tool. Replace tool
is something to be considered later. This restricted set of tools now intentional to support 
cases where you do not want to give the agent unrestricted access to the wiki files.

**In practice in many use cases the agent should have access** (running on same filesystem or wiki
directory is mounted to the sandbox container) **to the wiki files**. This is the way I use this. 
By giving the agent _also_ unrestricted access gives it most freedom to keep wiki pages up to date, 
save multiple documents from research etc. Plugin tool interfaces will always have friction compared
to agent being able to use grep, sed and other command line tools. However, the search tool will
outperform grep when amount of pages grows for recall, and also prevents polluting the results with 
hits from saved raw materials.

## Agent tools

| Tool | Purpose |
|---|---|
| `wiki_memory_search` | Ranked excerpts across the wiki (optional frontmatter filter) |
| `wiki_memory_read` | Full page by path, slug or `[[wikilink]]`, with links and backlinks |
| `wiki_memory_write` | Append a dated fact to a page in the memory zone |
| `wiki_memory_file` | File a synthesised answer as a reusable page (`answers/`) |
| `wiki_memory_status` | Index stats; with `lint`, dangling links and orphan pages |

The fact layer (`facts.py`) additionally stamps every recall hit with a citation:
`page` (the vault page slug) and `cite` (`[[<slug>]]`), plus `accepted` (the date
the fact was approved). Only the slug is exposed — never the absolute path. See
[docs/wiki-schema.md](docs/wiki-schema.md) for the full agent-facing contract:
source trust, approval states, provenance fields, citations, and lint severity.

## CLI

```bash
hermes hwiki status              # root, backend, counts
hermes hwiki search "release process"
hermes hwiki index               # force a full reindex
hermes hwiki lint                # wiki links + memory-tree findings
hermes hwiki lint --json         # same, machine-readable
hermes hwiki plan --from notes.md  # read-only: what would happen to these claims?
hermes hwiki file -q "Q" < ans.md   # file an answer back (same code as the tool)
hermes hwiki confidence "query"  # is this hit set trustworthy? (--json)
hermes hwiki config              # resolved configuration
hermes hwiki migrate --dry-run   # import facts from the holographic provider
```

`lint` is read-only and never repairs. It reports provenance that cannot be read
as **unknown** rather than guessing a value, and it checks hand-written notes in
the memory zone for secret material even though it skips their schema.

`plan` is also read-only — it writes no proposal rows, no pages and no catalog,
and refuses rather than creating a missing database. It reports a predicted action
per claim (`near_duplicate`, `conflict`, `merge_into_owner`, `new_page`,
`refused_secret`). It is **not an extractor**: candidate claims are split naively
without a model, so pass pre-extracted claims if you have them.

`recall_confidence(query)` is a pure observation on top of recall: `best`,
`margin`, `overlap`, `floor` and a `confident` flag that means "not empty and not
ambiguous" — **not** "this answers the question". The `caveat` field says so and
must travel with the number.

## Configuration

`$HERMES_HOME/hwiki.json`. Only `wiki_path` is required.

| Key | Default | Description |
|---|---|---|
| `wiki_path` | | Wiki root. `~` and `$HERMES_HOME` expand. |
| `backend` | `fts5` | `fts5`, or `qmd` (hybrid with FTS5 fallback) |
| `memory_dir` | `wiki/memory` | The only folder the agent may write to |
| `untrusted_dirs` | `["raw"]` | Never indexed, recalled or readable |
| `prefetch_limit` | `4` | Excerpts injected per turn |
| `prefetch_max_chars` | `1400` | Cap on injected characters per turn |
| `auto_recall` | `true` | `false` = tools only |

Environment overrides: `HWIKI_PATH`, `HWIKI_BACKEND`, `HWIKI_MEMORY_DIR`,
`HWIKI_QMD_BINARY`, `HWIKI_QMD_MODE`. All options, qmd tuning and Docker setup:
[docs/configuration.md](docs/configuration.md).

## Privacy and security

- **No network access** with the default `fts5` backend. Your notes never leave
  the machine. The optional qmd backend runs local models.
- **Data locations:** your wiki folder (which you choose) plus
  `$HERMES_HOME/hwiki.json` and `$HERMES_HOME/hwiki-index.sqlite` (disposable,
  rebuilt on demand).
- Writes are confined to `memory_dir`, path traversal is rejected, and search
  input is treated as data, never as query syntax.

Details: [docs/security.md](docs/security.md). Design rationale and benchmarks:
[docs/design.md](docs/design.md).

## Development

```bash
python3 tests/test_store_search.py   # store, chunking, search, trust boundary
python3 tests/test_provider.py       # full MemoryProvider contract
python3 tests/test_cli_load.py       # CLI loaded exactly as Hermes loads it
python3 -m pytest tests/             # naming, resolution (needs pytest + PyYAML)
hermes plugins validate .            # Hermes catalog admission checks
```

Contributions welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
