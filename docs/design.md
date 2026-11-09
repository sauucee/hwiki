# Design notes

## Principles

1. **The curated wiki is read-only to the agent.** Automatic writes land only in the
   memory zone, so they can never corrupt hand-maintained pages, and every agent
   write is reviewable in one `git diff`.
2. **Untrusted zones are never read** by memory tools (see [security.md](security.md)).
3. **Recall must be fast and must never block a turn.** Hermes times out prefetch;
   FTS5 answers in milliseconds and work happens behind a bounded lock.
4. **Degrade, never fail.** A missing qmd, Node, PyYAML or wiki folder produces a
   warning and reduced function, never an exception.

## Architecture

```
Hermes MemoryManager
  ├─ system_prompt_block()  "memory is a wiki at <path>, N pages"
  ├─ prefetch(query)        top excerpts injected before each turn
  ├─ queue_prefetch(query)  warms the next turn in the background
  ├─ handle_tool_call(...)  wiki_memory_{search,read,write,status}
  └─ on_memory_write(...)   mirrors built-in memory additions
          │
   WikiMemoryProvider (__init__.py)      config.py: shared by provider and CLI
          ├─ WikiStore (store.py)        frontmatter, wikilinks, chunking,
          │                              secret guard, locked appends
          └─ Backend (search.py)         Fts5Backend (default), QmdBackend,
                                         HybridBackend (RRF fusion)
```

`cli.py` imports shared helpers from `config.py`, not from the package. Hermes loads
a plugin's CLI with the plugin registered as an empty package (its `__init__.py` is
not executed), so `from . import X` would fail there. `tests/test_cli_load.py`
reproduces that loading path.

## Why heading-scoped chunks

Injecting whole pages would blow the context budget and bury the relevant paragraph.
Chunking on H1 to H3 keeps excerpts small, self-describing and citable as
`page › heading`.

## Why page identity, not the filename, keys the graph

Every page has a root-relative path that is unique by construction; filenames are
not. A vault may hold `Hermes/Decisions.md` and `Projects/Alpha/Decisions.md`, and a
slug-keyed graph then merges two different pages into one node: backlinks from
either page appear under both, a page linking both lists the same target twice, and
an orphan twin looks navigated-to because its sibling is.

`link_graph()` therefore takes an explicit key: `key="slug"` (the default, kept for
compatibility with existing callers) and `key="path"` (one node per page). The
default cannot be changed silently — prompt injection and status output read it by
slug — so callers that attribute a link to a page use the path view, **beside** the
old fields rather than instead of them: `backlink_paths` next to `backlinks` in
`wiki_memory_read`, `orphan_page_paths` next to `orphan_pages` in
`wiki_memory_status(lint=True)`, and `dangling_links(key="path")` next to the
slug-keyed default. `hermes hwiki lint` is human-facing and prints paths.

The catalog follows the same rule. It emits `[[Folder/Page|Page]]` only when two
pages share a filename, keeps the bare `[[Stem]]` everywhere else, and lists pages
whose paths contain `|`, `#` or `]]` as plain paths, because no wikilink can address
them. Lint does not demand a link it cannot express.

Stem uniqueness is not sufficient on its own: a page's bare stem can also name a
generated artifact or another page. Every candidate target is therefore resolved
against the store before it is emitted, and the folder-qualified form is preferred
when the bare one opens a different page (`Projects/index.md` beside the root
`index.md`). A page for which no form resolves back to it is listed as a plain path,
like an unlinkable name. `tests/test_catalog.py` asserts the invariant over a vault
built from every collision case: every emitted link resolves to a page the catalog is
supposed to list.

Lint judges catalog coverage by the same function the reader uses. An entry covers
exactly the page `resolve_page` returns: a true ambiguity (two pages matched
case-insensitively) resolves to neither and is warned about, while a bare shared stem
resolves to its first match and the other twin is reported `entry_missing`.
`_Identity.candidates()` is built on that same resolver, so it can never credit a page
the reader cannot open (`tests/test_lint.py` pins the parity).

### Compatibility rule: extend, never re-key

A tool payload is a published contract: an agent may read `backlinks[0]`, a script may
key a report by the string that used to be there. So the identity fixes land as
*additions* — an old field keeps its name, shape and ordering — and the payload carries
`contract_version` (currently `2`) so a consumer can branch instead of guessing. There
is no silent re-key anywhere in the tool surface.

| Surface | Before | After | Risk / mitigation |
|---|---|---|---|
| `WikiStore.dangling_links()` keys | source **slug** (`notes`) | **default unchanged** (slug); `key="path"` opt-in view | Existing callers keep working. Two same-named sources no longer merge *when the path view is asked for* (`wiki_memory_status` reports both views). |
| `wiki_memory_read` → `backlinks` | slugs | **unchanged** (slugs), plus additive `backlink_paths` | Additive; the twin-exact list arrives under a new name, so no consumer is re-keyed. |
| `wiki_memory_status(lint=True)` → `orphan_pages` | slugs | **unchanged** (slugs), plus `orphan_page_paths`, `dangling_link_paths`, `contract_version` | Same; a twin can no longer hide behind its sibling's backlinks because the exact view is present. |
| `hermes hwiki lint` orphan line | slugs, deduplicated by filename | paths | Human-facing only; more precise, occasionally longer. |
| `WikiStore.link_graph()` | slug keys, duplicate outlinks, first-match path folding | slug keys (**default unchanged**), deduped outlinks, `key="path"` opt-in | Existing callers keep working. Dedupe removes repeated entries; no consumer reads multiplicity. |
| `link_graph` target that cannot be resolved | attached to the first case-folded match | keyed by the raw target | An edge is never invented on the wrong twin. |
| Index with a bare link to a shared stem | counted as coverage for both pages | covers exactly the page the resolver picks; the twin is `catalog.entry_missing` | Overstating coverage hid a page; understating it flags one forever. A true ambiguity (case-folded twins) still warns `catalog.ambiguous` and covers neither, because no reader can open it. |
| Index entry for a path containing `|`, `#` or `]]` | emitted as a broken wikilink | listed as a plain path under "Not linkable from an index" | The catalog cannot express such a path; lint no longer demands a link it cannot represent. |
| `lint_memory` `link.dangling` | only provider-written fact pages | every page in the memory zone, and code fences/inline code are literal | Removes false negatives (hand-written notes) and false positives (samples). |
| Artifacts job | rewrote the index every tick once `maintenance.md` existed | treats its own report page as an artifact | No write per tick that changes nothing. |

Unchanged on purpose: the default `link_graph()` key space, `resolve_page`'s bare
slug/title fallback (first match wins), the bare `[[Stem]]` catalog form for unique
stems, and — pinned by `tests/data/index_v1_golden.md` — the entire `index.md` text for
an ordinary vault.

### Appending to the log, and reading it back

`log.md` is the audit trail, so the append path is stricter than the index path:

- **Append-only in bytes.** Existing content is never rewritten. If a crash left the
  last line without a newline, one separator newline is written first, so the next
  entry starts on its own line instead of merging into the broken one.
- **Locked, on one inode.** Appends take the same advisory file lock page writes use.
  Two turns cannot interleave entries, and a fresh log cannot lose an entry to two
  writers both writing the header. The lockfile is never deleted on release: unlinking
  it lets a waiter block on the removed inode while a newcomer creates a second one and
  locks that, so both hold "the lock" at once. A stale empty `*.lock` is the price.
- **Every unsafe write fails closed.** The vault may be a shared directory, so a write
  path is checked against the root and refused — never attempted — when a leaf symlink,
  a hardlink (`st_nlink > 1` is the same inode as some outside copy), or a symlinked
  parent directory would carry it outside. The leaf is opened with `O_NOFOLLOW`, so a
  link swapped in between the check and the open is refused by the kernel. The same
  rule guards memory-note writes; a refused write returns `None` (or raises
  `ValueError` before touching the file), and the existing file is unchanged.
- **Ownership is re-checked under the lock.** The pre-check is only a cheap skip: after
  the lock is taken the opened descriptor is re-read for the `source: hwiki-catalog`
  marker, so a log replaced by hand-written content in that window is left alone
  instead of being appended to.
- **No lock, no write.** A lock that cannot be acquired within its timeout raises
  `TimeoutError` rather than proceeding unlocked; callers turn that into "nothing was
  written". Degrading to an unlocked write would silently allow exactly the interleaved
  writes the lock exists to prevent.
- **Empty is adoptable, non-empty is not.** A zero-length `log.md` (full disk, crash)
  has nothing to preserve, so the header is restored and the trail continues. A
  non-empty file without the `source: hwiki-catalog` marker is still never touched.
- **A bad timestamp refuses the entry.** `when` must be a `datetime`; a string would
  otherwise raise into the caller. A wrong date is worse than a missing line in an
  audit trail, so the call returns `None` and logs a warning.
- **The lock is on a stable inode.** `_FileLock` opens the lockfile with
  `O_NOFOLLOW` too: a symlinked lockfile would otherwise lock (or touch) a file
  outside the vault.
- **Cheap to read and to append.** The log is never truncated, so both paths are
  bounded: a short tail is read from the end of the file (16 KB window, quadrupled
  only when it does not hold enough entries), and ownership checks read only the
  frontmatter head. Measured on a 7 MB log: tail read 148 ms → 0.8 ms; on a 2 MB log:
  append 4.5 ms → 0.7 ms. A short tail no longer allocates the whole file.

## Why FTS5 by default

| Backend | Query latency | Extra dependencies | Disk |
|---|---|---|---|
| `fts5` | under 1 ms (50 pages), 5 to 10 ms (2,000 pages) | none (stdlib) | about 1.5x the corpus |
| `qmd search` (BM25) | 200 to 300 ms | Node 22 + qmd | no models |
| `qmd vsearch` (vector) | seconds, slow first call | + models | about 0.3 GB |
| `qmd query` (hybrid + rerank) | 2 to 3 s warm | + models | about 1.6 GB |

A full FTS5 rebuild of a 2,000-file, 7.6 MB corpus takes about a quarter of a second.
There is no scale reason to leave FTS5 below several thousand pages; the reason to
add qmd is retrieval quality on inconsistent vocabulary. On a curated wiki, where
pages share consistent terms, BM25 over heading chunks answers conversational
questions well, and in testing it answered some that qmd's own BM25 mode missed.

### Alternatives considered

| Option | Why not |
|---|---|
| grep / ripgrep | No ranking, chunking or snippets. |
| sentence-transformers | Hundreds of MB of dependencies in the gateway process. |
| fastembed / sqlite-vec | Lighter, but still need an embedding stack and model download. |
| tantivy / whoosh | Keyword engines like FTS5, so no quality gain for an extra dependency. |
| Hosted embeddings | Sends the private knowledge base to a third party. |
| MCP note servers | Provide tools but no prefetch hook, so no automatic recall. |

## Why no automatic transcript ingestion

Writing every turn would turn the wiki into a chat log and remove what makes it
better than a database: a human can read it. `sync_turn()` is deliberately a no-op.

## Why no trust scores

Trust scores approximate "is this still true". A wiki answers that with `updated:`
frontmatter, git history and a person reading the page, all of which are legible in
a way a number is not.

## Migrating from holographic memory

`hermes hwiki migrate` reads a holographic `memory_store.db` read-only and files each
fact into the memory zone, one page per category. It introspects the source schema
rather than assuming it, keeps original dates, converts linked entities to
`#hashtags`, skips duplicates, and blocks secret-shaped facts. `--min-trust` filters
low-confidence facts; trust scores are not written into the wiki.

Recommended sequence: install hwiki while keeping the old provider active, run
`migrate --dry-run`, import, curate the imported pages into proper notes, then
switch `memory.provider` and restart. Keep the old database as a rollback.
