"""Wiki memory provider — a markdown wiki as Hermes Agent long-term memory.

Replaces opaque database-backed memory with a human-readable, git-versionable
markdown wiki (Obsidian-compatible). The agent recalls from the wiki before
each turn and files durable facts into a dedicated memory zone.

Design commitments
------------------
1. **The curated wiki is read-only to the agent.** Automatic writes land only
   under ``memory_dir``. Hand-maintained synthesis pages are never mutated.
2. **Untrusted zones are never read.** ``untrusted_dirs`` (default ``raw/``)
   holds third-party source material and is excluded from indexing and recall,
   because auto-injected context is the highest-value prompt-injection target.
3. **Recall must be fast and must never block a turn.** Hermes hard-timeouts
   ``prefetch()`` at 8s; the default FTS5 backend answers in single-digit ms
   and all work happens behind a bounded lock.
4. **Degrade, never fail.** No qmd, no node, no PyYAML, no wiki directory —
   every one of those is a warning plus reduced function, never an exception.

Config lives in ``$HERMES_HOME/hwiki.json``; see README.md.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from agent.memory_provider import MemoryProvider
except Exception:  # pragma: no cover - offline testing without Hermes installed
    class MemoryProvider:  # type: ignore[no-redef]
        """Minimal shim so the module imports outside a Hermes checkout."""

try:
    from tools.registry import tool_error
except Exception:  # pragma: no cover
    def tool_error(message: str) -> str:  # type: ignore[misc]
        return json.dumps({"success": False, "error": message})

from .store import WikiStore, scan_for_secrets, slugify
from .search import build_backend
from .facts_provider import FactsEngine, FACT_TOOL_SCHEMAS, FACT_TOOLS_READ_ONLY
from . import facts as facts_mod
from .config import (
    CONFIG_FILENAME,
    DEFAULTS,
    build_runtime,
    excluded_paths,
    default_index_path,
    expand_path as _expand,
    load_config,
    resolve_hermes_home,
    save_config_file,
)

try:  # catalog is a derived artifact; never brick the provider without it
    from .catalog import append_log, read_log_tail, write_index
    CATALOG_AVAILABLE = True
except Exception as _catalog_exc:  # pragma: no cover
    CATALOG_AVAILABLE = False
    _CATALOG_IMPORT_ERROR = str(_catalog_exc)

logger = logging.getLogger(__name__)

__all__ = [
    "WikiMemoryProvider", "register", "load_config", "build_runtime",
    "CONFIG_FILENAME", "DEFAULTS",
]


# ---------------------------------------------------------------------------
# Tool schemas  (bare OpenAI function shape — the manager wraps them)
# ---------------------------------------------------------------------------

WIKI_SEARCH_SCHEMA = {
    "name": "wiki_memory_search",
    "description": (
        "Search long-term memory: the markdown knowledge wiki. Use this BEFORE "
        "answering any question that durable notes may already cover (past "
        "decisions, people, systems, compliance posture, prior incidents, "
        "vendor choices, recurring answers).\n\n"
        "Returns ranked page excerpts with their wiki paths. Follow up with "
        "wiki_memory_read to load a full page when an excerpt looks relevant. "
        "Each result also lists that page's outbound [[wikilinks]] as "
        "follow-up hints — read a linked page when the excerpt is partial."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language question or keywords.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 6, max 20).",
            },
            "spine": {
                "type": "string",
                "description": "Optional frontmatter 'spine' filter (e.g. compliance, product).",
            },
        },
        "required": ["query"],
    },
}

WIKI_READ_SCHEMA = {
    "name": "wiki_memory_read",
    "description": (
        "Read a wiki page in full by path, slug, or [[wikilink]] name. Also "
        "returns the page's frontmatter, outbound links and backlinks so you "
        "can traverse the knowledge graph."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "page": {
                "type": "string",
                "description": "Page path ('wiki/concepts/le-chonk.md'), slug ('le-chonk'), or wikilink target.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Truncate the body at this many characters (default 8000).",
            },
        },
        "required": ["page"],
    },
}

WIKI_WRITE_SCHEMA = {
    "name": "wiki_memory_write",
    "description": (
        "File a durable fact into long-term memory. Writes a dated bullet into "
        "the wiki's memory zone — the curated wiki pages are never modified.\n\n"
        "WRITE when: the user states a preference or correction, a decision is "
        "made with a rationale, you learn a stable fact about the environment, "
        "or an interpretation should be reused next time.\n"
        "DO NOT write: secrets, tokens, credentials, personal data, task "
        "progress, or anything stale within a week.\n\n"
        "Group related facts by passing the same 'page'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact, as one self-contained declarative sentence.",
            },
            "page": {
                "type": "string",
                "description": "Memory page to append to (e.g. 'decisions', 'vendors'). Defaults to 'journal'.",
            },
            "topic": {
                "type": "string",
                "description": "Human-readable title used when the page is created.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags appended as #hashtags.",
            },
            "source": {
                "type": "string",
                "description": "Optional provenance (URL, ticket key, document id).",
            },
        },
        "required": ["content"],
    },
}

ANSWERS_SUBDIR = facts_mod.ANSWERS_SUBDIR

WIKI_FILE_SCHEMA = {
    "name": "wiki_memory_file",
    "description": (
        "File a synthesised answer back into the wiki as a page, so an "
        "exploration compounds instead of disappearing into chat history.\n\n"
        "USE WHEN: you just produced an analysis, comparison, or synthesis the user "
        "is likely to want again — the kind of thing worth re-reading later.\n"
        "DO NOT use for: transient progress, one-line facts (use wiki_memory_write), "
        "or anything containing secrets.\n\n"
        "Idempotent: filing the same question twice returns the existing page."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question or prompt this answer responds to.",
            },
            "answer": {
                "type": "string",
                "description": "The answer/synthesis, as markdown.",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags (optional).",
            },
        },
        "required": ["question", "answer"],
    },
}

WIKI_STATUS_SCHEMA = {
    "name": "wiki_memory_status",
    "description": (
        "Inspect memory health: wiki root, active search backend, page/chunk "
        "counts, memory-zone pages, and knowledge-graph hygiene (dangling "
        "wikilinks, orphan pages). Use when asked what the agent remembers, "
        "or to debug recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "lint": {
                "type": "boolean",
                "description": "Include dangling-wikilink and orphan-page report.",
            },
        },
    },
}

WIKI_CONFIDENCE_SCHEMA = {
    "name": "wiki_memory_confidence",
    "description": (
        "Check whether a memory retrieval is trustworthy BEFORE answering from it.\n\n"
        "Returns the top score against the noise floor, the margin over the runner-up, "
        "lexical overlap, the candidate count and a boolean `confident`.\n\n"
        "USE WHEN: you are about to state a remembered fact and want to know if the "
        "evidence is thin, ambiguous or absent.\n"
        "READ THE CAVEAT: `confident` means the retrieval is not empty and not "
        "ambiguous. It does NOT mean the fact answers the question. If `confident` is "
        "false, say what you do not know or ask, rather than guessing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question you are trying to answer from memory.",
            },
            "limit": {
                "type": "integer",
                "description": "How many facts to consider (default 8).",
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Crude prompt-size estimate: 4 characters per token.

    Deliberately not a tokenizer: the budget has to be free to compute on every turn, and
    being 20% wrong is harmless when the failure mode is "one block fewer".
    """
    return (len(text or "") + 3) // 4


class WikiMemoryProvider(MemoryProvider):
    """Markdown wiki as Hermes long-term memory."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or load_config()
        self._store: Optional[WikiStore] = None
        self._backend = None
        self._session_id = ""
        self._hermes_home = str(self._config.get("_hermes_home") or "")
        self._lock = threading.RLock()
        self._last_index = 0.0
        self._pending: Dict[str, str] = {}      # session_id -> prefetched context
        self._writes_this_session = 0
        self._init_error = ""
        self._agent_context = "primary"
        self._read_only = False
        self._facts: Optional[FactsEngine] = None

    # -- identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "hwiki"

    def is_available(self) -> bool:
        """True when a wiki root is configured and present. No network."""
        raw = str(self._config.get("wiki_path") or "")
        if not raw:
            return False
        home = self._hermes_home or os.environ.get("HERMES_HOME", "")
        return Path(_expand(raw, home)).is_dir()

    # -- lifecycle ---------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        home = kwargs.get("hermes_home") or self._hermes_home
        self._hermes_home = str(home or "")

        # Non-primary agent contexts (cron, subagent, flush) get READ-ONLY
        # memory. The ABC is explicit that letting them write corrupts the
        # store — a nightly cron would file its own system prompt as a
        # "durable fact" and quietly poison the wiki. Recall stays enabled so
        # scheduled jobs still benefit from what the primary agent learned.
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._read_only = self._agent_context != "primary"
        if self._read_only:
            logger.info(
                "wiki memory: agent_context=%s — recall enabled, writes disabled",
                self._agent_context,
            )

        raw_path = _expand(str(self._config.get("wiki_path") or ""), self._hermes_home)
        if not raw_path:
            self._init_error = (
                "wiki_path is not set — run `hermes memory setup` or set "
                f"wiki_path in {Path(self._hermes_home) / CONFIG_FILENAME}"
            )
            logger.warning("wiki memory: %s", self._init_error)
            return

        root = Path(raw_path)
        if not root.is_dir():
            self._init_error = f"wiki root does not exist: {root}"
            logger.warning("wiki memory: %s", self._init_error)
            return

        self._store = WikiStore(
            root,
            include=self._config.get("include") or ["**/*.md"],
            # generated artifacts included: this used to pass the raw exclude list and
            # silently diverged from build_runtime, so the CLI excluded the maintenance
            # report and the provider recalled it.
            exclude=excluded_paths(self._config),
            untrusted_dirs=self._config.get("untrusted_dirs") or [],
            memory_dir=str(self._config.get("memory_dir") or "wiki/memory"),
        )

        index_path = (_expand(str(self._config.get("index_path") or ""), self._hermes_home)
                      or default_index_path(self._hermes_home))

        self._backend = build_backend(
            self._store,
            backend=str(self._config.get("backend") or "fts5"),
            db_path=index_path,
            qmd_binary=str(self._config.get("qmd_binary") or "qmd"),
            qmd_collection=str(self._config.get("qmd_collection") or "hwiki"),
            qmd_mode=str(self._config.get("qmd_mode") or "search"),
            qmd_timeout=float(self._config.get("qmd_timeout_s") or 6.0),
            max_hits_per_page=int(self._config.get("max_hits_per_page") or 2),
            demote_paths=list(self._config.get("demote_paths") or []),
            fetch_multiplier=int(self._config.get("fetch_multiplier") or 4),
        )

        try:
            stats = self._backend.sync()
            self._last_index = time.time()
            logger.info(
                "wiki memory: indexed %s (backend=%s) %s",
                root, self._backend.name, stats,
            )
        except Exception as exc:
            self._init_error = f"initial index failed: {exc}"
            logger.warning("wiki memory: %s", self._init_error)

        # Derived navigation artifacts (index.md + log.md). Off when the
        # provider is read-only: a cron run must not rewrite the catalog.
        self._catalog_enabled = bool(
            self._config.get("catalog_enabled", True)) and not self._read_only
        self._index_path = (self._store.root / str(
            self._config.get("index_filename") or "index.md")) if self._store else None
        self._log_path = (self._store.root / str(
            self._config.get("log_filename") or "log.md")) if self._store else None
        if self._catalog_enabled and not CATALOG_AVAILABLE:  # pragma: no cover
            logger.warning("wiki memory: catalog module unavailable: %s",
                           _CATALOG_IMPORT_ERROR)
            self._catalog_enabled = False

        # review-gated fact write path (merged from hwiki-extractor, v2.0)
        try:
            self._facts = FactsEngine(
                self._hermes_home, root,
                Path(str(self._config.get("memory_dir") or "wiki/memory")),
                config=dict(self._config.get("facts") or {}))
            self._facts.read_only = self._read_only
            self._facts.store()  # opens + migrates + heals from the vault
            logger.info("wiki memory: fact store ready (read_only=%s)", self._read_only)
        except Exception as exc:
            logger.warning("wiki memory: fact store unavailable: %s", exc)
            self._facts = None

        # Catalog is regenerated only on startup and on writes — never on
        # recall, so the hot path stays free of filesystem walks.
        self._refresh_catalog()

    def shutdown(self) -> None:
        with self._lock:
            backend = self._backend
            self._backend = None
        if backend is not None and hasattr(backend, "close"):
            try:
                backend.close()
            except Exception as exc:
                logger.debug("wiki memory: close failed: %s", exc)

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id
        if kwargs.get("reset"):
            self._writes_this_session = 0
        self._pending.pop(new_session_id, None)
        if self._facts is not None:
            with self._facts._state_lock:
                self._facts._session_state.pop(new_session_id, None)

    # The remaining ABC hooks are deliberate no-ops. They are declared
    # explicitly (rather than inherited) so the plugin behaves identically
    # when imported outside a Hermes checkout, where MemoryProvider is a shim.

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """No per-turn bookkeeping needed — recall is index-driven."""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """End-of-session structural tidy (no transcript ingestion).

        Auto-summarising a transcript into the wiki would degrade it into a
        chat log. The tidy pass merges duplicate pending proposals, expires
        stale ones, promotes accepted-but-unsaved facts, and refreshes the
        fact-graph canvas. Diary + summary come from the plugin-level
        on_session_finalize hook, which owns the session state.
        """
        if self._facts is not None and self._facts.read_only is False:
            try:
                self._facts.session_tidy(self._session_id, write_diary=False)
            except Exception as exc:
                logger.debug("wiki memory: session tidy failed: %s", exc)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Contribute nothing to the compression summary."""
        return ""

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "",
                      **kwargs) -> None:
        """Subagent output is not auto-filed; the parent decides what matters."""

    def backup_paths(self) -> List[str]:
        """The wiki usually lives outside HERMES_HOME — declare it for backups."""
        raw = _expand(str(self._config.get("wiki_path") or ""), self._hermes_home)
        return [raw] if raw and Path(raw).is_dir() else []

    # -- indexing ----------------------------------------------------------

    def _maybe_reindex(self, *, force: bool = False) -> None:
        """Refresh the index at most once per ``reindex_interval_s``."""
        if self._backend is None:
            return
        interval = float(self._config.get("reindex_interval_s") or 60)
        if not force and (time.time() - self._last_index) < interval:
            return
        if not self._lock.acquire(blocking=False):
            return  # another thread is already syncing
        try:
            self._backend.sync()
            self._last_index = time.time()
        except Exception as exc:
            logger.debug("wiki memory: reindex failed: %s", exc)
        finally:
            self._lock.release()

    # -- catalogs (index.md / log.md) --------------------------------------

    def _refresh_catalog(self, *, kind: str = "", subject: str = "",
                         detail: str = "") -> None:
        """Regenerate ``index.md`` and optionally append to ``log.md``.

        Every failure is swallowed: these files are derived artifacts and a
        catalog problem must never break recall or an accepted write.
        """
        if not (self._catalog_enabled and CATALOG_AVAILABLE) or self._store is None:
            return
        try:
            if self._index_path is not None:
                write_index(self._store, path=self._index_path,
                            memory_dir=self._store.memory_dir)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("wiki memory: index regeneration failed: %s", exc)
        if kind and self._log_path is not None:
            try:
                append_log(self._store, kind, subject, detail=detail,
                           path=self._log_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("wiki memory: log append failed: %s", exc)

    # -- system prompt -----------------------------------------------------

    def system_prompt_block(self) -> str:
        if self._init_error:
            return (
                "# Wiki Memory\n"
                f"INACTIVE — {self._init_error}\n"
                "Long-term recall is unavailable this session; say so rather than "
                "claiming you have no memory of a topic."
            )
        if self._backend is None or self._store is None:
            return ""
        try:
            counts = self._backend.count()
        except Exception:
            counts = {"pages": 0, "chunks": 0}
        memory_dir = self._store.memory_dir
        block = (
            "# Wiki Memory\n"
            f"Long-term memory is a markdown wiki at `{self._store.root}` "
            f"({counts.get('pages', 0)} pages indexed, backend={self._backend.name}).\n"
            "- `wiki_memory_search` — search it BEFORE answering anything durable notes may cover.\n"
            "- `wiki_memory_read` — open a page in full and follow its [[wikilinks]].\n"
            "- `wiki_memory_write` — file durable facts (preferences, decisions, "
            "stable environment facts). Never secrets or task progress.\n"
            "- `memory_propose` — propose a durable fact for user review; accepted facts "
            "become notes in the memory zone.\n"
            "- `memory_recall` — search approved facts + the vault in one call.\n"
            f"Automatic writes land in `{memory_dir}/`; curated pages are edited by hand only."
        )
        if self._facts is not None:
            profile = self._facts.profile_block(
                max_chars=int((self._config.get("facts") or {}).get(
                    "injection", {}).get("max_chars", 3500)),
                max_facts=int((self._config.get("facts") or {}).get(
                    "injection", {}).get("max_facts", 25)))
            if profile:
                block += "\n\n" + profile
        return block

    # -- recall ------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._config.get("auto_recall", True):
            return ""
        cached = self._pending.pop(session_id or self._session_id, "")
        wiki_part = cached or self._recall(query)
        return self._fuse_recall(query, wiki_part)

    def _fuse_recall(self, query: str, wiki_part: str) -> str:
        """Per-turn dynamic injection: approved facts + builtin lines + wiki."""
        blocks: List[tuple] = []  # (priority, text); lower priority number = kept first

        def add(text: str, priority: int) -> None:
            if text:
                blocks.append((priority, text))

        if self._facts is not None:
            try:
                fact_hits = self._facts.recall_facts(query, limit=6)
                if fact_hits:
                    neighbors = {}
                    for hit in fact_hits[:3]:
                        found = self._facts.store().neighbors([hit["id"]], limit=2)
                        if found:
                            neighbors[hit["id"]] = found
                    add(self._facts._format_fact_recall(fact_hits, neighbors=neighbors), 0)
                add(self._open_question_hint(query), 1)
                add(self._pair_hint(query, "conflicts"), 1)
                add(self._pair_hint(query, "duplicates"), 1)
                builtin_lines = self._facts.builtin_recall_lines(query, self._facts.store().embedder)
                if builtin_lines:
                    add("**From Hermes builtin memory (read-only):**\n"
                        + "\n".join(builtin_lines), 2)
            except Exception as exc:
                logger.debug("wiki memory: fact recall fusion failed: %s", exc)
        add(wiki_part, 3)
        kept = self._apply_token_budget(blocks)
        if not kept:
            return ""
        return "## Memory (recalled)\n\n" + "\n\n".join(text for _p, text in kept)

    def _apply_token_budget(self, blocks: List[tuple]) -> List[tuple]:
        """Keep the highest-value blocks until the token budget is spent.

        Honcho's ``context()`` takes a token limit; this used to cap by hit count, which
        says nothing about prompt size. The estimate is deliberately crude — 4 characters
        per token — because it has to cost nothing on the turn path. Whole blocks are
        dropped, never cut mid-sentence, and the caller's display order is preserved.
        """
        try:
            budget = int(self._config.get("max_context_tokens") or 0)
        except (TypeError, ValueError):
            budget = 0
        if budget <= 0:
            return blocks
        order = sorted(range(len(blocks)), key=lambda i: (blocks[i][0], i))
        kept: List[int] = []
        spent = 0
        for index in order:
            cost = _estimate_tokens(blocks[index][1])
            if kept and spent + cost > budget:
                continue          # skip this block, still try smaller higher-priority ones
            kept.append(index)
            spent += cost
        chosen = set(kept)
        return [blocks[i] for i in range(len(blocks)) if i in chosen]

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Warm the next turn's recall in the background."""
        if not self._config.get("auto_recall", True) or self._backend is None:
            return
        key = session_id or self._session_id

        def _run() -> None:
            try:
                self._maybe_reindex()
                result = self._fuse_recall(query, self._recall(query))
                if result:
                    self._pending[key] = result
            except Exception as exc:
                logger.debug("wiki memory: queue_prefetch failed: %s", exc)

        threading.Thread(target=_run, daemon=True, name="wiki-memory-prefetch").start()

    _PAIR_HINTS = {
        # meta key -> (heading, closing line). Both hints quote the pair and refuse to
        # rank it: the memory is not entitled to decide which claim is right.
        "conflicts": ("**Two stored facts disagree — ask which is right:**",
                      "Neither has been changed."),
        "duplicates": ("**This fact is recorded twice — do not add a third copy:**",
                       "Both pages are still here; consolidating them is a review step."),
    }

    def _pair_hint(self, query: str, kind: str) -> str:
        """Raise a pair the maintenance tick recorded (conflict or duplicate).

        The tick stores what it found, so the read path pays nothing: two meta keys, no
        sweep, no model. Similarity is containment against the union of both claims,
        because the query is short and a stored claim is long.
        """
        heading, closing = self._PAIR_HINTS[kind]
        try:
            import json as _json
            from . import maintain as maintain_mod
        except ImportError:  # pragma: no cover - loaded as a top-level module
            try:
                import json as _json
                import maintain as maintain_mod  # type: ignore
            except ImportError:
                return ""
        meta_key = (maintain_mod.META_CONFLICTS if kind == "conflicts"
                    else maintain_mod.META_DUPLICATES)
        try:
            pairs = _json.loads(self._facts.store().get_meta(meta_key) or "[]")
        except Exception:
            return ""
        if not isinstance(pairs, list) or not pairs:
            return ""
        try:
            tokenise = facts_mod._tokens
        except AttributeError:
            logger.debug("wiki memory: token helper unavailable; no %s hint", kind)
            return ""
        query_tokens = set(tokenise(query))
        if not query_tokens:
            return ""
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            a, b = pair.get("a"), pair.get("b")
            if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
                continue
            reach = set(tokenise(a)) | set(tokenise(b))
            shared = query_tokens & reach
            if not shared:
                continue
            score = len(shared) / max(1, min(len(query_tokens), len(reach)))
            if score < 0.6:
                continue
            return (f"{heading}\n"
                    f"- \"{a[:160]}\"\n"
                    f"- \"{b[:160]}\"\n"
                    f"{closing}")
        return ""

    def _open_question_hint(self, query: str) -> str:
        """Raise an unanswered question this conversation is about.

        Maintenance records what recall could not answer. Until now nothing ever
        surfaced those, so the memory knew it was missing something and said nothing.
        Here a relevant gap is offered as a question to ask, never as a fact to assert:
        the agent should ask the user, not guess.
        """
        try:
            import json as _json
            from . import maintain as maintain_mod
        except ImportError:  # pragma: no cover - loaded as a top-level module
            try:
                import json as _json
                import maintain as maintain_mod  # type: ignore
            except ImportError:
                return ""
        try:
            questions = _json.loads(
                self._facts.store().get_meta(maintain_mod.META_OPEN_QUESTIONS) or "[]")
        except Exception:
            return ""
        if not questions:
            return ""
        # facts_mod._tokens, not a bare _tokens: this module never imported the
        # helper, and a broad `except Exception` around it turned that NameError into
        # a silently empty hint. A bug here must be visible, not swallowed.
        try:
            tokenise = facts_mod._tokens
        except AttributeError:
            logger.debug("wiki memory: token helper unavailable; no gap hint")
            return ""
        query_tokens = set(tokenise(query))
        if not query_tokens:
            return ""
        scored = []
        for question in questions:
            if not isinstance(question, str):
                continue
            question_tokens = set(tokenise(question))
            shared = query_tokens & question_tokens
            if not shared:
                continue
            # Containment, not jaccard: the query is short and a stored question is
            # long, so jaccard would punish an exact topical match ("what is the
            # escalation policy?" against a question that names SEV1 as well).
            score = len(shared) / max(1, min(len(query_tokens), len(question_tokens)))
            if score >= 0.6:
                scored.append((score, question))
        if not scored:
            return ""
        scored.sort(reverse=True)
        lines = [f"- {question[:160]}" for _score, question in scored[:2]]
        return ("**Memory gaps this looks like (ask, do not assume):**\n"
                + "\n".join(lines))

    def _recall(self, query: str) -> str:
        if self._backend is None:
            return ""
        query = (query or "").strip()
        if len(query) < int(self._config.get("min_query_chars") or 8):
            return ""
        self._maybe_reindex()
        try:
            hits = self._backend.search(
                query, limit=int(self._config.get("prefetch_limit") or 4)
            )
        except Exception as exc:
            logger.debug("wiki memory: recall failed: %s", exc)
            return ""
        if not hits:
            return ""

        budget = int(self._config.get("prefetch_max_chars") or 1400)
        lines = ["## Wiki Memory (recalled)"]
        used = 0
        for hit in hits:
            label = f"{hit.title or hit.path}"
            if hit.heading and hit.heading != hit.title:
                label += f" › {hit.heading}"
            snippet = hit.snippet[: max(0, budget - used)]
            if not snippet:
                break
            lines.append(f"- **{label}** (`{hit.path}`): {snippet}")
            used += len(snippet)
            if used >= budget:
                break
        lines.append("_Use wiki_memory_read for the full page before relying on an excerpt._")
        return "\n".join(lines)

    # -- writes ------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None) -> None:
        """No automatic transcript ingestion.

        Conversation dumps would bury curated synthesis under noise and make
        the wiki unreadable to a human — the whole point of this backend. The
        agent files facts explicitly via ``wiki_memory_write``.
        """
        return

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror built-in MEMORY.md/USER.md additions into the wiki."""
        if action != "add" or not content:
            return
        if self._read_only:
            return  # cron/subagent builtin writes must not reach the wiki
        if not self._config.get("mirror_builtin_memory", True):
            return
        if self._store is None:
            return
        if scan_for_secrets(content):
            logger.warning("wiki memory: skipped mirroring a secret-shaped builtin entry")
            return
        # This signature declares `metadata`, so MemoryManager's
        # _provider_memory_write_metadata_mode() resolves to "keyword" mode and
        # passes provenance (write_origin, session_id, platform, tool_name...).
        origin = ""
        if metadata:
            origin = str(metadata.get("write_origin") or metadata.get("platform") or "")
        try:
            self._store.append_note(
                content,
                page=str(self._config.get("mirror_page") or "builtin-mirror"),
                topic="Built-in memory mirror",
                source=f"builtin:{target}" + (f" via {origin}" if origin else ""),
            )
        except Exception as exc:
            logger.debug("wiki memory: mirror failed: %s", exc)

    # -- tools -------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = [WIKI_SEARCH_SCHEMA, WIKI_READ_SCHEMA, WIKI_WRITE_SCHEMA,
                   WIKI_FILE_SCHEMA, WIKI_STATUS_SCHEMA, WIKI_CONFIDENCE_SCHEMA]
        if self._read_only:
            # Don't advertise a tool we will refuse — a cron/subagent run
            # shouldn't waste turns discovering the write path is closed.
            schemas = [s for s in schemas
                       if s is not WIKI_WRITE_SCHEMA and s is not WIKI_FILE_SCHEMA]
        if self._facts is not None:
            for fact_schema in self._facts.tool_schemas(self._read_only):
                full = {"name": fact_schema["name"],
                        "description": fact_schema["description"],
                        "parameters": fact_schema["parameters"]}
                schemas.append(full)
        return schemas

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._store is None or self._backend is None:
            return tool_error(
                self._init_error or "wiki memory is not initialized (no wiki_path configured)"
            )
        try:
            if tool_name == "wiki_memory_search":
                return self._tool_search(args)
            if tool_name == "wiki_memory_read":
                return self._tool_read(args)
            if tool_name == "wiki_memory_write":
                return self._tool_write(args)
            if tool_name == "wiki_memory_file":
                return self._tool_file(args)
            if tool_name == "wiki_memory_status":
                return self._tool_status(args)
            if tool_name == "wiki_memory_confidence":
                return self._tool_confidence(args)
            if self._facts is not None and tool_name.startswith("memory_"):
                result = self._facts.handle_tool_call(
                    tool_name, args, tool_error,
                    user_text=str(args.get("_user_text") or ""),
                    session_id=self._session_id)
                if result is not None:
                    return result
        except KeyError as exc:
            return tool_error(f"missing required argument: {exc}")
        except ValueError as exc:
            return tool_error(str(exc))
        except Exception as exc:
            logger.exception("wiki memory: tool %s failed", tool_name)
            return tool_error(f"{tool_name} failed: {exc}")
        return tool_error(f"unknown tool: {tool_name}")

    def _tool_search(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        limit = max(1, min(int(args.get("limit") or 6), 20))
        self._maybe_reindex()
        hits = self._backend.search(query, limit=limit, spine=str(args.get("spine") or ""))
        return json.dumps({
            "success": True,
            "query": query,
            "backend": self._backend.name,
            "count": len(hits),
            "results": [hit.to_dict() for hit in hits],
            "hint": "Call wiki_memory_read for the full page before relying on a snippet.",
        }, ensure_ascii=False)

    def _tool_read(self, args: Dict[str, Any]) -> str:
        reference = str(args.get("page") or "").strip()
        if not reference:
            return tool_error("page is required")
        page = self._store.resolve_page(reference)
        if page is None:
            return tool_error(f"page not found: {reference}")
        max_chars = max(200, min(int(args.get("max_chars") or 8000), 40000))
        body = page.body
        truncated = len(body) > max_chars
        _, backlinks = self._store.link_graph()
        return json.dumps({
            "success": True,
            "path": page.path,
            "title": page.title,
            "frontmatter": {k: v for k, v in page.frontmatter.items()},
            "links": page.links(),
            "backlinks": sorted(set(backlinks.get(page.slug, []))),
            "truncated": truncated,
            "content": body[:max_chars],
        }, ensure_ascii=False, default=str)

    def _tool_write(self, args: Dict[str, Any]) -> str:
        if self._read_only:
            return tool_error(
                f"wiki memory is read-only in agent_context='{self._agent_context}'. "
                "Only the primary agent may write durable facts."
            )
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        tags = args.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        result = self._store.append_note(
            content,
            topic=str(args.get("topic") or ""),
            page=str(args.get("page") or ""),
            tags=list(tags),
            source=str(args.get("source") or ""),
            today=date.today().isoformat(),
        )
        self._writes_this_session += 1
        self._maybe_reindex(force=True)
        self._refresh_catalog(
            kind="write",
            subject=str(args.get("topic") or args.get("page") or result["path"]),
            detail=("created page" if result["created"] else "appended note")
                   + (" (duplicate, skipped)" if result["duplicate"] else ""),
        )
        return json.dumps({
            "success": True,
            "path": result["path"],
            "created_page": result["created"],
            "duplicate": result["duplicate"],
            "note": ("An equivalent fact was already recorded; nothing appended."
                     if result["duplicate"] else "Fact filed to long-term memory."),
        }, ensure_ascii=False)

    def _tool_file(self, args: Dict[str, Any]) -> str:
        """File a synthesised answer back into the wiki as a reusable page."""
        if self._read_only:
            return tool_error(
                f"wiki memory is read-only in agent_context='{self._agent_context}'. "
                "Only the primary agent may file answers into the wiki."
            )
        tags = args.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        # The filing itself lives in facts.file_answer: the CLI verb uses the
        # same code, so the page format cannot drift between the two.
        result = facts_mod.file_answer(
            self._store.memory_root(),
            args.get("question") or "",
            args.get("answer") or "",
            tags,
        )
        if not result.get("success"):
            return tool_error(str(result.get("error") or "could not file answer"))
        if result.get("created_page"):
            self._writes_this_session += 1
            self._maybe_reindex(force=True)
            self._refresh_catalog(kind="file", subject=str(result["path"])[:120],
                                  detail="filed an answer")
        result["path"] = self._store.relpath(Path(result["path"]))
        return json.dumps(result, ensure_ascii=False)

    def _tool_confidence(self, args: Dict[str, Any]) -> str:
        """Advisory: is this retrieval trustworthy? Never changes what recall does."""
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        if self._facts is None:
            return tool_error(
                "the facts layer is not active, so there is no confidence signal; "
                "use wiki_memory_search instead"
            )
        limit = max(1, min(int(args.get("limit") or 8), 20))
        try:
            conf = self._facts.store().recall_confidence(query, limit=limit)
        except Exception as exc:
            logger.debug("wiki memory: confidence failed: %s", exc)
            return tool_error(f"could not measure confidence: {exc}")
        payload: Dict[str, Any] = {"success": True}
        payload.update(conf)
        payload["advice"] = (
            "Say what you know with its score, or abstain and ask. Do not treat "
            "`confident: true` as proof the fact answers the question."
        )
        return json.dumps(payload, ensure_ascii=False)

    def _tool_status(self, args: Dict[str, Any]) -> str:
        self._maybe_reindex()
        counts = self._backend.count() if hasattr(self._backend, "count") else {}
        memory_root = self._store.memory_root()
        memory_pages = (
            sorted(p.name for p in memory_root.glob("*.md")) if memory_root.is_dir() else []
        )
        payload: Dict[str, Any] = {
            "success": True,
            "wiki_root": str(self._store.root),
            "backend": self._backend.name,
            "indexed": counts,
            "memory_dir": self._store.memory_dir,
            "memory_pages": memory_pages,
            "untrusted_dirs_excluded": list(self._store.untrusted_dirs),
            "writes_this_session": self._writes_this_session,
        }
        if args.get("lint"):
            dangling = self._store.dangling_links()
            _out, backlinks = self._store.link_graph()
            orphans = sorted(
                page.slug for page in self._store.iter_pages()
                if not backlinks.get(page.slug) and page.slug != "index"
            )
            payload["lint"] = {
                "dangling_links": dangling,
                "orphan_pages": orphans[:50],
                "orphan_count": len(orphans),
            }
        if self._catalog_enabled and CATALOG_AVAILABLE and self._store is not None:
            entries: List[Dict[str, str]] = []
            try:
                entries = read_log_tail(self._store, limit=5, path=self._log_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("wiki memory: log read failed: %s", exc)
            payload["catalog"] = {
                "enabled": True,
                "index": (self._store.relpath(self._index_path)
                          if self._index_path is not None and self._index_path.exists()
                          else None),
                "log": (self._store.relpath(self._log_path)
                        if self._log_path is not None and self._log_path.exists()
                        else None),
                "recent_log": entries,
            }
        return json.dumps(payload, ensure_ascii=False)

    # -- setup -------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Minimal wizard: only what the user must decide."""
        return [
            {
                "key": "wiki_path",
                "description": "Absolute path to your markdown wiki root",
                "required": True,
                "default": str(Path.home() / "wiki"),
            },
            {
                "key": "backend",
                "description": "Search backend (fts5 = stdlib, fast; qmd = hybrid semantic, needs qmd CLI)",
                "default": "fts5",
                "choices": ["fts5", "qmd"],
            },
            {
                "key": "memory_dir",
                "description": "Wiki-relative folder the agent may write to",
                "default": "wiki/memory",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = save_config_file(values, hermes_home)
        print(f"\n  hwiki config written to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Called by the Hermes memory plugin discovery system."""
    provider = WikiMemoryProvider(config=load_config())
    ctx.register_memory_provider(provider)

    # plugin-level lifecycle hooks for the fact write path (extraction needs
    # the per-turn side-question window; providers alone don't get one)
    def _wire_hooks():
        if provider._facts is None:
            return
        ctx.register_hook("post_llm_call", provider._facts.hook_handler(ctx))

        def _on_session_start(**kwargs):
            provider._facts.sync_from_vault()

        def _on_session_finalize(**kwargs):
            if not provider._facts.read_only:
                try:
                    provider._facts.session_tidy(
                        str(kwargs.get("session_id") or ""), write_diary=True)
                except Exception as exc:
                    logger.debug("wiki memory: finalize tidy failed: %s", exc)

        ctx.register_hook("on_session_start", _on_session_start)
        ctx.register_hook("on_session_finalize", _on_session_finalize)
        logger.info("wiki memory: fact extraction hooks registered")

    # provider.initialize() runs during agent init; hooks can be registered
    # immediately — the engine lazy-initializes its store on first use
    provider._facts = _maybe_build_facts(provider)
    _wire_hooks()


def _maybe_build_facts(provider: "WikiMemoryProvider") -> Optional[FactsEngine]:
    """Build the facts engine early (for hook registration); paths come from
    config, availability errors are deferred to initialize()."""
    try:
        raw_path = _expand(str(provider._config.get("wiki_path") or ""),
                           provider._hermes_home)
        if not raw_path:
            return None
        engine = FactsEngine(
            provider._hermes_home, Path(raw_path),
            Path(str(provider._config.get("memory_dir") or "wiki/memory")),
            config=dict(provider._config.get("facts") or {}))
        engine.store()  # opens + migrates + heals
        return engine
    except Exception as exc:
        logger.warning("wiki memory: fact store unavailable at registration: %s", exc)
        return None
