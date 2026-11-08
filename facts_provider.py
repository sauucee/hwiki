"""Fact-write integration for the hwiki memory provider.

This is the review-gated write path (ported from the standalone
hwiki-extractor plugin at merge, v2.0): per-turn extraction proposes durable
facts, the user accepts by ID, and accepted facts become one Obsidian note
each in the vault's memory zone. As part of the provider it gains what a
separate plugin could never have: per-turn dynamic injection of accepted
facts (via ``prefetch``), compression-window participation, and one tool
surface for facts + wiki.

Raw conversations are never stored: proposals carry claim + evidence only,
and the session diary is a structured summary.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .facts import (
    ProposalStore, ProposalRejected, Embedder, hermes_home,
    load_builtin_entries, read_fact_pages, render_profile, write_fact_page,
    mark_superseded_page, write_session_diary, append_audit_entry,
    write_fact_canvas, resolve_vault, _tokens, _cosine_bytes,
)

logger = logging.getLogger(__name__)

REVIEW_CADENCE_TURNS = 5
REVIEW_HISTORY_BUDGET_CHARS = 12000
MAX_DEFERRED_TURNS = 6
MAX_DEFERRED_CHARS = 6000

DEFAULT_CONFIG = {
    "injection": {"enabled": True, "max_chars": 3500, "max_facts": 25},
    "review": {"cadence_turns": REVIEW_CADENCE_TURNS, "decay_days": 90,
               "proposal_ttl_days": 14},
    "diary": {"enabled": True},
    "embeddings": {"mode": "auto"},
}

EXTRACT_PROMPT = (
    "Extract any durable facts from this conversation. Return a JSON array of objects with "
    "kind, claim, evidence, and peer fields. Only extract facts that are stable and worth "
    "remembering across sessions. Return empty array if nothing to extract. "
    "Kind should be one of: preference, decision, environment, milestone, blocker, correction. "
    "Peer is 'user' for facts about the user, or 'assistant' for facts about you (the assistant): "
    "your name, standing instructions, settled behaviors, capabilities or limits."
)

_FACTS_PROMPT_NOTE = (
    "\n\nAlready-known accepted facts (do NOT restate these; if one is outdated, propose "
    "kind=correction with evidence starting 'Replaces: <old claim>'):\n{facts}"
)

_FEEDBACK_GUIDANCE_HEADER = (
    "\n\nThe user's recent accept/reject decisions (learn from this):\n{hints}"
)

REVIEW_PROMPT_PREFIX = (
    "You are reviewing a session so far (reread the conversation). Current accepted memory facts:\n"
    "{facts}\n"
    "{last_session}"
    'Return JSON: {{"summary": "2-3 sentence factual summary of this session", '
    '"facts": [{{"kind": "...", "claim": "...", "evidence": "...", "peer": "user|assistant"}}], '
    '"stale": [{{"claim": "existing fact that is now outdated", "reason": "why", '
    '"replacement": "corrected claim"}}]}}. '
    "Facts must be stable and durable; omit trivial or transient content. "
    "Peer is 'user' for facts about the user, 'assistant' for facts about you. "
    "For stale facts propose a correction whose evidence starts with 'Replaces: <old claim>'. "
    "Return empty lists if nothing to add."
)

FACT_TOOL_SCHEMAS = {
    "memory_propose": {
        "description": (
            "Propose a durable fact for review. Flags possible conflicts with existing facts and "
            "duplicates of builtin memory. Secret-shaped claims are rejected. Use peer='assistant' "
            "for facts about the assistant itself, 'user' (default) for facts about the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "Kind of fact: preference, decision, environment, milestone, blocker, correction."},
                "claim": {"type": "string", "description": "The claim to record."},
                "evidence": {"type": "string", "description": "Brief evidence or rationale. For corrections, start with 'Replaces: <old claim>'."},
                "peer": {"type": "string", "enum": ["user", "assistant"], "description": "Who the fact is about.", "default": "user"},
            },
            "required": ["kind", "claim", "evidence"],
        },
    },
    "memory_accept": {
        "description": "Accept a pending fact proposal: it becomes an Obsidian note in the vault's memory zone. Idempotent.",
        "parameters": {
            "type": "object",
            "properties": {"proposal_id": {"type": "string", "description": "The proposal ID to accept."}},
            "required": ["proposal_id"],
        },
    },
    "memory_reject": {
        "description": "Reject a pending fact proposal. Idempotent.",
        "parameters": {
            "type": "object",
            "properties": {"proposal_id": {"type": "string", "description": "The proposal ID to reject."}},
            "required": ["proposal_id"],
        },
    },
    "memory_supersede": {
        "description": "Replace an accepted fact with a newer one: the old note is tombstoned in the vault with a link to the replacement. User-confirmed healing.",
        "parameters": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string", "description": "New (already accepted) proposal ID."},
                "supersede_id": {"type": "string", "description": "Old accepted proposal ID to replace."},
            },
            "required": ["proposal_id", "supersede_id"],
        },
    },
    "memory_forget": {
        "description": (
            "Revoke an accepted fact: its vault note is deleted and it leaves memory. Requires the "
            "exact proposal_id OR the exact claim text (no fuzzy matching — never guess)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string", "description": "Exact proposal ID of the accepted fact."},
                "claim": {"type": "string", "description": "Exact claim text (used only when proposal_id is absent)."},
            },
            "required": [],
        },
    },
    "memory_list": {
        "description": "List pending, accepted, or rejected fact proposals with conflict/duplicate annotations.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["proposed", "accepted", "rejected"],
                           "description": "Filter status.", "default": "proposed"},
            },
            "required": [],
        },
    },
    "memory_recall": {
        "description": (
            "Search approved memory facts and the whole vault for anything relevant to a question. "
            "Call this before answering from memory when the answer is not already in context. "
            "Returns facts with acceptance dates and supersede lineage, plus matching vault notes. "
            "Pass `project` to scope to one project."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up."},
                "project": {"type": "string", "description": "Optional project scope."},
                "peer": {"type": "string", "description": "Optional peer scope (user/assistant)."},
            },
            "required": ["query"],
        },
    },
    "memory_review_now": {
        "description": "Schedule the deep memory review pass for the next turn: rereads session history against accepted facts, proposes missed facts and corrections for stale ones.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

FACT_TOOLS_READ_ONLY = {"memory_propose", "memory_accept", "memory_supersede",
                        "memory_forget", "memory_review_now"}
FACT_TOOLS = list(FACT_TOOL_SCHEMAS)


def _parse_json_array(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text, flags=re.M).strip()
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        return None
    try:
        return json.loads(text[min(starts):])
    except json.JSONDecodeError:
        return None


class FactsEngine:
    """Owns the fact store, extraction hooks, injection profile and tools.

    One instance per provider; all vault paths come from the provider's
    WikiStore so the vault stays the single source of truth.
    """

    def __init__(self, hermes_home: str, vault_root: Path, memory_rel: Path,
                 config: Optional[Dict[str, Any]] = None):
        self.hermes_home = Path(hermes_home) if hermes_home else hermes_home()
        self.vault_root = Path(vault_root)
        self.memory_rel = Path(memory_rel)
        self.config = config or {}
        self._store: Optional[ProposalStore] = None
        self._builtin_cache = {"mtime": 0.0, "entries": []}
        self._builtin_lock = threading.Lock()
        self._facts_prompt_cache = {"mtime": 0.0, "lines": ""}
        self._session_state: "OrderedDict[str, dict]" = OrderedDict()
        self._state_lock = threading.Lock()
        self.read_only = False

    # -- store + config ----------------------------------------------------

    def data_dir(self) -> Path:
        return self.hermes_home / "plugins-data" / "hwiki"

    def store(self) -> ProposalStore:
        if self._store is None:
            data_dir = self.data_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            self._migrate_legacy()
            mode = self.config.get("embeddings", {}).get("mode", "auto")
            self._store = ProposalStore(str(data_dir / "facts.sqlite"),
                                        embedder=Embedder(mode=mode))
            self._heal_from_vault()
        return self._store

    def _migrate_legacy(self) -> None:
        """First run after the merge: adopt the standalone extractor's state."""
        legacy_db = self.hermes_home / "plugins-data" / "hwiki-extractor" / "proposals.sqlite"
        new_db = self.data_dir() / "facts.sqlite"
        if legacy_db.exists() and not new_db.exists():
            try:
                import shutil
                shutil.copy2(legacy_db, new_db)
                logger.info("facts: migrated legacy extractor DB (%s)", legacy_db)
            except Exception:
                logger.debug("facts: legacy DB migration failed", exc_info=True)

    def _heal_from_vault(self) -> None:
        store = self._store
        try:
            if not store.list_proposals(status='accepted', limit=1) \
                    and not store.list_proposals(status='proposed', limit=1):
                summary = store.rebuild_from_vault(self.vault_root, self.memory_rel)
                if summary["imported"]:
                    logger.warning("facts DB was empty — rebuilt %d fact(s) from the vault: %s",
                                   summary["imported"], summary)
        except Exception:
            logger.exception("facts: vault rebuild check failed")

    def config_value(self, section, key, default):
        return (self.config.get(section) or {}).get(key, default)

    # -- session state -----------------------------------------------------

    def _session(self, session_id):
        with self._state_lock:
            state = self._session_state.get(session_id)
            if state is None:
                state = {"turns": 0, "since_review": 0, "chars_since_review": 0,
                         "topics": {}, "user_chars": 0, "summary": "", "deferred": []}
                self._session_state[session_id] = state
                while len(self._session_state) > 8:
                    self._session_state.popitem(last=False)
            else:
                self._session_state.move_to_end(session_id)
            return state

    def _top_topics(self, state, count=8):
        return [t for t, _ in sorted(state["topics"].items(), key=lambda kv: kv[1], reverse=True)[:count]]

    # -- builtin memory (read-only dedup + recall) --------------------------

    def _builtin_entries(self):
        memories_dir = self.hermes_home / "memories"
        newest = 0.0
        for name in ("MEMORY.md", "USER.md"):
            try:
                newest = max(newest, (memories_dir / name).stat().st_mtime)
            except OSError:
                pass
        with self._builtin_lock:
            if newest and newest != self._builtin_cache["mtime"]:
                self._builtin_cache["entries"] = load_builtin_entries(memories_dir)
                self._builtin_cache["mtime"] = newest
            return self._builtin_cache["entries"]

    def builtin_recall_lines(self, query, embedder, limit=3, threshold=0.35):
        try:
            entries = self._builtin_entries()
        except Exception:
            return []
        if not entries:
            return []
        query_toks = set(_tokens(query))
        query_vec = embedder.embed(query)
        scored = []
        for entry in entries:
            toks = set(_tokens(entry["text"]))
            overlap = len(query_toks & toks) / max(1, len(query_toks | toks))
            cos = _cosine_bytes(query_vec, embedder.embed(entry["text"]))
            score = max(cos, 0.5 * cos + 0.5 * overlap)
            if score >= threshold:
                scored.append((score, entry))
        scored.sort(key=lambda p: p[0], reverse=True)
        return [f"- (builtin {e['target']}.md) {e['text'][:180]}" for _, e in scored[:limit]]

    # -- promotion / tidy / canvas -----------------------------------------

    def _promote(self, proposal_id, proposal=None):
        store = self.store()
        try:
            if not proposal:
                try:
                    proposal = store.get(proposal_id)
                except ValueError:
                    proposal = None
            if not proposal or proposal.get("status") != "accepted":
                return None
            if proposal.get("page_path") and Path(proposal["page_path"]).exists():
                return proposal["page_path"]
            supersedes_claim = ""
            if proposal.get("kind") == "correction" and proposal.get("related"):
                top = proposal["related"][0]
                supersedes_claim = str(top.get("text", ""))[:200] if isinstance(top, dict) else ""
            page = write_fact_page(self.vault_root, self.memory_rel, proposal,
                                   supersedes_claim=supersedes_claim)
            store.set_page_path(proposal_id, page)
            return page
        except Exception:
            logger.exception("facts: promote failed for %s", proposal_id)
            return None

    def _audit(self, text):
        try:
            append_audit_entry(self.vault_root, self.memory_rel, text)
        except Exception:
            logger.debug("facts: audit append failed", exc_info=True)

    def _refresh_canvas(self):
        try:
            write_fact_canvas(self.store(), self.vault_root, self.memory_rel)
        except Exception:
            logger.debug("facts: canvas regeneration failed", exc_info=True)

    def session_tidy(self, session_id, write_diary=True):
        store = self.store()
        ttl = self.config_value("review", "proposal_ttl_days", 14)
        try:
            tidy = store.consolidate(session_id=session_id or None,
                                     proposal_ttl_days=None if self.read_only else ttl)
        except Exception:
            tidy = {"duplicates": 0, "expired": 0}
            logger.debug("facts: consolidate failed", exc_info=True)
        promoted = 0
        try:
            for proposal in store.unpromoted_accepted():
                if self._promote(proposal["id"], proposal):
                    promoted += 1
        except Exception:
            logger.exception("facts: unpromoted-fact promotion failed")
        diary = ""
        if write_diary and session_id and not self.read_only:
            with self._state_lock:
                state = self._session_state.pop(session_id, None)
            if state is not None and self.config_value("diary", "enabled", True):
                try:
                    fact_claims = [f["claim"] for f in read_fact_pages(
                        self.vault_root, self.memory_rel, max_facts=10)]
                    page = write_session_diary(
                        self.vault_root, self.memory_rel, session_id=session_id,
                        summary=state.get("summary", ""), facts=fact_claims,
                        topics=self._top_topics(state))
                    diary = str(page) if page else ""
                    if state.get("summary"):
                        try:
                            store.set_meta('last_session_summary', state["summary"])
                        except Exception:
                            logger.debug("facts: meta write failed", exc_info=True)
                except Exception:
                    logger.exception("facts: session diary failed")
        if (removed := tidy.get("duplicates", 0)) or promoted or tidy.get("expired", 0):
            self._audit(f"session tidy (session={session_id or '-'}): {removed} duplicate(s) removed, "
                        f"{tidy.get('expired', 0)} expired, {promoted} fact(s) promoted")
            self._refresh_canvas()
        return {"duplicates_removed": tidy.get("duplicates", 0),
                "expired": tidy.get("expired", 0),
                "promoted": promoted, "diary": diary}

    def sync_from_vault(self):
        try:
            summary = self.store().sync_from_vault()
            if summary["updated"] or summary["revoked"] or summary["skipped_secrets"]:
                logger.info("facts: vault sync-back %s", summary)
                self._refresh_canvas()
            return summary
        except Exception:
            logger.exception("facts: vault sync-back failed")
            return {"updated": 0, "revoked": 0, "skipped_secrets": 0}

    # -- injection surfaces -------------------------------------------------

    def accepted_fact_lines(self, max_facts=30):
        vault_mtimes = self.vault_root / self.memory_rel
        try:
            mtime = (vault_mtimes).stat().st_mtime
        except OSError:
            mtime = 0.0
        with self._state_lock:
            if mtime != self._facts_prompt_cache["mtime"]:
                try:
                    claims = [f["claim"] for f in read_fact_pages(
                        self.vault_root, self.memory_rel, max_facts=max_facts)]
                    self._facts_prompt_cache["lines"] = "\n".join(f"- {c}" for c in claims)
                except Exception:
                    self._facts_prompt_cache["lines"] = ""
                self._facts_prompt_cache["mtime"] = mtime
            return self._facts_prompt_cache["lines"]

    def profile_block(self, max_chars=3500, max_facts=25):
        """Static per-session profile for the system prompt (kind/peer grouped)."""
        if not self.config_value("injection", "enabled", True):
            return ""
        try:
            facts = read_fact_pages(self.vault_root, self.memory_rel, max_facts=max_facts)
            try:
                usage = self.store().usage_counts()
            except Exception:
                usage = {}
            return render_profile(facts, max_chars=max_chars, usage=usage)
        except Exception:
            logger.exception("facts: profile render failed")
            return ""

    def recall_facts(self, query, limit=8, project=None, peer=None):
        try:
            return self.store().recall(query, limit=limit, project=project or None,
                                       peer=peer or None)
        except Exception:
            logger.debug("facts: vector recall failed", exc_info=True)
            return []

    def feedback_hints(self):
        try:
            feedback = json.loads(self.store().get_meta('feedback') or '{}')
        except Exception:
            return ""
        hints = []
        for kind, counts in sorted(feedback.items()):
            total = counts.get("accepted", 0) + counts.get("rejected", 0)
            if total < 3:
                continue
            rejected = counts.get("rejected", 0)
            if rejected / total >= 0.6:
                hints.append(f"- be very selective with '{kind}' facts — the user rejected "
                             f"{rejected}/{total} of them recently; only propose clear, high-value ones")
            elif rejected / total <= 0.2:
                hints.append(f"- '{kind}' facts are valued ({counts.get('accepted', 0)}/{total} accepted)")
        return "\n".join(hints)

    # -- extraction (side-question window) ----------------------------------

    def _propose_facts(self, ctx, token, prompt, user_message, session_id, project):
        store = self.store()
        try:
            result = ctx.side_question(token, prompt, timeout=25.0)
        except Exception:
            logg = logging.getLogger(__name__)
            logg.debug("facts: side_question failed (token consumed or expired)")
            return None
        if not result:
            return None
        parsed = _parse_json_array(result)
        if isinstance(parsed, list):
            parsed = {"facts": parsed}
        if not isinstance(parsed, dict):
            logger.warning("facts: failed to parse extraction result")
            return None
        for fact in parsed.get("facts") or []:
            if isinstance(fact, dict) and all(k in fact for k in ("kind", "claim", "evidence")):
                try:
                    proposal = store.propose(
                        fact, user_text=user_message or "", session_id=session_id,
                        project=project, builtin_entries=self._builtin_entries())
                    if proposal.get("duplicate_of_builtin"):
                        logger.info("facts: proposal duplicates builtin memory, kept for review: %s",
                                    str(proposal["claim"])[:80])
                except ProposalRejected:
                    logger.warning("facts: rejected secret-shaped proposal: %s",
                                   str(fact.get("claim", ""))[:80])
                except Exception as exc:
                    logger.warning("facts: propose failed: %s", exc)
        for stale in parsed.get("stale") or []:
            if not (isinstance(stale, dict) and stale.get("replacement")):
                continue
            fact = {
                "kind": "correction",
                "claim": str(stale["replacement"])[:300],
                "evidence": f"Replaces: {str(stale.get('claim', ''))[:200]}. Reason: {str(stale.get('reason', ''))[:200]}",
            }
            try:
                store.propose(fact, user_text=user_message or "", session_id=session_id,
                              project=project, builtin_entries=self._builtin_entries())
            except Exception as exc:
                logger.debug("facts: stale-fact proposal failed: %s", exc)
        return parsed

    def hook_handler(self, ctx):
        """post_llm_call handler — extraction + adaptive review passes."""
        store = self.store()
        config = self.config

        def _handle_post_llm_call(**kwargs):
            if self.read_only:
                return
            session_id = kwargs.get('session_id', '')
            user_message = kwargs.get('user_message', '')
            project = kwargs.get('project', '')
            if not user_message or len(user_message.strip()) < 10:
                return

            state = self._session(session_id)
            with self._state_lock:
                state["turns"] += 1
                state["since_review"] += 1
                state["user_chars"] += len(user_message)
                state["chars_since_review"] = state.get("chars_since_review", 0) + len(user_message)
            for token in _tokens(user_message):
                state["topics"][token] = state["topics"].get(token, 0) + 1

            side_question_token = kwargs.get('side_question_token')
            if not side_question_token:
                with self._state_lock:
                    deferred = state.setdefault("deferred", [])
                    deferred.append(user_message)
                    while len(deferred) > MAX_DEFERRED_TURNS or \
                            sum(len(t) for t in deferred) > MAX_DEFERRED_CHARS:
                        deferred.pop(0)
                return

            cadence = max(2, int(config.get("review", {}).get("cadence_turns", REVIEW_CADENCE_TURNS)))
            forced = state.pop("force_review", False)
            review_due = (forced
                          or state["since_review"] >= cadence
                          or (state["since_review"] >= 3 and state.get("chars_since_review", 0) >= 4000))
            if not forced and state.get("chars_since_review", 0) < 200:
                review_due = False
            with self._state_lock:
                deferred = list(state.pop("deferred", []))
            try:
                if review_due:
                    with self._state_lock:
                        state["since_review"] = 0
                        state["chars_since_review"] = 0
                    accepted = [f["claim"] for f in read_fact_pages(
                        self.vault_root, self.memory_rel, max_facts=40)]
                    last_summary = ""
                    try:
                        last_summary = store.get_meta('last_session_summary')
                    except Exception:
                        last_summary = ""
                    last_session = (f"Last session ended with: {last_summary}\n" if last_summary else "")
                    history = kwargs.get("conversation_history") or []
                    blob, used = [], 0
                    for message in history[-40:]:
                        line = f"{message.get('role', '')}: {str(message.get('content', ''))[:1500]}"
                        if used + len(line) > REVIEW_HISTORY_BUDGET_CHARS:
                            break
                        blob.append(line)
                        used += len(line)
                    prompt = (REVIEW_PROMPT_PREFIX.format(
                        facts="\n".join(f"- {c}" for c in accepted) or "- (none)",
                        last_session=last_session)
                        + "\n\nRecent conversation:\n" + "\n".join(blob))
                    decay = []
                    try:
                        decay = store.never_used_older_than(
                            int(config.get("review", {}).get("decay_days", 90)), limit=10)
                    except Exception:
                        decay = []
                    if decay:
                        prompt += ("\n\nOld facts never recalled since acceptance (if the conversation "
                                   "shows one is outdated, propose a correction; otherwise ignore):\n"
                                   + "\n".join(f"- {d}" for d in decay))
                    hints = self.feedback_hints()
                    if hints:
                        prompt += _FEEDBACK_GUIDANCE_HEADER.format(hints=hints)
                    parsed = self._propose_facts(ctx, side_question_token, prompt,
                                                 user_message, session_id, project)
                    if parsed and parsed.get("summary"):
                        state["summary"] = str(parsed["summary"])[:600]
                    elif parsed is None:
                        self._requeue(state, deferred, user_message)
                    else:
                        logger.info("facts: review pass completed for session %s", session_id)
                else:
                    prompt = EXTRACT_PROMPT
                    known = self.accepted_fact_lines()
                    if known:
                        prompt += _FACTS_PROMPT_NOTE.format(facts=known)
                    hints = self.feedback_hints()
                    if hints:
                        prompt += _FEEDBACK_GUIDANCE_HEADER.format(hints=hints)
                    if deferred:
                        prompt += ("\n\nEarlier turns from this session that were not yet processed "
                                   "(extract durable facts from these too):\n"
                                   + "\n".join(f"- {t[:800]}" for t in deferred))
                    parsed = self._propose_facts(ctx, side_question_token, prompt,
                                                 user_message, session_id, project)
                    if parsed is None:
                        self._requeue(state, deferred, user_message)
            except Exception:
                self._requeue(state, deferred, user_message)
                logger.exception("facts: post_llm_call handler failed")
        return _handle_post_llm_call

    def _requeue(self, state, deferred, current_text):
        with self._state_lock:
            merged = list(deferred) + [current_text]
            while len(merged) > MAX_DEFERRED_TURNS or sum(len(t) for t in merged) > MAX_DEFERRED_CHARS:
                merged.pop(0)
            state["deferred"] = merged

    # -- tools ---------------------------------------------------------------

    def tool_schemas(self, read_only: bool) -> List[Dict[str, Any]]:
        schemas = []
        for name in FACT_TOOLS:
            if read_only and name in FACT_TOOLS_READ_ONLY:
                continue
            schema = {"name": name}
            schema.update(FACT_TOOL_SCHEMAS[name])
            schemas.append(schema)
        return schemas

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any],
                         tool_error, **kwargs) -> Optional[str]:
        """Dispatch a fact tool; returns None when the tool is not ours."""
        store = self.store()
        read_only = self.read_only

        def _readonly_error():
            return tool_error("facts are read-only in this agent context "
                              "(only the primary agent may change memory)")

        if tool_name == "memory_propose":
            if read_only:
                return _readonly_error()
            try:
                proposal = store.propose(
                    args, user_text=str(kwargs.get("user_text") or ""),
                    session_id=str(kwargs.get("session_id") or ""),
                    project=str(args.get("project") or ""),
                    builtin_entries=self._builtin_entries())
                return json.dumps({
                    "success": True,
                    "proposal_id": proposal["id"],
                    "status": proposal["status"],
                    "claim": proposal["claim"],
                    "kind": proposal["kind"],
                    "possible_conflict": bool(proposal.get("possible_conflict")),
                    "related": [
                        {"proposal_id": r.get("key"), "claim": r.get("text"), "score": r.get("score")}
                        for r in proposal.get("related", []) if isinstance(r, dict)
                    ],
                    "duplicate_of_builtin": proposal.get("duplicate_of_builtin", ""),
                })
            except ProposalRejected as exc:
                return tool_error(str(exc))
            except Exception as exc:
                logger.exception("facts: propose failed")
                return tool_error(f"propose failed: {exc}")

        if tool_name == "memory_accept":
            if read_only:
                return _readonly_error()
            prop_id = args.get("proposal_id")
            if not prop_id:
                return tool_error("proposal_id is required")
            ok = store.accept(prop_id)
            page = self._promote(prop_id) if ok else None
            if ok:
                try:
                    kind = store.get(prop_id).get("kind", "unknown")
                    store.record_feedback(kind, accepted=True)
                    self._audit(f"accepted fact \"{store.get(prop_id)['claim'][:100]}\" ({kind})")
                    self._refresh_canvas()
                except Exception:
                    logger.debug("facts: accept bookkeeping failed", exc_info=True)
            note = ("Fact is now in memory: recallable immediately and included in the "
                    "session profile from the next session on.") if ok and page else ""
            return json.dumps({"success": ok, "proposal_id": prop_id, "accepted": ok,
                               "page": str(page) if page else "", "note": note})

        if tool_name == "memory_reject":
            if read_only:
                return _readonly_error()
            prop_id = args.get("proposal_id")
            if not prop_id:
                return tool_error("proposal_id is required")
            ok = store.reject(prop_id)
            if ok:
                try:
                    kind = store.get(prop_id).get("kind", "unknown")
                    store.record_feedback(kind, accepted=False)
                    self._audit(f"rejected proposal \"{store.get(prop_id)['claim'][:100]}\" ({kind})")
                except Exception:
                    logger.debug("facts: reject bookkeeping failed", exc_info=True)
            return json.dumps({"success": ok, "proposal_id": prop_id, "rejected": ok})

        if tool_name == "memory_supersede":
            if read_only:
                return _readonly_error()
            new_id, old_id = args.get("proposal_id"), args.get("supersede_id")
            if not new_id or not old_id:
                return tool_error("need proposal_id (new fact) and supersede_id (old fact)")
            ok, old = store.supersede(new_id, old_id)
            if not ok:
                return tool_error("supersede failed: old fact must exist and be accepted")
            new = store.get(new_id)
            if not new.get("page_path"):
                self._promote(new_id)
            tombstoned = mark_superseded_page(old.get("page_path"), new["claim"])
            self._audit(f"superseded \"{old['claim'][:80]}\" with \"{new['claim'][:80]}\"")
            self._refresh_canvas()
            return json.dumps({"success": True, "superseded": old_id, "by": new_id,
                               "old_page_tombstoned": tombstoned})

        if tool_name == "memory_forget":
            if read_only:
                return _readonly_error()
            result = store.forget(proposal_id=args.get("proposal_id"), claim=args.get("claim"))
            if result.get("ok"):
                self._audit(f"forgot fact \"{result.get('forgot', '')[:100]}\"")
                self._refresh_canvas()
            return json.dumps({"success": result.get("ok", False), **result})

        if tool_name == "memory_list":
            status = str(args.get("status") or "proposed")
            props = store.list_proposals(status=status)
            keys = ("id", "kind", "claim", "evidence", "status", "created_at", "peer",
                    "possible_conflict", "related", "duplicate_of_builtin", "page_path")
            return json.dumps({"success": True,
                               "proposals": [{k: p.get(k) for k in keys} for p in props]})

        if tool_name == "memory_recall":
            query = str(args.get("query") or "").strip()
            if len(query) < 2:
                return tool_error("query is required")
            project = str(args.get("project") or "").strip()
            peer = str(args.get("peer") or "").strip().lower()
            fact_hits = store.recall(query, limit=8, project=project or None, peer=peer or None)
            fused = self._fuse_fact_hits(fact_hits)
            neighbors = {}
            for rank, hit in enumerate(fact_hits[:3]):
                found = store.neighbors([hit["id"]], limit=2)
                if found:
                    neighbors[hit["id"]] = found
            rendered = self._format_fact_recall(fused, neighbors=neighbors)
            builtin_lines = self.builtin_recall_lines(query, store.embedder)
            if builtin_lines:
                rendered = ((rendered + "\n\n**From Hermes builtin memory (read-only):**\n"
                             + "\n".join(builtin_lines)) if rendered else
                            "## From Hermes builtin memory (read-only)\n" + "\n".join(builtin_lines))
            return json.dumps({"success": True, "query": query,
                               "fact_hits": len(fact_hits),
                               "result": rendered or "No relevant memory found."})

        if tool_name == "memory_review_now":
            if read_only:
                return _readonly_error()
            session_id = str(kwargs.get("session_id") or "")
            state = self._session(session_id)
            with self._state_lock:
                state["force_review"] = True
            return json.dumps({"success": True, "note": (
                "Deep review scheduled: it runs inside the next turn's thinking window "
                "(the per-turn side-question), reviewing session history, accepted facts, "
                "and never-used old facts.")})

        if tool_name == "memory_status":
            stats = store.stats()
            return json.dumps({
                "success": True,
                "proposals": stats["counts"],
                "accepted_by_peer": stats.get("peers", {}),
                "feedback_learning": stats.get("feedback", {}),
                "last_accepted": stats["last_accepted"],
                "embedding_tier": stats["embedding_tier"],
                "vault": {"root": str(self.vault_root), "memory_dir": str(self.memory_rel),
                          "exists": (self.vault_root / self.memory_rel).is_dir()},
                "injection_enabled": self.config_value("injection", "enabled", True),
                "diary_enabled": self.config_value("diary", "enabled", True),
                "review_cadence_turns": self.config_value("review", "cadence_turns",
                                                          REVIEW_CADENCE_TURNS),
                "decay_days": self.config_value("review", "decay_days", 90),
                "proposal_ttl_days": self.config_value("review", "proposal_ttl_days", 14),
                "read_only": self.read_only,
            })
        return None

    # -- recall fusion -------------------------------------------------------

    def _fuse_fact_hits(self, fact_hits, k=60):
        """Order fact hits by score; the wiki backend ranks its own hits."""
        return sorted(fact_hits, key=lambda h: h.get("score", 0), reverse=True)

    def _format_fact_recall(self, fact_hits, neighbors=None, budget=1400):
        if not fact_hits:
            return ""
        lines = ["### Facts (user-approved)"]
        used = len(lines[0])
        for fact in fact_hits:
            bits = [fact["kind"]]
            if fact.get("peer") == "assistant":
                bits.insert(0, "self")
            if fact.get("project"):
                bits.append(fact["project"])
            if fact.get("accepted"):
                bits.append(f"accepted {fact['accepted']}")
            line = f"- **{fact['claim']}** ({', '.join(bits)})"
            if fact.get("replaced"):
                line += f" ↻ replaced: \"{fact['replaced']}\""
            line += f" [score {fact['score']}]"
            if used + len(line) + 1 > budget:
                break
            lines.append(line)
            used += len(line) + 1
            for nb in (neighbors or {}).get(fact["id"], []):
                tag = "replaced by context" if nb["relation"] == "replaced" else "related"
                nb_line = f"  - ({tag}) **{nb['claim']}** [{nb['peer']}]"
                if used + len(nb_line) + 1 <= budget:
                    lines.append(nb_line)
                    used += len(nb_line) + 1
        lines.append("_Synthesize the answer from these approved facts rather than quoting verbatim._")
        return "\n".join(lines)
