"""Fact store for the hwiki memory provider — the review-gated write path.

Accepted facts become one Obsidian page each inside the vault's memory zone
(the hwiki-indexed source of truth); this SQLite store tracks proposal
lifecycle and local match vectors. Zero external services: a pure-Python
hashed n-gram embedder is always available; an optional local ONNX MiniLM
model upgrades matching when installed (see README "Semantic embeddings").

Ported from the standalone hwiki-extractor plugin when it merged into hwiki
(v2.0); the public API is unchanged.
"""

from contextlib import contextmanager
from collections import OrderedDict

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import threading
import sys
import time
import uuid
from datetime import date, datetime, timezone

try:  # POSIX advisory locks serialize independent hwiki writers.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback is process-local
    fcntl = None
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - import style depends on how the module is loaded
    from . import answerability as answerability_mod
except ImportError:  # loaded as a top-level module
    import answerability as answerability_mod

try:  # pragma: no cover - optional NLI tier, may be absent entirely
    from . import verdict as verdict_mod
except ImportError:
    try:
        import verdict as verdict_mod  # type: ignore
    except ImportError:
        verdict_mod = None


_NLI_DUP_MIN_SCORE = 0.75


try:  # pragma: no cover - optional, only used to record unanswered questions
    from . import maintain as maintain_mod
except ImportError:
    try:
        import maintain as maintain_mod  # type: ignore
    except ImportError:
        maintain_mod = None


# Recall is a read path, so a miss is buffered in memory and flushed by the next
# maintenance tick instead of writing to SQLite on every empty lookup.
_QUESTION_BUFFER = None

# Reentrant per-thread state for ProposalStore.lifecycle_lock().
_LIFECYCLE_LOCK_STATE = threading.local()


def question_buffer():
    """Process-wide buffer of questions recall could not answer."""
    global _QUESTION_BUFFER
    if _QUESTION_BUFFER is None and maintain_mod is not None:
        _QUESTION_BUFFER = maintain_mod.QuestionBuffer()
    return _QUESTION_BUFFER


def _record_miss(query) -> None:
    buffer = question_buffer()
    if buffer is not None:
        try:
            buffer.note(query)
        except Exception:  # pragma: no cover - never let telemetry break recall
            pass


def _nli_conflict_enabled() -> bool:
    """Opt-in only: importing/loading the model is not free."""
    if verdict_mod is None:
        return False
    try:
        return verdict_mod.enabled()
    except Exception:  # pragma: no cover - defensive
        return False


def _nli_dup_enabled() -> bool:
    """Paraphrase-as-duplicate, opt-in separately: it changes what gets stored."""
    if verdict_mod is None:
        return False
    try:
        return verdict_mod.dup_enabled()
    except Exception:  # pragma: no cover - defensive
        return False


# Value-shape detection: reject secret MATERIAL (assignments, real token
# formats, key blobs), not discussion ABOUT secrets — so a rule like
# "never paste API keys in chat" is storable while "api_key=sk-..." is not.
SECRET_PATTERN = re.compile(
    r"""(?ix)
    (?: api[_-]?key | secret | password | passwd | token | credential
      | bearer | authorization | private[_-]?key )
    \s*[:=]\s*\S+
    | -----BEGIN[ A-Z]*PRIVATE KEY-----
    | \b(?:sk|pk|ghp|gho|xoxb|AKIA|AIza)[-_A-Za-z0-9]{12,}\b
    | \b[A-Za-z0-9+/_]{40,}\b
    | \b[0-9a-f]{32,}\b
    """)


def _secret_formed(text):
    return bool(SECRET_PATTERN.search(text or ""))

ENTRY_DELIMITER = "\n§\n"  # Hermes builtin memory (MEMORY.md / USER.md) entry separator

# ---------------------------------------------------------------------------
# embeddings — hashed n-gram tier (stdlib) with optional local ONNX MiniLM
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the and or but if then else for of to in on at by with from as is are was were be "
    "been being am it its this that these those i you he she we they me my your our their "
    "his her them us do does did done doing have has had having will would shall should can "
    "could may might must nor so too very just about into over under again than also "
    "s t don now here there when where why how all any both each few more most other some "
    "such only own same user users".split()
)

_STEM_RULES = (("'s", ""), ("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""), ("ly", ""))

_DIM = 2048  # enough slots that small feature sets rarely collide-and-cancel
_UNIGRAM_WEIGHT = 1.0
_BIGRAM_WEIGHT = 0.6


def _stem(token: str) -> str:
    if len(token) <= 3:
        return token
    for suffix, repl in _STEM_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: len(token) - len(suffix)] + repl
            if len(stem) >= 3 and stem[-1] == stem[-2]:  # running -> runn -> run
                stem = stem[:-1]
            return stem
    return token


def _tokens(text: str) -> List[str]:
    raw = re.findall(r"[a-z0-9']+", (text or "").lower())
    out = []
    for token in raw:
        stem = _stem(token)
        if len(stem) > 1 and stem not in _STOPWORDS:
            out.append(stem)
    return out


def _iso_date(epoch):
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""


def _features(text: str) -> List[Tuple[str, float]]:
    toks = _tokens(text)
    feats = [(t, _UNIGRAM_WEIGHT) for t in toks]
    feats.extend((f"{a} {b}", _BIGRAM_WEIGHT) for a, b in zip(toks, toks[1:]))
    return feats


def hash_embedding(text: str, dim: int = _DIM) -> bytes:
    """Deterministic hashed unigram+bigram vector, L2-normalized, as float32 bytes."""
    vec = [0.0] * dim
    for feat, weight in _features(text):
        digest = hashlib.md5(feat.encode("utf-8")).digest()
        slot = int.from_bytes(digest[:8], "big") % dim
        sign = 1.0 if digest[8] & 1 else -1.0
        vec[slot] += sign * weight
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return struct.pack(f"<{dim}f", *(v / norm for v in vec))


def _cosine_bytes(a: bytes, b: bytes) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    count = len(a) // 4
    va = struct.unpack(f"<{count}f", a)
    vb = struct.unpack(f"<{count}f", b)
    return sum(x * y for x, y in zip(va, vb))


def _cosine_batch(query_vec: bytes, vectors: List[bytes]) -> List[float]:
    """Cosine of the query against many vectors — pure Python, deliberately.

    No numpy here: importing it (any copy) before the vendored onnxruntime
    initializes breaks onnxruntime's capi with an ABI mismatch and silently
    downgrades recall to the hash tier. Pure-Python cosine over 2k-dim vectors
    costs ~0.1 ms per fact — irrelevant next to everything else recall does.
    """
    return [_cosine_bytes(query_vec, v) for v in vectors]


class OnnxEmbedder:
    """Optional local MiniLM (all-MiniLM-L6-v2, int8 ONNX) embedder.

    Requires onnxruntime + model files under ``model_dir`` — everything stays
    on this machine, no network at runtime. Raises RuntimeError when unusable
    so callers fall back to hash embeddings.
    """

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        model_path = self.model_dir / "model.onnx"
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not model_path.exists() or not tokenizer_path.exists():
            raise RuntimeError(f"ONNX model files not found under {self.model_dir}")
        try:
            vendored = self.model_dir.parent / "site-packages"
            if vendored.is_dir() and str(vendored) not in sys.path:
                sys.path.insert(0, str(vendored))  # Hermes runs -I: user site is invisible
            import onnxruntime  # type: ignore
            from tokenizers import Tokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional tier
            raise RuntimeError(f"onnxruntime/tokenizers unavailable: {exc}") from exc
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=128)
        self.input_names = {i.name for i in self._session.get_inputs()}

    def embed(self, text: str) -> bytes:
        encoded = self._tokenizer.encode(text or "")
        ids = encoded.ids
        mask = encoded.attention_mask
        feed = {"input_ids": [ids], "attention_mask": [mask]}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = [[0] * len(ids)]
        out = self._session.run(None, feed)[0][0]  # (seq, hidden)
        vec = [0.0] * out.shape[1]
        for row, m in zip(out, mask):
            if m:
                for i, v in enumerate(row):
                    vec[i] += float(v)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return struct.pack(f"<{len(vec)}f", *(v / norm for v in vec))


_ONNX_SESSIONS: Dict[str, "OnnxEmbedder"] = {}


class Embedder:
    """Dispatches to the ONNX tier when configured+usable, else hashed n-grams."""

    # recall relevance floor: the cosine level below which a match is noise.
    # Measured MiniLM distributions OVERLAP: unrelated pairs 0.20-0.32, weak-but-
    # valid paraphrases 0.27-0.37 — so 0.26 sits mid-band, refusing the clear-noise
    # half and handing borderline cases to the agent with a visible low score.
    # Hash tiers have no semantic blur: unrelated ~0.0-0.18.
    NOISE_FLOOR_ONNX = 0.26
    NOISE_FLOOR_HASH = 0.18

    # Query text repeats constantly: the same question asked twice, a prefetch warmed and
    # then used, a confidence probe over a recall. Re-embedding the query is ~20 of the
    # ~22 ms a recall costs on the ONNX tier, so a small exact-match cache removes most of
    # it. Deliberately NOT a fuzzy cache: an approximate key would silently answer a
    # different question with the previous vector.
    DEFAULT_CACHE_SIZE = 128

    def __init__(self, mode: str = "auto", model_dir=None, cache_size=None):
        self.mode = mode
        self._onnx = None
        self._onnx_failed = False
        self.model_dir = model_dir
        self._cache_size = (self.DEFAULT_CACHE_SIZE if cache_size is None
                            else max(0, int(cache_size)))
        self._cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_info(self) -> dict:
        with self._cache_lock:
            return {"size": len(self._cache), "limit": self._cache_size,
                    "hits": self.cache_hits, "misses": self.cache_misses}

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    @property
    def noise_floor(self):
        return self.NOISE_FLOOR_ONNX if self._onnx is not None else self.NOISE_FLOOR_HASH

    def _try_onnx(self):
        if self._onnx is not None or self._onnx_failed:
            return self._onnx
        if self.mode == "hash":
            return None
        if self.model_dir is None:
            self.model_dir = hermes_home() / "plugins-data" / "hwiki-extractor" / "models" / "minilm"
        # one session per model_dir per process: many stores (tests, gateway +
        # CLI sessions) share a single InferenceSession instead of ~15 of them
        # exhausting memory on small machines
        key = str(Path(self.model_dir).resolve())
        shared = _ONNX_SESSIONS.get(key)
        if shared is None:
            try:
                shared = OnnxEmbedder(self.model_dir)
                _ONNX_SESSIONS[key] = shared
            except Exception as exc:
                # visible, not silent: a silent fallback makes recall quietly worse
                import logging
                import traceback
                logging.getLogger(__name__).warning(
                    "onnx embedding tier unavailable (%s); using hashed n-grams\n%s",
                    exc, traceback.format_exc())
                self._onnx_failed = True
                return None
        self._onnx = shared
        return self._onnx

    def embed(self, text: str) -> bytes:
        # invariant: embed NEVER raises — the optional tier can fail in any
        # state (load, session, inference) and the caller always gets a vector
        key = " ".join(str(text or "").split())     # "  Foo bar " hits "Foo bar"
        if self._cache_size:
            with self._cache_lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    self.cache_hits += 1
                    return cached
                self.cache_misses += 1
        vector = self._embed_uncached(key)
        if self._cache_size and vector:
            with self._cache_lock:
                self._cache[key] = vector
                self._cache.move_to_end(key)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return vector

    def _embed_uncached(self, text: str) -> bytes:
        try:
            onnx = self._try_onnx()
            if onnx is not None:
                return onnx.embed(text)
        except Exception:
            self._onnx_failed = True
            self._onnx = None
        return hash_embedding(text)


# ---------------------------------------------------------------------------
# matching helpers
# ---------------------------------------------------------------------------

_NEGATION_TOKENS = frozenset(
    ("not", "never", "dont", "doesnt", "didnt", "isnt", "wasnt", "wont", "cant",
     "stopped", "switched", "longer", "removed", "quit", "instead", "rather", "hate", "hates")
)

_TEMPORAL_TOKENS = frozenset(
    ("today", "yesterday", "tonight", "week", "month", "recently", "recent",
     "latest", "now", "currently", "current", "just")
)

_BROAD_TOKENS = frozenset(
    ("overview", "summarize", "summarise", "summary", "everything", "anything",
     "know", "profile", "remind", "brief", "recap", "preferences", "setup")
)

_TEMPORAL_WINDOW_PATTERNS = (
    (re.compile(r"\byesterday\b", re.I), 1),
    (re.compile(r"\bthis week\b|\bpast week\b|\blast week\b", re.I), 7),
    (re.compile(r"\bthis month\b|\bpast month\b|\blast month\b", re.I), 30),
    (re.compile(r"\bthis quarter\b|\bpast quarter\b", re.I), 90),
    (re.compile(r"\blast (\d+) days?\b", re.I), None),  # N captured at runtime
)


def _negation_signal(text: str) -> bool:
    toks = set(_tokens(text))
    return bool(toks & _NEGATION_TOKENS)


def related_and_conflict(claim, others, embedder,
                         related_threshold=0.45, dup_threshold=0.90):
    """Score ``claim`` against ``(key, text)`` pairs.

    Returns (related entries sorted by score, possible_conflict flag,
    best duplicate key-or-None). Cosine is blended with token-overlap
    jaccard so the lexical hash tier stays useful on shared vocabulary.
    """
    query_vec = embedder.embed(claim)
    query_toks = set(_tokens(claim))
    related: List[Dict[str, Any]] = []
    best_dup_key = None
    best_dup_score = 0.0
    for key, text in others:
        score = _cosine_bytes(query_vec, embedder.embed(text))
        toks = set(_tokens(text))
        overlap = len(query_toks & toks) / max(1, len(query_toks | toks))
        blended = max(score, 0.5 * score + 0.5 * overlap)
        if blended >= dup_threshold and blended > best_dup_score:
            best_dup_key, best_dup_score = key, blended
        if blended >= related_threshold:
            related.append({"key": key, "score": round(blended, 3), "text": text})
    related.sort(key=lambda r: r["score"], reverse=True)
    conflict = False
    confirmed_dup = None
    claim_negated = _negation_signal(claim)
    for rel in related[:3]:
        # negation pairs legitimately score below 0.5 in embedding space —
        # check every related entry, flagging is cheap and user-reviewed
        #
        # Slot-value contradictions ("runs on Fly.io" vs "runs on Hetzner") are
        # NOT detected here, on purpose. A token-divergence heuristic was built and
        # measured (see benchmarks/phase4-floor-shape.md): it flagged "runs on
        # Fly.io" against "runs Postgres", because both are value-shaped tokens and
        # only the predicate says which slot they fill. Telling those apart needs
        # semantics, so the flag stays negation-only and review still shows
        # `related` for a human to judge.
        if _negation_signal(rel["text"]) != claim_negated:
            conflict = True
            break

    # Optional second opinion from the local NLI model. Off by default: loading it
    # costs ~165 MB of RSS on a path that must stay cheap, and the default retrieval
    # behaviour must not move. Two independent opt-ins share one model load:
    #
    #   HWIKI_NLI_CONFLICT=1  annotate candidates with the relation the model reads
    #   HWIKI_NLI_DUP=1       treat a paraphrase as a duplicate (see below)
    #
    # Neither can change `conflict`: measured on a labelled 21-pair set the model
    # finds 7/7 genuine contradictions against the negation rule's 1/7, but also
    # reads 3/14 compatible pairs as contradictory (benchmarks/phase4-nli-verdict.md).
    # So it annotates for the reviewer instead of deciding.
    if related and (_nli_conflict_enabled() or _nli_dup_enabled()):
        # Two ways to score the same model. The worker is a one-shot subprocess:
        # ~1 s of cold start per proposal instead of holding ~315 MB forever, which
        # on a memory-constrained host is the better trade. The resident scorer stays
        # the default so existing behaviour is unchanged.
        use_worker = bool(getattr(verdict_mod, "worker_enabled", lambda: False)())
        scorer = None
        worker_reads = None
        candidates = []
        annotate = _nli_conflict_enabled()
        for rel in related[:3]:
            # Similarity already decided (>= 0.90) or the candidate is too distant
            # to be a paraphrase: asking the model cannot change anything, so do not
            # pay for it.
            if not annotate and (best_dup_key is not None
                                 or rel["score"] < _NLI_DUP_MIN_SCORE):
                continue
            candidates.append(rel)
        if candidates and use_worker:
            worker_reads = verdict_mod.score_pairs_via_worker(
                [(claim, rel["text"]) for rel in candidates])
        elif candidates:
            try:
                scorer = verdict_mod.try_get_scorer()
            except Exception:  # pragma: no cover - defensive
                scorer = None
        if candidates and (worker_reads is not None or scorer is not None):
            annotate = _nli_conflict_enabled()
            # For the worker path, collect the pairs that need a reverse check and
            # score them in ONE more subprocess rather than one per candidate.
            reverse_reads = None
            need_reverse: list = []
            if worker_reads is not None:
                need_reverse = [
                    i for i, (c, e, _n) in enumerate(worker_reads)
                    if c == c and e == e  # not NaN
                    and c < verdict_mod.MIN_CONTRADICTION
                    and e >= verdict_mod.MIN_ENTAILMENT
                ]
                if need_reverse:
                    reverse_reads = verdict_mod.score_pairs_via_worker(
                        [(candidates[i]["text"], claim) for i in need_reverse])
            for index, rel in enumerate(candidates):
                try:
                    if worker_reads is not None:
                        contrad, entail, _neutral = worker_reads[index]
                        if contrad != contrad or entail != entail:  # NaN
                            break
                        reverse = None
                        if (contrad < verdict_mod.MIN_CONTRADICTION
                                and entail >= verdict_mod.MIN_ENTAILMENT
                                and reverse_reads is not None
                                and index in need_reverse):
                            reverse = reverse_reads[need_reverse.index(index)][1]
                        # Same decision function the resident scorer uses, so the
                        # two paths cannot disagree about identical probabilities.
                        relation = verdict_mod.relation_from_probabilities(
                            contrad, entail, reverse)
                        read = {"relation": relation,
                                "contradiction": round(contrad, 3),
                                "entailment": round(entail, 3)}
                        if reverse is not None:
                            read["entailment_reverse"] = round(reverse, 3)
                    else:
                        read = verdict_mod.classify_pair(claim, rel["text"], scorer)
                except Exception:  # pragma: no cover - model hiccup
                    break
                if annotate:
                    rel["relation"] = read["relation"]
                    rel["nli"] = {"contradiction": read["contradiction"],
                                  "entailment": read["entailment"]}
                # A paraphrase the similarity threshold just missed: "terse status
                # updates" vs "short status updates" scores 0.887 blended, under the
                # 0.90 bar, but reads 0.997 entailment. Only taken when no stronger
                # duplicate exists, and never when it contradicts (same rule as above).
                if (best_dup_key is None and confirmed_dup is None
                        and read["relation"] == "restatement"
                        and not conflict):
                    best_dup_key, best_dup_score = rel["key"], rel["score"]
                    confirmed_dup = rel["text"]
                    if not annotate:
                        break
    return related, conflict, best_dup_key


# ---------------------------------------------------------------------------
# vault (Obsidian) helpers — the memory source of truth
# ---------------------------------------------------------------------------

FACT_MARKER = "extractor: hwiki-extractor"
KIND_PRIORITY = {"preference": 0, "correction": 1, "environment": 2,
                 "decision": 3, "milestone": 4, "blocker": 5}

DEFAULT_VAULT = Path.home() / "Documents" / "Obsidian Vault"
DEFAULT_MEMORY_DIR = "Hermes/Memory/Auto"


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def resolve_vault(hermes_home_path=None) -> Tuple[Path, Path]:
    """(vault_root, memory_dir) from <hermes-home>/hwiki.json, with defaults."""
    home = Path(hermes_home_path) if hermes_home_path else hermes_home()
    root, memory_rel = DEFAULT_VAULT, DEFAULT_MEMORY_DIR
    try:
        config = json.loads((home / "hwiki.json").read_text(encoding="utf-8"))
        root = Path(config.get("wiki_path") or root)
        memory_rel = Path(config.get("memory_dir") or memory_rel)
    except Exception:
        pass
    return root, memory_rel


def slug_for_claim(claim: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (claim or "").lower()).strip("-")[:48].strip("-") or "fact"
    digest = hashlib.sha256(claim.encode("utf-8")).hexdigest()[:6]
    return f"fact-{slug}-{digest}"


def _frontmatter_dict(page_text: str) -> Dict[str, str]:
    if not page_text.startswith("---"):
        return {}
    try:
        end = page_text.index("\n---", 3)
    except ValueError:
        return {}
    out: Dict[str, str] = {}
    for line in page_text[4:end].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


def _one_line(value: Any, limit: int = 0) -> str:
    """Single-line, injection-proof text for frontmatter and headings.

    Claims, evidence, peers and session ids come from a model or from the CLI. An embedded
    newline in any of them used to open a second frontmatter block, so a claim beginning
    with ``---`` could forge ``status: accepted`` or point ``peer`` at someone else. Collapse
    whitespace at the boundary instead of trusting the caller.
    """
    text = " ".join(str(value or "").replace("\x00", " ").split())
    return text[:limit] if limit else text


def write_fact_page(vault_root: Path, memory_rel: Path, proposal: Dict[str, Any],
                    supersedes_claim: str = "") -> Path:
    """Write one Obsidian note per accepted fact; returns the page path."""
    memory_dir = vault_root / memory_rel
    memory_dir.mkdir(parents=True, exist_ok=True)
    slug = slug_for_claim(proposal["claim"])
    page = memory_dir / f"{slug}.md"
    accepted_iso = datetime.fromtimestamp(
        proposal.get("accepted_at") or time.time(), tz=timezone.utc).strftime("%Y-%m-%d")
    tags = {"memory/auto", str(proposal.get("kind") or "fact")}
    lines = [
        "---",
        f"title: {_one_line(proposal['claim'], 80)}",
        "spine: memory",
        "type: memory",
        "status: auto",
        FACT_MARKER,
        f"proposal_id: {proposal['id']}",
        f"kind: {_one_line(proposal.get('kind', 'fact')) or 'fact'}",
        f"peer: {_one_line(proposal.get('peer')) or 'user'}",
        f"project: {_one_line(proposal.get('project'))}",
        f"accepted_at: {accepted_iso}",
        f"tags: [{', '.join(sorted(_one_line(t).replace(',', ' ') for t in tags))}]",
    ]
    if supersedes_claim:
        lines.append(f"supersedes: \"[[{slug_for_claim(supersedes_claim)}]]\"")
    for rel in (proposal.get("related") or [])[:3]:
        if isinstance(rel, dict) and rel.get("text"):
            lines.append(f"related: \"[[{slug_for_claim(rel['text'])}|{rel['text'][:60]}]]\"")
    lines += [
        "---",
        "",
        f"# {_one_line(proposal['claim'])}",
        "",
        f"> **Evidence:** {_one_line(proposal.get('evidence', ''))}",
        "",
        f"*Accepted {accepted_iso} · session `{_one_line(proposal.get('session_id', ''))}` · via hwiki-extractor.*",
        "",
    ]
    page.write_text("\n".join(lines), encoding="utf-8")
    return page


_COMPILED_SECTION_RE = re.compile(
    r"<!-- hwiki:compiled:start -->(.*?)<!-- hwiki:compiled:end -->", re.DOTALL)


def _unstor_heading_truncation(page_claim: str, stored_claim: str) -> str:
    """Return ``stored_claim`` when the page copy is only its truncated heading.

    ``compile`` renders a merged claim as a ``###`` heading capped at
    ``compile._MAX_CLAIM`` characters, so a longer claim is stored shortened. A
    later ``sync_from_vault`` reads the heading back and, because the vault is
    authoritative, would write that shortened text over the full claim — losing
    the tail permanently, with nothing recording that anything was cut.

    A heading that is a strict prefix of the stored claim, and is at the cap, is
    that truncation rather than an edit: keep the full claim. Any other
    difference is a genuine hand edit and is honoured. Comparing against the
    stored claim (not a fixed length) keeps this correct whatever the cap is.
    """
    page_claim = (page_claim or "").strip()
    stored_claim = (stored_claim or "").strip()
    if not page_claim or not stored_claim:
        return page_claim
    if len(page_claim) >= len(stored_claim):
        return page_claim  # same length or longer: not a truncation
    if stored_claim.startswith(page_claim):
        return stored_claim
    return page_claim


def _compiled_section_records(text: str) -> List[Dict[str, str]]:
    """Claim and metadata of every ``compile``-merged section on a page.

    ``compile`` writes a refinement as a ``###`` heading inside marker comments,
    with a ``*kind: … · peer: … · accepted: … · proposal: <id>*`` footer. The
    lifecycle paths need that metadata: a merged claim is a first-class accepted
    fact, so rebuild must import it and forget must remove only it.

    Kept local and regex-only (no import of ``compile.py``) so these paths have
    no dependency on the write-side module.
    """
    records: List[Dict[str, str]] = []
    for block in _COMPILED_SECTION_RE.findall(text or ""):
        claim = ""
        for line in block.splitlines():
            if line.startswith("### "):
                claim = line[4:].strip()
                break
        if not claim:
            continue
        record = {"claim": claim, "evidence": "", "kind": "", "peer": "",
                  "accepted": "", "proposal_id": "", "superseded": "",
                  "superseded_by": ""}
        evidence = re.search(r"^> \*\*Evidence:\*\* (.+)$", block, re.M)
        if evidence:
            record["evidence"] = evidence.group(1).strip()
        meta = re.search(r"^\*(.+?)\*$", block, re.M)
        if meta:
            for part in meta.group(1).split(" · "):
                key, _, value = part.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key in ("kind", "peer", "accepted"):
                    record[key] = value
                elif key == "proposal":
                    record["proposal_id"] = value
        tombstone = re.search(r"^\*Superseded by \[\[([^\]|]+)", block, re.M)
        if tombstone:
            record["superseded"] = "1"
            record["superseded_by"] = tombstone.group(1).strip()
        elif re.search(r"^\*Superseded by", block, re.M):
            record["superseded"] = "1"
        records.append(record)
    return records


def _compiled_sections(text: str) -> List[str]:
    """Claim headings inside ``compile``'s delimited sections."""
    return [record["claim"] for record in _compiled_section_records(text)]


def _section_claim_key(claim: str) -> str:
    return " ".join((claim or "").split()).lower()


def _select_compiled_sections(matches, *, proposal_id: str = "",
                              claim: str = "") -> List["re.Match[str]"]:
    """Pick exactly the section(s) identified by an id, or one unique claim.

    An id always wins over a claim: two hand-edited sections may share a claim
    but only one of them is the fact being forgotten/superseded.  Claim-only
    selection is refused when it is ambiguous, so a duplicated heading can
    never make one call remove several independent facts.
    """
    records = []
    for match in matches:
        parsed = _compiled_section_records(
            "<!-- hwiki:compiled:start -->" + match.group(1)
            + "<!-- hwiki:compiled:end -->")
        records.append(parsed[0] if parsed else None)
    if proposal_id:
        id_matches = [m for m, r in zip(matches, records)
                      if r and r["proposal_id"] == proposal_id]
        if id_matches:
            return id_matches
        # Legacy sections predate per-section proposal ids.  Only fall back to
        # a claim when exactly one such section can be meant.
        legacy = [m for m, r in zip(matches, records)
                  if r and not r["proposal_id"] and claim
                  and _section_claim_key(r["claim"]) == _section_claim_key(claim)]
        return legacy if len(legacy) == 1 else []
    if not claim:
        return []
    claim_matches = [m for m, r in zip(matches, records)
                     if r and _section_claim_key(r["claim"]) == _section_claim_key(claim)]
    return claim_matches if len(claim_matches) == 1 else []


def _splice_compiled_sections(text: str, matches) -> str:
    """Delete the selected marker blocks, touching only the seams they leave.

    Blank-line cleanup is local to each seam: content elsewhere on the page is
    returned byte for byte, so hand-written notes are never reflowed.
    """
    segments = []
    last = 0
    for match in matches:
        segments.append(text[last:match.start()])
        last = match.end()
    segments.append(text[last:])
    updated = segments[0]
    for segment in segments[1:]:
        if not updated or not segment:
            updated += segment
            continue
        updated = (updated.rstrip(" \t\n") + "\n\n" + segment.lstrip(" \t\n"))
    return updated.rstrip("\n") + "\n"


def _remove_compiled_section(text: str, *, proposal_id: str = "",
                             claim: str = "") -> Optional[str]:
    """Remove one matching compiled section; returns None when none matched.

    Deleting the whole page instead would silently destroy every other fact
    merged into it, so ``forget`` rewrites just the requested section.
    """
    matches = list(_COMPILED_SECTION_RE.finditer(text))
    selected = _select_compiled_sections(matches, proposal_id=proposal_id,
                                         claim=claim)
    if not selected:
        return None
    updated = _splice_compiled_sections(text, selected)
    header = re.match(r"\A---\r?\n(.*?)\r?\n---[ \t]*\r?\n", updated, re.DOTALL)
    if header and re.search(r"^compiled_sections:", header.group(1), re.M):
        count = len(_COMPILED_SECTION_RE.findall(updated))
        new_header = re.sub(r"^compiled_sections:.*$",
                            f"compiled_sections: {count}",
                            header.group(1), count=1, flags=re.M)
        updated = f"---\n{new_header}\n---\n" + updated[header.end():]
    return updated


def read_fact_pages(vault_root: Path, memory_rel: Path, max_facts: int = 25) -> List[Dict[str, Any]]:
    """Parse extractor-written fact pages (skips superseded), ranked for injection."""
    memory_dir = vault_root / memory_rel
    facts: List[Dict[str, Any]] = []
    if not memory_dir.is_dir():
        return facts
    for page in memory_dir.glob("*.md"):
        try:
            text = page.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        meta = _frontmatter_dict(text)
        if meta.get("extractor") != "hwiki-extractor":
            continue
        # The page's H1 and each merged section carry independent status: one
        # `supersede` of a merged claim must not hide the page's other facts,
        # and superseding the owner must not hide still-live refinements.
        page_superseded = meta.get("status") == "superseded"
        match = re.search(r"^# (.+)$", text, re.M)
        if not page_superseded:
            facts.append({
                "claim": match.group(1).strip() if match else meta.get("title", page.stem),
                "kind": meta.get("kind", "fact"),
                "peer": meta.get("peer", "user"),
                "project": meta.get("project", ""),
                "accepted_at": meta.get("accepted_at", ""),
                "path": str(page),
            })
        # Claims merged into this page by `compile` live as ### sections, not
        # as the H1. Without this they would be accepted facts that recall can
        # never see — the one failure mode compile must not introduce.
        for section in _compiled_section_records(text):
            if section["superseded"]:
                continue
            # Each merged section records its own kind/peer/date; reporting the
            # owner page's values mislabels the fact during recall injection and
            # profile rendering. Fall back to the page only for old sections
            # that predate per-section metadata.
            section_kind = (section["kind"] if section["kind"] in KIND_PRIORITY
                            else meta.get("kind", "fact"))
            section_peer = (section["peer"] if section["peer"] in ("user", "assistant")
                            else meta.get("peer", "user"))
            facts.append({
                "claim": section["claim"],
                "kind": section_kind,
                "peer": section_peer,
                "project": meta.get("project", ""),
                "accepted_at": section["accepted"] or meta.get("accepted_at", ""),
                "path": str(page),
            })
    facts.sort(key=lambda f: (KIND_PRIORITY.get(f["kind"], 9), f["accepted_at"]), reverse=True)
    facts.sort(key=lambda f: KIND_PRIORITY.get(f["kind"], 9))
    return facts[:max_facts]


GUARDRAIL_KINDS = ("blocker", "correction")

def render_profile(facts: List[Dict[str, Any]], max_chars: int = 3500,
                   usage: Optional[Dict[str, int]] = None) -> str:
    if not facts:
        return ""
    header = [
        "# Memory Profile (user-approved facts)",
        "Snapshot from session start — facts accepted since may exist; call the `recall` tool "
        "before relying on assumptions about anything not listed here.",
        "",
    ]
    lines: List[str] = list(header)
    used = len("\n".join(lines))

    def emit(line):
        nonlocal used
        if used + len(line) + 1 > max_chars:
            return False
        lines.append(line)
        used += len(line) + 1
        return True

    def fact_line(fact):
        line = f"- **{fact['claim']}**"
        if fact.get("project"):
            line += f" _({fact['project']})_"
        return line

    def order(facts):
        """kind priority, then usage (facts the agent actually uses first)."""
        counts = usage or {}
        return sorted(facts, key=lambda f: (KIND_PRIORITY.get(f["kind"], 9),
                                            -counts.get(f.get("proposal_id", ""), 0),
                                            f["accepted_at"]))

    # peer sections: user facts first, then assistant self-facts (Honcho-style duet)
    for peer, heading in (("user", "## User"), ("assistant", "## Assistant (self)")):
        peer_facts = [f for f in facts if (f.get("peer") or "user") == peer]
        if not peer_facts:
            continue
        if not emit(heading):
            break
        # guardrails first: blockers and corrections steer behavior hardest
        rails = [f for f in peer_facts if f.get("kind") in GUARDRAIL_KINDS]
        rest = [f for f in peer_facts if f.get("kind") not in GUARDRAIL_KINDS]
        if rails and emit("### Guardrails (never do)"):
            for fact in order(rails):
                if not emit(fact_line(fact)):
                    break
            lines.append("")
            used += 1
        projects = sorted({f["project"] for f in rest if f.get("project")})
        rest = order(rest)
        if len(projects) > 1:
            for project in projects:
                if not emit(f"### {project}"):
                    break
                for fact in (f for f in rest if f.get("project") == project):
                    if not emit(fact_line(fact)):
                        break
                lines.append("")
                used += 1
            other = [f for f in rest if not f.get("project")]
            if other and emit("### general"):
                for fact in other:
                    if not emit(fact_line(fact)):
                        break
        else:
            for fact in rest:
                if not emit(fact_line(fact)):
                    break
        lines.append("")
        used += 1
    return "\n".join(lines).rstrip()


def mark_superseded_page(page_path, new_claim: str, *, claim: str = "",
                         proposal_id: str = "") -> bool:
    """Tombstone a superseded fact and link it to its replacement.

    A page can carry several accepted facts: its H1 plus every ``compile``-merged
    section. When the superseded row is one of those sections, only that section
    is marked; tombstoning the whole page would hide the owner and every other
    refinement from recall and from rebuild.
    """
    page = Path(page_path) if page_path else None
    if page is None or not page.exists():
        return False
    try:
        text = page.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    slug = slug_for_claim(new_claim)
    matches = list(_COMPILED_SECTION_RE.finditer(text))
    selected = _select_compiled_sections(matches, proposal_id=proposal_id,
                                         claim=claim)
    if selected:
        note = f"*Superseded by [[{slug}|{_one_line(new_claim, 80)}]].*"
        pieces: List[str] = []
        last = 0
        changed = False
        for match in selected:
            block = match.group(0)
            if "*Superseded by" not in block:
                block = block.replace(
                    "\n<!-- hwiki:compiled:end -->",
                    f"\n{note}\n\n<!-- hwiki:compiled:end -->")
                changed = True
            pieces.append(text[last:match.start()])
            pieces.append(block)
            last = match.end()
        pieces.append(text[last:])
        updated = "".join(pieces)
        written = _atomic_write(page, updated) if changed else True
        return bool(written and (changed or updated == text) and True)
    # The row did not match a merged section, so only the page's own claim may
    # justify tombstoning the file. A caller that passes no identity predates
    # per-fact provenance and meant "tombstone this page"; honour that only for
    # a standalone page, since a shared page would lose its merged facts.
    if not (claim or proposal_id):
        if matches:
            return False
        owner_match = True
    else:
        meta = _frontmatter_dict(text)
        heading = re.search(r"^# (.+)$", text, re.M)
        owner_claim = (heading.group(1).strip() if heading
                       else meta.get("title", page.stem))
        owner_match = ((proposal_id and meta.get("proposal_id") == proposal_id)
                       or (claim and _section_claim_key(owner_claim)
                           == _section_claim_key(claim)))
    if not owner_match:
        return False
    if "status: auto" not in text or "status: superseded" in text:
        return False
    text = text.replace("status: auto", "status: superseded", 1)
    text = text.replace("status: superseded",
                        f"status: superseded\nsuperseded_by: \"[[{slug}]]\"", 1)
    text = text.rstrip("\n") + f"\n\n*Superseded by [[{slug}|{new_claim[:80]}]].*\n"
    return bool(_atomic_write(page, text))


def _atomic_write(page: Path, text: str) -> bool:
    """Replace ``page`` atomically; a failed write leaves the original intact."""
    tmp = page.with_name(page.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(page)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def load_builtin_entries(memories_dir=None) -> List[Dict[str, str]]:
    """Entries from Hermes builtin memory (MEMORY.md/USER.md), split on the § delimiter."""
    directory = Path(memories_dir) if memories_dir else hermes_home() / "memories"
    entries: List[Dict[str, str]] = []
    for target in ("MEMORY.md", "USER.md"):
        path = directory / target
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for chunk in raw.split(ENTRY_DELIMITER):
            text = " ".join(chunk.split())
            if text:
                entries.append({"text": text,
                                "target": "memory" if target == "MEMORY.md" else "user"})
    return entries


AUDIT_PAGE = "audit-log"

CANVAS_MAX_NODES = 100
_CANVAS_W, _CANVAS_H, _CANVAS_GAP = 260, 72, 40


def write_fact_canvas(store, vault_root: Path, memory_rel: Path) -> Optional[Path]:
    """Regenerate the Obsidian Canvas of the fact graph (nodes = facts,
    edges = related / superseded-by). Bounded at CANVAS_MAX_NODES; newest first."""
    with _db_connection(store.database) as conn:
        rows = conn.execute(
            "SELECT id, claim, kind, peer, project, status, related, supersedes, superseded_by, accepted_at "
            "FROM proposals WHERE status IN ('accepted','superseded') "
            "ORDER BY COALESCE(accepted_at, created_at) DESC LIMIT ?", (CANVAS_MAX_NODES,)).fetchall()
    if not rows:
        return None
    by_id = {r["id"]: r for r in rows}
    columns: Dict[str, int] = {}
    nodes, edges = [], []
    positions = {}
    for order, r in enumerate(reversed(rows)):  # oldest bottom-left, newest on top
        kind = r["kind"] if r["kind"] in KIND_PRIORITY else "other"
        col = columns.setdefault(kind, len(columns))
        row_idx = sum(1 for p in positions.values() if p[0] == col)
        x = col * (_CANVAS_W + _CANVAS_GAP)
        y = row_idx * (_CANVAS_H + _CANVAS_GAP)
        positions[r["id"]] = (col, row_idx)
        label = r["claim"][:180]
        if r["status"] == "superseded":
            label = "⃠ " + label + "\n(superseded)"
        else:
            label = label + f"\n({r['kind']}" + (f", {r['project']})" if r["project"] else ")")
        color = "2" if r["kind"] in GUARDRAIL_KINDS and r["status"] == "accepted" else (
            "5" if r["status"] == "superseded" else ("4" if (r["peer"] or "user") == "assistant" else None))
        node = {"id": r["id"], "type": "text", "text": label,
                "x": x, "y": y, "width": _CANVAS_W, "height": _CANVAS_H}
        if color:
            node["color"] = color
        nodes.append(node)
    def add_edge(src, dst, label):
        if src in positions and dst in positions and src != dst:
            edges.append({"id": f"e{len(edges)}", "fromNode": src, "toNode": dst, "label": label})
    for r in rows:
        try:
            rel = json.loads(r["related"] or "[]")
        except json.JSONDecodeError:
            rel = []
        for entry in rel[:3]:
            if isinstance(entry, dict) and entry.get("key") in by_id:
                add_edge(r["id"], entry["key"], "related")
        if r["supersedes"] and r["supersedes"] in by_id:
            add_edge(r["id"], r["supersedes"], "superseded")
        if r["superseded_by"] and r["superseded_by"] in by_id:
            add_edge(r["superseded_by"], r["id"], "superseded")
    memory_dir = Path(vault_root) / memory_rel
    memory_dir.mkdir(parents=True, exist_ok=True)
    page = memory_dir / "memory-graph.canvas"
    page.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=1), encoding="utf-8")
    return page


AUDIT_MAX_LINES = 300
AUDIT_KEEP_LINES = 250


def append_audit_entry(vault_root: Path, memory_rel: Path, text: str,
                       today: Optional[str] = None) -> Optional[Path]:
    """Append a timestamped line to the audit-log page (no fact marker, so it
    never enters profiles/recall as a fact; hwiki still indexes it for search).
    Bounded: oldest lines are trimmed past AUDIT_MAX_LINES."""
    if not text:
        return None
    memory_dir = Path(vault_root) / memory_rel
    memory_dir.mkdir(parents=True, exist_ok=True)
    page = memory_dir / f"{AUDIT_PAGE}.md"
    stamp = today or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    entry = f"- **{stamp}** — {text}"
    if not page.exists():
        page.write_text(
            "---\n"
            f"title: Memory Audit Log\n"
            "spine: memory\n"
            "type: log\n"
            "status: auto\n"
            "---\n\n"
            "# Memory Audit Log\n\n"
            "Every accept, reject, forget, supersede and session tidy-up, newest last.\n\n"
            "## Log\n\n" + entry + "\n", encoding="utf-8")
        return page
    body = page.read_text(encoding="utf-8", errors="replace").rstrip("\n") + "\n" + entry + "\n"
    lines = body.splitlines()
    if len(lines) > AUDIT_MAX_LINES:  # keep the header + the most recent entries
        head = next(i for i, ln in enumerate(lines) if ln.startswith("## Log")) + 1
        lines = lines[:head] + ["_(older entries trimmed)_"] + lines[-AUDIT_KEEP_LINES:]
    page.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return page


def write_session_diary(vault_root: Path, memory_rel: Path, session_id: str,
                        summary: str, facts: Sequence[str], topics: Sequence[str]) -> Optional[Path]:
    """Short structured session note (never raw conversation) into the memory zone."""
    if not summary and not facts and not topics:
        return None
    memory_dir = vault_root / memory_rel
    memory_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    sid = re.sub(r"[^a-zA-Z0-9]", "", str(session_id))[-6:] or "nosid"
    page = memory_dir / f"session-{today}-{sid}.md"
    lines = [
        "---",
        f"title: Session {today} ({sid})",
        "spine: memory",
        "type: memory",
        "status: auto",
        FACT_MARKER,
        f"session_id: {_one_line(session_id)}",
        "tags: [memory/auto, session-diary]",
        "---",
        "",
        f"# Session {today} ({sid})",
        "",
    ]
    if summary:
        lines += [summary, ""]
    if topics:
        lines += ["**Topics:** " + ", ".join(_one_line(t) for t in list(topics)[:8]), ""]
    if facts:
        lines += ["**Facts captured:**"]
        lines += [f"- {_one_line(f)}" for f in list(facts)[:10]]
        lines.append("")
    page.write_text("\n".join(lines), encoding="utf-8")
    return page


# ---------------------------------------------------------------------------
# proposal store
# ---------------------------------------------------------------------------


class ProposalRejected(ValueError):
    """Raised when a proposal contains secret-shaped content."""


class LifecycleLockError(OSError):
    """The lifecycle lock could not be acquired.

    An ``OSError`` subclass so existing ``except OSError`` callers keep working,
    and distinct so the tool layer can turn it into a structured refusal instead
    of letting a bare traceback escape from ``forget``/``supersede``/``compile``.
    """


@contextmanager
def _db_connection(database):
    conn = sqlite3.connect(database, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        # WAL: gateway and CLI sessions can hit the store concurrently without
        # writer/reader lock contention; busy_timeout covers the residual case
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        with conn:
            yield conn
    finally:
        conn.close()


class ProposalStore:
    @contextmanager
    def lifecycle_lock(self):
        """Serialize vault read-modify-write sequences across writers.

        Forget, supersede and compile all read a page, decide, then replace it.
        Without one lock spanning that sequence two writers can each write a
        whole-page snapshot and resurrect the other's removed section.  The
        lock lives beside the SQLite database, which is the shared authority
        for a vault, and is reentrant within a thread so nested operations
        (tool -> store.supersede) cannot self-deadlock.
        """
        lock_path = str(self.database) + ".lifecycle.lock"
        state = getattr(_LIFECYCLE_LOCK_STATE, "entry", None)
        if state and state[0] == lock_path:
            state[1] += 1
            try:
                yield
            finally:
                state[1] -= 1
            return
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError as exc:
            # A bare traceback here would escape forget/supersede/compile
            # unstructured, while every other failure in the lifecycle is a
            # {"ok": false, "error": ...}. Fail as a typed refusal instead.
            raise LifecycleLockError(
                f"could not open the lifecycle lock at {lock_path}: {exc}") from exc
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            _LIFECYCLE_LOCK_STATE.entry = [lock_path, 1]
            try:
                yield
            finally:
                _LIFECYCLE_LOCK_STATE.entry = None
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    _NEW_COLUMNS = {
        "embedding": "BLOB",
        "accepted_at": "REAL",
        "superseded_by": "TEXT",
        "supersedes": "TEXT",
        "related": "TEXT",
        "possible_conflict": "INTEGER DEFAULT 0",
        "duplicate_of_builtin": "TEXT",
        "promoted": "INTEGER DEFAULT 0",
        "page_path": "TEXT",
        "peer": "TEXT NOT NULL DEFAULT 'user'",
        "recall_count": "INTEGER DEFAULT 0",
    }

    def __init__(self, database, embedder: Optional[Embedder] = None, *,
                 answerability: Optional[bool] = None):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Embedder()
        # Opt-in, default OFF. Env var wins so the behavior suite can toggle the
        # capability without editing a live config file.
        if answerability is None:
            env = os.environ.get("HWIKI_ANSWERABILITY", "").strip().lower()
            if env in ("1", "true", "yes", "on"):
                answerability = True
            elif env in ("0", "false", "no", "off"):
                answerability = False
        self.answerability = bool(answerability)
        with _db_connection(self.database) as conn:
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS proposals (
                    id TEXT PRIMARY KEY,
                    dedup_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    embedding BLOB,
                    accepted_at REAL,
                    superseded_by TEXT,
                    supersedes TEXT,
                    related TEXT,
                    possible_conflict INTEGER DEFAULT 0,
                    duplicate_of_builtin TEXT,
                    promoted INTEGER DEFAULT 0,
                    page_path TEXT,
                    peer TEXT NOT NULL DEFAULT 'user',
                    recall_count INTEGER DEFAULT 0
                )''')
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_proposals_dedup_key_status ON proposals(dedup_key, status)'
            )
            conn.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(proposals)")}
            for column, decl in self._NEW_COLUMNS.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE proposals ADD COLUMN {column} {decl}")

    @staticmethod
    def _source_hash(user_text):
        return hashlib.sha256(user_text.encode()).hexdigest()

    @staticmethod
    def _dedup_key(peer, kind, project, claim):
        return hashlib.sha256(
            f"{peer}|{kind}|{project}|{claim.strip().lower()}".encode()).hexdigest()

    # -- meta (cross-session state that survives restarts) ------------------

    def question_buffer(self):
        """The process-wide buffer of unanswered questions (see maintain.py)."""
        return question_buffer()

    def get_meta(self, key, default=""):
        with _db_connection(self.database) as conn:
            row = conn.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key, value):
        with _db_connection(self.database) as conn:
            conn.execute(
                'INSERT INTO meta (key, value) VALUES (?,?) '
                'ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key, value))

    def record_feedback(self, kind, accepted):
        """Learn from the user's accept/reject decisions, per fact kind."""
        data = json.loads(self.get_meta('feedback') or '{}')
        entry = data.setdefault(str(kind), {"accepted": 0, "rejected": 0})
        entry["accepted" if accepted else "rejected"] += 1
        self.set_meta('feedback', json.dumps(data))
        return data[kind]

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        data = dict(row)
        related = data.get("related")
        data["related"] = json.loads(related) if related else []
        data["related"] = [r for r in data["related"] if isinstance(r, dict)]
        return data

    def _load_proposal(self, conn, proposal_id):
        cur = conn.execute('SELECT * FROM proposals WHERE id=?', (proposal_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError('Proposal not found')
        return self._row_to_dict(row)

    def propose(self, payload, *, user_text, session_id, project='',
                builtin_entries: Optional[List[Dict[str, str]]] = None, peer='user'):
        if _secret_formed(payload.get('claim', '')) or _secret_formed(payload.get('evidence', '')):
            raise ProposalRejected('secret-shaped proposal rejected')
        peer = str(payload.get('peer') or peer or 'user').strip().lower()
        if peer not in ('user', 'assistant'):
            peer = 'user'
        dedup_key = self._dedup_key(peer, payload['kind'], project, payload['claim'])
        proposal_id = uuid.uuid4().hex
        created_at = time.time()
        claim = payload['claim']

        related: List[Dict[str, Any]] = []
        conflict = 0
        duplicate_of_builtin = ""
        try:
            with _db_connection(self.database) as conn:
                rows = conn.execute(
                    "SELECT id, claim FROM proposals WHERE status='accepted' "
                    "AND (superseded_by IS NULL OR superseded_by='') AND peer=?",
                    (peer,)).fetchall()
                others = [(row["id"], row["claim"]) for row in rows]
            related, conflict_flag, dup_key = related_and_conflict(claim, others, self.embedder)
            conflict = 1 if conflict_flag else 0
            related = related[:5]
            # a near-duplicate is only a duplicate when it agrees: a negated
            # restatement ("does not prefer X" vs "prefers X") scores ~0.96 in
            # embedding space but is a contradiction, not a copy — let it
            # through so it becomes a conflict-flagged proposal
            if dup_key is not None and not conflict_flag:
                with _db_connection(self.database) as conn:
                    return self._load_proposal(conn, dup_key)
        except Exception:
            related, conflict = [], 0
        if builtin_entries and peer == 'user':
            try:
                # score against the entry's assertion (leading chars), not the
                # whole entry — real entries carry quotes/history that dilute
                # the embedding and hide genuine duplicates (measured: a real
                # USER.md pair scored 0.44 full-text vs 0.51+ on its assertion)
                # score against the assertion (leading chars); the full text is
                # returned as the flag so the user sees the whole entry
                _rel, _conf, builtin_dup = related_and_conflict(
                    claim, [(entry["text"], entry["text"][:120]) for entry in builtin_entries],
                    self.embedder, related_threshold=1.1, dup_threshold=0.48)
                if builtin_dup is not None:
                    duplicate_of_builtin = str(builtin_dup)[:240]
            except Exception:
                import os as _os
                if _os.environ.get("HWIKI_DEBUG"):
                    import traceback, sys as _s
                    traceback.print_exc(file=_s.stderr)
                duplicate_of_builtin = ""

        with _db_connection(self.database) as conn:
            cur = conn.execute(
                'SELECT id FROM proposals WHERE dedup_key=? AND status=?',
                (dedup_key, 'proposed'),
            )
            existing = cur.fetchone()
            if existing is not None:
                return self._load_proposal(conn, existing['id'])
            conn.execute(
                '''INSERT INTO proposals
                   (id, dedup_key, status, kind, claim, evidence, session_id, source_hash, project,
                    created_at, embedding, related, possible_conflict, duplicate_of_builtin, peer)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (
                    proposal_id, dedup_key, 'proposed', payload['kind'], claim, payload['evidence'],
                    session_id, self._source_hash(user_text), project, created_at,
                    self.embedder.embed(claim),
                    json.dumps(related), conflict, duplicate_of_builtin, peer,
                ),
            )
            return self._load_proposal(conn, proposal_id)

    def get(self, proposal_id):
        with _db_connection(self.database) as conn:
            return self._load_proposal(conn, proposal_id)

    def accept(self, proposal_id):
        with _db_connection(self.database) as conn:
            cur = conn.execute(
                'UPDATE proposals SET status=?, accepted_at=? WHERE id=? AND status=?',
                ('accepted', time.time(), proposal_id, 'proposed'),
            )
            return cur.rowcount > 0

    def reject(self, proposal_id):
        with _db_connection(self.database) as conn:
            cur = conn.execute(
                'UPDATE proposals SET status=? WHERE id=? AND status=?',
                ('rejected', proposal_id, 'proposed'),
            )
            return cur.rowcount > 0

    def supersede(self, new_id, old_id):
        """Mark an accepted fact superseded by a newer accepted one.

        A proposed (unreviewed) replacement is not authority: allowing it to
        supersede would hide a review-gated fact behind something the user has
        not accepted. Returns (ok, old_row).
        """
        with self.lifecycle_lock():
            with _db_connection(self.database) as conn:
                new = conn.execute('SELECT * FROM proposals WHERE id=?', (new_id,)).fetchone()
                old = conn.execute('SELECT * FROM proposals WHERE id=?', (old_id,)).fetchone()
                if (new is None or old is None
                        or new["status"] != "accepted"
                        or old["status"] != "accepted"):
                    return False, None
                conn.execute(
                    'UPDATE proposals SET status=?, superseded_by=? WHERE id=?',
                    ('superseded', new_id, old_id),
                )
                # record lineage on the new fact so recall can show what replaced what
                conn.execute('UPDATE proposals SET supersedes=? WHERE id=?', (old_id, new_id))
                updated = conn.execute('SELECT * FROM proposals WHERE id=?', (old_id,)).fetchone()
                return True, self._row_to_dict(updated)

    def set_page_path(self, proposal_id, page_path):
        with _db_connection(self.database) as conn:
            conn.execute('UPDATE proposals SET page_path=?, promoted=1 WHERE id=?',
                         (str(page_path), proposal_id))

    def sync_from_vault(self):
        """Hand-edit support — the vault page is the source of truth.

        Claim/evidence edited in Obsidian → DB updated + re-embedded.
        Fact page deleted in Obsidian → fact revoked (excluded from recall).
        Secret-shaped edits are skipped for safety. Returns a summary dict.
        """
        updated = revoked = skipped = 0
        with _db_connection(self.database) as conn:
            rows = conn.execute(
                "SELECT id, claim, evidence, peer, page_path FROM proposals "
                "WHERE status='accepted' AND page_path IS NOT NULL").fetchall()
        # Several accepted facts can live on one shared page (page H1 plus
        # compile's merged sections), so parse each page once.
        parsed: Dict[str, tuple] = {}
        for row in rows:
            page = Path(row["page_path"]) if row["page_path"] else None
            if page is None or not page.exists():
                with _db_connection(self.database) as conn:
                    conn.execute(
                        "UPDATE proposals SET status='deleted_by_edit' WHERE id=? AND status='accepted'",
                        (row["id"],))
                revoked += 1
                continue
            key = str(page)
            if key not in parsed:
                try:
                    text = page.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                meta = _frontmatter_dict(text)
                match = re.search(r"^# (.+)$", text, re.M)
                ev_match = re.search(r"^> \*\*Evidence:\*\* (.+)$", text, re.M)
                parsed[key] = (
                    meta,
                    match.group(1).strip() if match else "",
                    ev_match.group(1).strip() if ev_match else "",
                    _compiled_section_records(text),
                )
            meta, h1, page_evidence, sections = parsed[key]
            # A merged fact belongs to a ``###`` section, not to the page H1.
            # Reading the H1 for every row used to rewrite merged claims as the
            # owner's claim (and then the ID-keyed rebuild could not repair it).
            owner_id = meta.get("proposal_id", "")
            is_owner = (owner_id == row["id"]) if owner_id else not sections
            if is_owner:
                if meta.get("status") == "superseded":
                    # A human can tombstone the owner page by hand. The vault
                    # is authoritative, so mirror that status instead of
                    # leaving the DB fact accepted (which recall would still
                    # serve and a later rebuild would "resurrect").
                    with _db_connection(self.database) as conn:
                        cur = conn.execute(
                            "UPDATE proposals SET status='superseded' "
                            "WHERE id=? AND status='accepted'", (row["id"],))
                    if cur.rowcount:
                        updated += 1
                    continue
                claim = h1 or row["claim"]
                evidence = page_evidence or row["evidence"]
                peer = meta.get("peer", "user")
            else:
                section = next(
                    (s for s in sections
                     if s["proposal_id"] and s["proposal_id"] == row["id"]),
                    None)
                if section is None:
                    section = next(
                        (s for s in sections
                         if s["claim"].strip().lower() == row["claim"].strip().lower()),
                        None)
                if section is None:
                    # The page is still owned by another (live) fact, but this
                    # row's own section was deleted from it: revoke only this
                    # fact instead of rewriting it as the owner's claim.
                    with _db_connection(self.database) as conn:
                        conn.execute(
                            "UPDATE proposals SET status='deleted_by_edit' WHERE id=? AND status='accepted'",
                            (row["id"],))
                    revoked += 1
                    continue
                if section["superseded"]:
                    # Tombstone written by ``supersede``; trust the vault. Keep
                    # any lineage already recorded in the DB.
                    with _db_connection(self.database) as conn:
                        conn.execute(
                            "UPDATE proposals SET status='superseded' WHERE id=? AND status='accepted'",
                            (row["id"],))
                    continue
                claim = section["claim"]
                evidence = section["evidence"] or row["evidence"]
                peer = section["peer"] or meta.get("peer", "user")
            if peer not in ("user", "assistant"):
                peer = "user"
            claim = _unstor_heading_truncation(claim, row["claim"])
            if (claim == row["claim"] and evidence == row["evidence"]
                    and peer == (row["peer"] or "user")):
                continue
            if _secret_formed(claim) or _secret_formed(evidence):
                skipped += 1
                continue
            with _db_connection(self.database) as conn:
                conn.execute(
                    "UPDATE proposals SET claim=?, evidence=?, peer=?, embedding=? WHERE id=?",
                    (claim, evidence, peer, self.embedder.embed(claim), row["id"]))
            updated += 1
        return {"updated": updated, "revoked": revoked, "skipped_secrets": skipped}

    def rebuild_from_vault(self, vault_root, memory_rel):
        """Self-healing: rebuild proposal rows from the vault's fact pages.

        The vault is the source of truth, so a lost/corrupt/empty DB is
        repaired by re-importing every extractor-written page (claim, kind,
        peer, project, accepted_at, status). Idempotent by proposal_id
        (hand-made pages without one get a stable derived id). Returns counts.
        """
        memory_dir = Path(vault_root) / memory_rel
        if not memory_dir.is_dir():
            return {"imported": 0, "skipped": 0}
        imported = skipped = 0
        pages = sorted(memory_dir.glob("*.md"))
        slug_to_id = {}
        for page in pages:
            try:
                text = page.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            meta = _frontmatter_dict(text)
            if meta.get("extractor") != "hwiki-extractor":
                skipped += 1
                continue
            match = re.search(r"^# (.+)$", text, re.M)
            claim = match.group(1).strip() if match else meta.get("title", page.stem)
            ev_match = re.search(r"^> \*\*Evidence:\*\* (.+)$", text, re.M)
            evidence = ev_match.group(1).strip() if ev_match else ""
            kind = meta.get("kind", "fact") if meta.get("kind", "fact") in KIND_PRIORITY else "fact"
            peer = meta.get("peer", "user")
            if peer not in ("user", "assistant"):
                peer = "user"
            project = meta.get("project", "")
            status = "superseded" if meta.get("status") == "superseded" else "accepted"
            proposal_id = meta.get("proposal_id") or (
                "vault-" + hashlib.sha256(str(page).encode()).hexdigest()[:16])
            try:
                accepted_at = datetime.strptime(meta.get("accepted_at", ""), "%Y-%m-%d").replace(
                    tzinfo=timezone.utc).timestamp()
            except (ValueError, TypeError):
                accepted_at = None
            # A secret-shaped owner page must not suppress safe compiled
            # sections: those are independent facts and remain importable.
            if _secret_formed(claim) or _secret_formed(evidence):
                skipped += 1
            else:
                slug_to_id[page.stem] = proposal_id
                slug_to_id[slug_for_claim(claim)] = proposal_id
                with _db_connection(self.database) as conn:
                    cur = conn.execute(
                        '''INSERT INTO proposals
                           (id, dedup_key, status, kind, claim, evidence, session_id, source_hash,
                            project, created_at, embedding, accepted_at, promoted, page_path, peer)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(id) DO NOTHING''',
                        (
                            proposal_id, self._dedup_key(peer, kind, project, claim), status, kind,
                            claim, evidence, 'vault-rebuild', '', project,
                            accepted_at or time.time(), self.embedder.embed(claim),
                            accepted_at, 1, str(page), peer,
                        ))
                # Count actual insertions, not attempts. Reporting every page as
                # "imported" on a re-run made an idempotent rebuild look like it had
                # duplicated work (and hid genuine conflicts from callers).
                if cur.rowcount:
                    imported += 1
                else:
                    skipped += 1
            # Claims merged into this page by `compile` are accepted facts too,
            # but they have no page of their own. Without this pass a rebuild
            # from the vault drops them (the DB "loses" facts the vault still
            # holds) and they can never be recalled again.
            for section in _compiled_section_records(text):
                section_claim = section["claim"]
                section_evidence = section["evidence"]
                if _secret_formed(section_claim) or _secret_formed(section_evidence):
                    skipped += 1
                    continue
                section_kind = section["kind"] if section["kind"] in KIND_PRIORITY else kind
                section_peer = section["peer"] if section["peer"] in ("user", "assistant") else peer
                section_id = section["proposal_id"] or (
                    "vault-" + hashlib.sha256(
                        (str(page) + "\x00" + section_claim).encode()).hexdigest()[:16])
                section_accepted = accepted_at
                try:
                    section_accepted = datetime.strptime(
                        section["accepted"], "%Y-%m-%d").replace(
                            tzinfo=timezone.utc).timestamp()
                except (ValueError, TypeError):
                    pass
                # Each section has its own status; a tombstoned owner page no
                # longer hides still-live refinements, and a section tombstone
                # must survive a rebuild instead of being revived as accepted.
                section_status = "superseded" if section["superseded"] else "accepted"
                # Tombstones name the replacement by claim slug. Register the
                # section itself so a compiled replacement resolves to its id,
                # not to the owner page's id (or to nothing at all).
                slug_to_id[slug_for_claim(section_claim)] = section_id
                with _db_connection(self.database) as conn:
                    cur = conn.execute(
                        '''INSERT INTO proposals
                           (id, dedup_key, status, kind, claim, evidence, session_id, source_hash,
                            project, created_at, embedding, accepted_at, promoted, page_path, peer)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(id) DO NOTHING''',
                        (
                            section_id,
                            self._dedup_key(section_peer, section_kind, project, section_claim),
                            section_status, section_kind, section_claim, section_evidence,
                            'vault-rebuild', '', project, section_accepted or time.time(),
                            self.embedder.embed(section_claim), section_accepted, 1,
                            str(page), section_peer,
                        ))
                if cur.rowcount:
                    imported += 1
                else:
                    skipped += 1
        # resolve lineage links between imported pages (second pass)
        for page in pages:
            try:
                text = page.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            meta = _frontmatter_dict(text)
            proposal_id = meta.get("proposal_id") or (
                "vault-" + hashlib.sha256(str(page).encode()).hexdigest()[:16])
            old_slug = str(meta.get("supersedes", "")).strip('"[]|').split("|")[0]
            new_slug = str(meta.get("superseded_by", "")).strip('"[]|').split("|")[0]
            with _db_connection(self.database) as conn:
                if old_slug and old_slug in slug_to_id:
                    conn.execute('UPDATE proposals SET supersedes=? WHERE id=?',
                                 (slug_to_id[old_slug], proposal_id))
                if new_slug and new_slug in slug_to_id:
                    conn.execute(
                        "UPDATE proposals SET status='superseded', superseded_by=? WHERE id=?",
                        (slug_to_id[new_slug], proposal_id))
                # Merged sections carry their own supersede link, so rebuilding
                # restores lineage for them too.
                for section in _compiled_section_records(text):
                    if not section["superseded_by"]:
                        continue
                    section_id = section["proposal_id"] or (
                        "vault-" + hashlib.sha256(
                            (str(page) + "\x00" + section["claim"]).encode()).hexdigest()[:16])
                    conn.execute(
                        "UPDATE proposals SET status='superseded', superseded_by=? WHERE id=?",
                        (slug_to_id.get(section["superseded_by"], ""), section_id))
        return {"imported": imported, "skipped": skipped}

    def forget(self, proposal_id=None, claim=None):
        """Serialized revoke; see _forget_locked for the fail-closed rules."""
        with self.lifecycle_lock():
            return self._forget_locked(proposal_id=proposal_id, claim=claim)

    def _stale_duplicate_pages(self, row):
        """Pages other than the row's page that carry the same proposal id.

        An interrupted compile can repoint the row at the owner page before it
        unlinks the standalone copy.  That leftover page is byte-identical
        authority to a rebuild, so forget must reconcile it (or refuse) before
        it may report success.

        Returns ``None`` when any candidate cannot be read. That is deliberate:
        an unreadable page may be exactly the duplicate that would resurrect the
        fact after a rebuild, so proceeding would be a guess. The trade-off is
        liveness — one unreadable file in the memory directory blocks every forget
        — so the refusal names the file (``_unreadable_duplicate``); fix its
        permissions and the forget proceeds. Without the name, the block is
        invisible and unfixable.
        """
        page_path = row.get("page_path")
        if not page_path:
            return []
        primary = Path(page_path)
        memory_dir = primary.parent
        if not memory_dir.is_dir():
            return []
        stale = []
        for page in sorted(memory_dir.glob("*.md")):
            try:
                if page.resolve() == primary.resolve():
                    continue
                text = page.read_text(encoding="utf-8", errors="replace")
            except OSError:
                self._unreadable_duplicate = str(page)
                return None
            meta = _frontmatter_dict(text)
            if str(meta.get("proposal_id") or "") == row["id"]:
                stale.append((page, text))
        return stale

    def _plan_forget(self, row, page, text: str) -> dict:
        """Decide what a forget may do, without touching anything.

        Pure: reads ``text`` and returns either ``{"error": ...}`` or
        ``{"stripped": <new page text or None>}``. Every refusal is raised here, so
        the caller can be sure nothing has been mutated when it declines.
        """
        if not text:
            return {}  # no page on disk: nothing to plan
        page = Path(page)
        sections = _compiled_section_records(text)
        meta = _frontmatter_dict(text)
        heading = re.search(r"^# (.+)$", text, re.M)
        owner_claim = (heading.group(1).strip() if heading
                       else meta.get("title", page.stem))
        owner_id = str(meta.get("proposal_id") or "")
        is_owner = ((owner_id == row["id"]) if owner_id
                    else (not sections
                          and _section_claim_key(owner_claim)
                          == _section_claim_key(row["claim"])))
        # A page can own several accepted facts: its H1 claim plus every
        # `compile`-merged section. Deleting the file would destroy facts the
        # caller never asked to forget, so a merged claim is removed from the
        # page and an owner page that still carries merged facts is not touched.
        stripped = _remove_compiled_section(text, proposal_id=row["id"],
                                            claim=row["claim"])
        if stripped is not None:
            return {"stripped": stripped}
        if not is_owner:
            # The row points at a shared page that no longer contains its
            # section (stale/renamed/hand-edited). Unlinking here would delete
            # the owner page and every other fact on it.
            return {"error": ("page no longer contains this fact; refusing "
                              "to delete a shared page")}
        if sections:
            return {"error": ("page also owns compiled facts; forget those first "
                              "so one forget cannot delete accepted facts")}
        return {"stripped": None}

    def _remove_stale_duplicates(self, row) -> Optional[str]:
        """Delete safe stale copies; return an error string when unsafe.

        Two passes, deliberately: validate *every* copy first, then unlink. The
        single-pass version unlinked each safe page as it went and only then hit an
        unsafe one, so a refusal could arrive after the last on-disk copy of the
        fact had already been deleted — a forget that reported ``ok: false`` while
        destroying data. A refusal must leave the vault untouched.
        """
        stale = self._stale_duplicate_pages(row)
        if stale is None:
            where = getattr(self, "_unreadable_duplicate", "") or "a duplicate page"
            return (f"could not read {where}; refusing to forget because it may be "
                    "a duplicate that would resurrect this fact")
        doomed = []
        for page, text in stale:
            sections = _compiled_section_records(text)
            heading = re.search(r"^# (.+)$", text, re.M)
            meta = _frontmatter_dict(text)
            owner_claim = (heading.group(1).strip() if heading
                           else meta.get("title", page.stem))
            if (sections
                    or _section_claim_key(owner_claim)
                    != _section_claim_key(row["claim"])):
                return ("a duplicate page for this fact carries other facts; "
                        "refusing to forget")
            doomed.append(page)
        for page in doomed:
            try:
                page.unlink()
            except OSError:
                return "could not remove a duplicate fact page; refusing to forget"
        return None

    def _prune_forgotten_sections(self, page) -> Optional[bool]:
        """Remove sections on ``page`` whose row is already ``deleted_by_edit``.

        A second writer can finish its forget while ours is between reading the
        page and writing it back: our snapshot then reintroduces a section the
        other writer removed. The lock prevents that across processes, but this
        post-write sweep makes the outcome correct even when a writer is
        interleaved inside our write (or when advisory locks are unavailable).
        Returns True when the page is clean, None when it cannot be verified.
        """
        try:
            text = page.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        sections = [s for s in _compiled_section_records(text)
                    if s.get("proposal_id")]
        if not sections:
            return True
        ids = [s["proposal_id"] for s in sections]
        with _db_connection(self.database) as conn:
            rows = {r["id"]: r["status"] for r in conn.execute(
                "SELECT id, status FROM proposals WHERE id IN (%s)"
                % ",".join("?" * len(ids)), ids)}
        removed = False
        for section in sections:
            if rows.get(section["proposal_id"]) != "deleted_by_edit":
                # Superseded sections keep their tombstone as lineage history.
                continue
            stripped = _remove_compiled_section(
                text, proposal_id=section["proposal_id"],
                claim=section["claim"])
            if stripped is not None:
                text = stripped
                removed = True
        if removed and not _atomic_write(page, text):
            return None
        return True

    def _forget_locked(self, proposal_id=None, claim=None):
        """Revoke an accepted fact: tombstone its page (if any) and mark the row
        deleted_by_edit. Requires the exact id OR the exact claim (no fuzzy)."""
        row = None
        if proposal_id:
            try:
                row = self.get(proposal_id)
            except ValueError:
                return {"ok": False, "error": "no fact with that id"}
        elif claim:
            with _db_connection(self.database) as conn:
                found = conn.execute(
                    "SELECT * FROM proposals WHERE status='accepted' AND lower(claim)=lower(?)",
                    (claim.strip(),)).fetchall()
            if len(found) != 1:
                return {"ok": False,
                        "error": f"expected exactly one match, found {len(found)} — pass the proposal_id instead"}
            row = self._row_to_dict(found[0])
        else:
            return {"ok": False, "error": "provide proposal_id or exact claim"}
        if row["status"] != "accepted":
            return {"ok": False, "error": f"fact is not accepted (status={row['status']})"}
        # Decide everything this forget will do BEFORE touching the disk. A refusal
        # must leave the vault byte-identical: the old order ran the duplicate
        # cleanup first, so the "page no longer contains this fact" refusal arrived
        # after the last on-disk copy of the fact had already been unlinked.
        page = Path(row["page_path"]) if row.get("page_path") else None
        page_text = ""
        if page is not None and page.exists():
            try:
                page_text = page.read_text(encoding="utf-8", errors="replace")
            except OSError:
                # Without the bytes we cannot prove the page belongs to this
                # row. Deleting it could destroy unrelated accepted facts, so
                # fail closed and leave both the page and the row untouched.
                return {"ok": False,
                        "error": "could not read the fact page; refusing to delete it"}
        plan = self._plan_forget(row, page, page_text)
        if plan.get("error"):
            return {"ok": False, "error": plan["error"]}
        duplicate_error = self._remove_stale_duplicates(row)
        if duplicate_error:
            return {"ok": False, "error": duplicate_error}
        page_removed = False
        if page_text:
            stripped = plan.get("stripped")
            if stripped is not None:
                if not _atomic_write(page, stripped):
                    return {"ok": False,
                            "error": "could not rewrite the shared fact page"}
                # Verify after writing: an interleaved writer may have written
                # its own page while we were writing, in which case our
                # snapshot has just resurrected a section it forgot.
                if self._prune_forgotten_sections(page) is None:
                    return {"ok": False,
                            "error": ("could not verify the shared fact page was "
                                      "reconciled; retry the forget")}
            else:
                try:
                    page.unlink()
                    page_removed = True
                except OSError:
                    return {"ok": False,
                            "error": "could not delete the fact page"}
        with _db_connection(self.database) as conn:
            conn.execute("UPDATE proposals SET status='deleted_by_edit' WHERE id=?", (row["id"],))
        return {"ok": True, "forgot": row["claim"], "page_removed": page_removed}

    def usage_counts(self):
        """proposal_id -> recall hit count (usage-ranked memory)."""
        with _db_connection(self.database) as conn:
            rows = conn.execute(
                "SELECT id, recall_count FROM proposals WHERE recall_count > 0").fetchall()
        return {r["id"]: r["recall_count"] for r in rows}

    def never_used_older_than(self, days=90, limit=10):
        """Accepted facts with zero recalls and real age — decay-review candidates."""
        cutoff = time.time() - days * 86400
        with _db_connection(self.database) as conn:
            rows = conn.execute(
                "SELECT claim, kind, accepted_at FROM proposals "
                "WHERE status='accepted' AND (superseded_by IS NULL OR superseded_by='') "
                "AND recall_count = 0 AND accepted_at < ? "
                "ORDER BY accepted_at LIMIT ?", (cutoff, limit)).fetchall()
        return [f"{r['claim']} (accepted {_iso_date(r['accepted_at'])})" for r in rows]

    def neighbors(self, proposal_ids, limit=6):
        """One-hop graph context: claims linked from the given facts via their
        stored `related` matches and supersedes lineage (accepted facts only)."""
        wanted = list(proposal_ids)[:3]
        if not wanted:
            return []
        related_ids = set()
        with _db_connection(self.database) as conn:
            for pid in wanted:
                row = conn.execute("SELECT related, supersedes FROM proposals WHERE id=?", (pid,)).fetchone()
                if row is None:
                    continue
                try:
                    rel = json.loads(row["related"] or "[]")
                except json.JSONDecodeError:
                    rel = []
                for entry in rel:
                    if isinstance(entry, dict) and entry.get("key"):
                        related_ids.add(entry["key"])
                if row["supersedes"]:
                    related_ids.add(row["supersedes"])
            out = []
            for rid in list(related_ids)[:limit]:
                r = conn.execute(
                    "SELECT id, claim, kind, status, peer FROM proposals WHERE id=?", (rid,)).fetchone()
                if r is not None and r["status"] in ("accepted", "superseded"):
                    out.append({"id": r["id"], "claim": r["claim"], "kind": r["kind"],
                                "peer": r["peer"] or "user",
                                "relation": "replaced" if r["status"] == "superseded" else "related"})
        return out

    def stats(self):
        """Status snapshot for the memory_status tool."""
        with _db_connection(self.database) as conn:
            counts = {r["status"]: r["n"] for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM proposals GROUP BY status")}
            peers = {r["peer"]: r["n"] for r in conn.execute(
                "SELECT peer, COUNT(*) AS n FROM proposals WHERE status='accepted' GROUP BY peer")}
            last = conn.execute("SELECT MAX(accepted_at) AS t FROM proposals").fetchone()["t"]
        feedback = json.loads(self.get_meta('feedback') or '{}')
        return {"counts": counts, "peers": peers, "last_accepted": _iso_date(last),
                "feedback": feedback,
                "embedding_tier": "onnx-minilm" if self.embedder._onnx is not None else "hash"}

    def unpromoted_accepted(self, limit=50):
        with _db_connection(self.database) as conn:
            rows = conn.execute(
                "SELECT * FROM proposals WHERE status='accepted' AND promoted=0 "
                "ORDER BY accepted_at, id LIMIT ?", (limit,)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_proposals(self, *, status='proposed', limit=20):
        with _db_connection(self.database) as conn:
            rows = conn.execute(
                'SELECT * FROM proposals WHERE status=? ORDER BY created_at,id LIMIT ?',
                (status, limit),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def consolidate(self, session_id=None, proposal_ttl_days=None):
        """Structural pass: drop duplicate pending proposals (exact or near-duplicate),
        and expire pending proposals older than the TTL.

        Exact duplicates share a dedup_key (earliest kept); near-duplicates are
        found by embedding similarity among pending rows. Returns counts dict.
        """
        removed = 0
        expired = 0
        with _db_connection(self.database) as conn:
            if proposal_ttl_days:
                cutoff = time.time() - proposal_ttl_days * 86400
                cur = conn.execute(
                    "UPDATE proposals SET status='expired' "
                    "WHERE status='proposed' AND created_at < ?", (cutoff,))
                expired = cur.rowcount
            accepted = conn.execute(
                "SELECT id, dedup_key, claim FROM proposals WHERE status='accepted' "
                "AND (superseded_by IS NULL OR superseded_by='')").fetchall()
            if session_id:
                rows = conn.execute(
                    "SELECT id, dedup_key, claim FROM proposals WHERE status='proposed' AND session_id=? "
                    "ORDER BY created_at, id", (session_id,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, dedup_key, claim FROM proposals WHERE status='proposed' "
                    "ORDER BY created_at, id").fetchall()
            # accepted facts are the baseline: pending rows duplicating them are noise
            seen_keys = {row["dedup_key"] for row in accepted}
            kept_claims = [(row["id"], row["claim"]) for row in accepted]
            for row in rows:
                drop = row["dedup_key"] in seen_keys
                if not drop and kept_claims:
                    try:
                        _rel, _conf, dup_key = related_and_conflict(
                            row["claim"], kept_claims[-50:], self.embedder,
                            related_threshold=1.1, dup_threshold=0.90)
                        drop = dup_key is not None
                    except Exception:
                        drop = False
                if drop:
                    conn.execute("DELETE FROM proposals WHERE id=?", (row["id"],))
                    removed += 1
                else:
                    seen_keys.add(row["dedup_key"])
                    kept_claims.append((row["id"], row["claim"]))
        return {"duplicates": removed, "expired": expired}

    def recall(self, query, *, limit=8, status_filter="accepted", project=None, peer=None, now=None,
               answerability=None):
        """Vector recall over stored facts (active ones by default).

        Scoring blends the embedding cosine with token-overlap jaccard (two
        independent signals, one meaning-level one lexical), plus a recency
        boost when the query shows time intent ("recently", "this week", ...).
        Relevance uses a relative-margin rule: keep facts within 0.15 of the
        best match, but only when the best clears the floor — a flat, low-score
        field means the honest answer is "nothing relevant", never the
        least-bad guess.
        """
        # Hostile-but-easy arguments are normalised here, not trusted: ``None`` used to
        # raise from inside a regex and negative limits were passed to SQLite, where
        # LIMIT -1 means "no limit" — quietly ignoring the caller's cap.
        query = " ".join(str(query or "").split())
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 8
        query_vec = self.embedder.embed(query)
        query_toks = set(_tokens(query))
        temporal_intent = bool(query_toks & _TEMPORAL_TOKENS)
        broad_intent = bool(query_toks & _BROAD_TOKENS)
        window_days = None
        for pattern, days in _TEMPORAL_WINDOW_PATTERNS:
            match = pattern.search(query)
            if match:
                window_days = days or int(match.group(1))
                break
        now = now if now is not None else time.time()
        sql = ("SELECT id, kind, claim, evidence, project, accepted_at, supersedes, peer, embedding, page_path "
               "FROM proposals WHERE status=? AND (superseded_by IS NULL OR superseded_by='') "
               "AND embedding IS NOT NULL")
        params: list = [status_filter]
        if project:
            sql += " AND project=?"
            params.append(project)
        if peer:
            sql += " AND peer=?"
            params.append(peer)
        with _db_connection(self.database) as conn:
            rows = conn.execute(sql, params).fetchall()
            lineage = {}
            superseded_ids = {row["supersedes"] for row in rows if row["supersedes"]}
            if superseded_ids:
                marks = ",".join("?" for _ in superseded_ids)
                for old in conn.execute(
                        f"SELECT id, claim FROM proposals WHERE id IN ({marks})",
                        tuple(superseded_ids)).fetchall():
                    lineage[old["id"]] = old["claim"]
        scored = []
        outside_window = []
        cosines = _cosine_batch(query_vec, [row["embedding"] or b"" for row in rows])
        for row, cos in zip(rows, cosines):
            overlap = len(query_toks & set(_tokens(row["claim"]))) / max(
                1, len(query_toks | set(_tokens(row["claim"]))))
            score = max(cos, 0.5 * cos + 0.5 * overlap)
            if temporal_intent and row["accepted_at"]:
                days = max(0.0, (now - row["accepted_at"]) / 86400.0)
                score += 0.10 * math.exp(-days / 14.0)  # fresh facts surface on "recently"
                if window_days is not None and days > window_days:
                    outside_window.append(row["id"])  # demote at ranking, never hide
            if score > 0.0 or broad_intent:
                page_stem = Path(row["page_path"]).stem if row["page_path"] else ""
                scored.append({
                    "id": row["id"], "kind": row["kind"], "claim": row["claim"],
                    "evidence": (row["evidence"] or "")[:140],
                    "project": row["project"] or "",
                    "peer": row["peer"] or "user",
                    "accepted": _iso_date(row["accepted_at"]),
                    "replaced": lineage.get(row["supersedes"], ""),
                    # Citation surface: a wikilink back to the vault page a caller
                    # can cite, plus the accepted date. Only the slug is exposed —
                    # never the absolute filesystem path, which may contain a
                    # home directory. Empty when the fact has no page yet.
                    "page": page_stem,
                    "cite": (f"[[{page_stem}]]" if page_stem else ""),
                    "score": round(score, 3), "source": "fact",
                })
        if not scored:
            _record_miss(query)
            return []
        best = max(r["score"] for r in scored)
        noise_floor = self.embedder.noise_floor
        if broad_intent:
            # synthesis queries want coverage, not just the top cluster: take
            # everything above a low floor, ordered kind-first then by score
            kept = [r for r in scored if r["score"] >= max(0.0, best - 0.45)]
            kept.sort(key=lambda r: (KIND_PRIORITY.get(r["kind"], 9), -r["score"]))
            kept = kept[:limit]
        else:
            # explicit project scope is itself an intent signal: accept weaker matches
            floor = noise_floor * 0.55 if project else noise_floor
            if best < floor:
                _record_miss(query)
                return []
            cutoff = best - 0.15
            kept = [r for r in scored if r["score"] >= cutoff]
            kept.sort(key=lambda r: r["score"], reverse=True)
            kept = kept[:limit]
        # Opt-in answer-type gate (OFF by default so the retrieval path that the
        # Phase-0/1 baselines were measured on is byte-for-byte unchanged).
        # Only ever removes hits, and only on a confident type mismatch.
        if answerability if answerability is not None else self.answerability:
            kept = answerability_mod.filter_hits(query, kept)
        # usage tracking: facts that actually get recalled get ranked higher later
        try:
            with _db_connection(self.database) as conn:
                marks = ",".join("?" for _ in kept)
                conn.execute(f"UPDATE proposals SET recall_count = recall_count + 1 "
                             f"WHERE id IN ({marks})", tuple(r["id"] for r in kept))
        except Exception:
            pass
        return kept

    def recall_confidence(self, query, *, limit=8, **kwargs) -> dict:
        """Advisory confidence in the top hit. Never changes ranking.

        Retrieval answers "is this related?". It does not know whether the fact
        answers the question: two deterministic attempts to derive answerability
        were measured and failed (see benchmarks/phase2-answerability.md). Rather
        than fake a verdict, this returns the raw signals a caller can judge, so an
        agent with more context can decide to abstain.

        Returned keys:
          ``best``      top blended score (0 when nothing was found)
          ``margin``    best minus runner-up; small means an ambiguous field
          ``overlap``   lexical jaccard of the top hit (0 when nothing found)
          ``candidates`` how many facts were within the relevance cutoff
          ``floor``     the noise floor the tier applies
          ``confident`` best >= floor and (no runner-up or margin >= 0.05)
          ``caveat``    fixed string: relatedness is not answerability

        Callers must not treat ``confident`` as "this answers the question".
        """
        hits = self.recall(query, limit=limit, **kwargs)
        floor = float(getattr(self.embedder, "noise_floor", 0.0))
        best = float(hits[0]["score"]) if hits else 0.0
        second = float(hits[1]["score"]) if len(hits) > 1 else 0.0
        query_toks = set(_tokens(query))
        overlap = 0.0
        if hits:
            claim_toks = set(_tokens(str(hits[0]["claim"])))
            overlap = len(query_toks & claim_toks) / max(1, len(query_toks | claim_toks))
        margin = round(best - second, 3)
        return {
            "best": round(best, 3),
            "margin": margin,
            "overlap": round(overlap, 3),
            "candidates": len(hits),
            "floor": round(floor, 3),
            "confident": bool(hits) and best >= floor and (margin >= 0.05 or len(hits) == 1),
            "caveat": "relatedness is not answerability",
        }


# --------------------------------------------------------------------------
# Filing a synthesised answer back into the wiki
# --------------------------------------------------------------------------
#
# One implementation, two callers: the `wiki_memory_file` agent tool and
# `hermes hwiki file`. Keeping the page format here means the CLI cannot drift
# from the tool, and the tests only have to pin one behaviour.

ANSWERS_SUBDIR = "answers"


def file_answer(memory_root: Path, question: str, answer: str,
                tags: Optional[Sequence[str]] = None, *,
                today: Optional[str] = None) -> Dict[str, Any]:
    """Write a synthesised answer into ``<memory_root>/answers/``.

    Returns ``{success, path, created_page, duplicate, note}`` and never raises
    for the expected cases (empty input, secret material, slug collision).
    Idempotent: re-filing the same question returns the existing page rather than
    overwriting it. Two *different* questions that slugify the same way each keep
    their own file via a short digest suffix.
    """
    from .store import scan_for_secrets, slugify

    question = " ".join(str(question or "").split())
    answer = str(answer or "").strip()
    if not question:
        return {"success": False, "error": "question is required"}
    if not answer:
        return {"success": False, "error": "answer is required"}

    found = scan_for_secrets(question + "\n" + answer)
    if found:
        return {
            "success": False,
            "error": (
                "refusing to file an answer containing apparent secret material "
                f"({', '.join(found)}); cite where the secret lives instead"
            ),
        }

    tag_list = [_one_line(t).replace(",", " ") for t in (tags or []) if _one_line(t)]
    answers_dir = Path(memory_root) / ANSWERS_SUBDIR
    slug = slugify(question, fallback="answer")
    target = answers_dir / f"{slug}.md"

    def already_filed(path: Path) -> bool:
        try:
            previous = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        match = re.search(r"^# (.+)$", previous, re.M)
        return bool(match) and match.group(1).strip().casefold() == question.casefold()

    if target.exists():
        if already_filed(target):
            return {"success": True, "path": str(target), "created_page": False,
                    "duplicate": True,
                    "note": "This question is already filed; nothing was overwritten."}
        digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]
        target = answers_dir / f"{slug}-{digest}.md"
        if target.exists():
            return {"success": True, "path": str(target), "created_page": False,
                    "duplicate": True,
                    "note": "This question is already filed; nothing was overwritten."}

    stamp = today or date.today().isoformat()
    title = question[:160]
    tag_line = ""
    if tag_list:
        tag_line = "\n" + " ".join(f"#{slugify(t)}" for t in tag_list) + "\n"
    page_text = (
        "---\n"
        f"title: {title}\n"
        "type: answer\n"
        "status: filed\n"
        "source: hwiki\n"
        f"filed: {stamp}\n"
        f"updated: {stamp}\n"
        "---\n\n"
        f"# {title}\n\n"
        f"> **Filed:** {stamp} · from a conversation answer.\n\n"
        f"{answer}\n"
        f"{tag_line}"
    )

    tmp = target.with_name(target.name + ".tmp")
    try:
        answers_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(page_text, encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        return {"success": False, "error": f"could not file answer: {exc}"}

    return {"success": True, "path": str(target), "created_page": True,
            "duplicate": False,
            "note": "Answer filed — searchable like any other page."}
