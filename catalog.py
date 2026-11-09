"""Auto-maintained navigation artifacts: ``index.md`` and ``log.md``.

The LLM-wiki pattern keeps two special files beside the pages themselves:

``index.md``
    Content-oriented. A catalog of every page — path, title, one-line
    summary, updated date — grouped by folder. Navigable by a human in
    Obsidian and by the agent when it needs orientation rather than recall.

``log.md``
    Chronological, append-only. One ``## [YYYY-MM-DD] kind | subject`` line
    per event, so ``grep "^## \\[" log.md | tail -5`` gives the timeline.

Both are **derived artifacts**. They are regenerated/appended by the provider
and are never a source of truth: if they are deleted, nothing is lost. They
therefore live OUTSIDE the memory zone (``memory_dir``) — that zone is what
``store.py`` owns and what accepted facts are written to, and mixing generated
catalogs into it would pollute fact recall and the review queue.

They are also deliberately kept OUT of the recall ranking (``demote_paths``):
retrieval here is FTS5 + local embeddings, so an index-first navigation step
would be a regression rather than the improvement it is for a RAG-less wiki.

Design rules:

* stdlib only, and no writes ever leave ``root``;
* both files are written atomically (temp + ``os.replace``) so a crash
  mid-write cannot leave a truncated catalog;
* neither file lists itself;
* summary text is derived deterministically from the page — first real
  sentence of the first prose paragraph, frontmatter ``summary``/``description``
  honoured when present, links/code stripped. No LLM call: an index that
  depends on a model call would silently drift every time the call failed.
"""

from __future__ import annotations

import logging
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .store import (WikiStore, _FileLock, _open_anchored, _open_dir_beneath,
                    _stat_beneath, slugify)

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.md"
LOG_FILENAME = "log.md"
#: The tick report the maintainer writes into the memory zone (see ``maintain``).
MAINTENANCE_FILENAME = "maintenance.md"
# Frontmatter marker that says "hwiki owns this file". Both generated artifacts carry it.
# A catalog file WITHOUT it was written by a person, and regenerating it would delete that
# person's work with nothing in the output to say so — so it is left alone instead.
OWNER_MARKER = "source: hwiki-catalog"
OWNER_VALUE = "hwiki-catalog"


_NOT_OURS = object()


def _owned_state(path: Path):
    """``_NOT_OURS`` for a foreign file, else the stat state (None when absent/empty)."""
    info = _lstat(path)
    if info is None:
        return None
    if info.st_size == 0:
        return _stat_state(info)  # empty is adoptable; still pin the state
    try:
        with open(path, "rb") as handle:
            head = handle.read(64 * 1024).decode("utf-8", "replace")
    except OSError:
        return _NOT_OURS
    return _stat_state(info) if _owned_head(head) else _NOT_OURS


def _frontmatter_source(text: str) -> str:
    """The value of the frontmatter ``source:`` field, or ``""``.

    Parsed as a field, not searched as a substring: a hand-written page that merely
    *mentions* ``source: hwiki-catalog`` in prose or in another field is not ours.
    """
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    if end == -1:
        return ""
    block = text[3:end]
    match = re.search(r"^source:[ \t]*([^\n]*)$", block, re.MULTILINE)
    if not match:
        return ""
    return match.group(1).strip().strip("'\"")


def _owned_head(text: str) -> bool:
    """True when the frontmatter names this generator as the source."""
    return _frontmatter_source(text) == OWNER_VALUE


def owns_artifact(path: Path) -> bool:
    """True when ``path`` is absent (safe to create) or was written by hwiki."""
    if path.is_symlink():
        return False  # never read or adopt through a planted link
    try:
        if not path.exists():
            return True
        if path.stat().st_size == 0:
            # A crash or a full disk can leave an empty file behind. There is nothing
            # in it to preserve, and refusing would stop the artifact from ever being
            # written again.
            return True
        with path.open("rb") as handle:
            text = handle.read(_OWNER_SCAN_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return False
    return _owned_head(text)

# Prefix that makes log entries machine-parseable:
#   ## [2026-11-08] ingest | Article Title
LOG_ENTRY_RE = re.compile(r"^## \[(\d{4}-\d{2}-\d{2})\] (\S+) \| (.*)$")

#: Bytes read from the end of the log for a short tail; quadrupled when too small.
_TAIL_WINDOW = 16 * 1024
#: Ownership lives in the frontmatter, so only the head of an artifact is inspected.
#: Reading a grown log whole on every append made each 40-byte entry cost the file.
_OWNER_SCAN_BYTES = 4 * 1024

MAX_SUMMARY_CHARS = 160
# Never scan a whole huge page when hunting for the first sentence.
_SUMMARY_SCAN_CHARS = 4000

_WIKILINK_STRIP = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
_MDLINK_STRIP = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_HTML_TAG = re.compile(r"<[^>]+>")
_FENCE = re.compile(r"^\s*(```|~~~)")
_CALLOUT_QUOTE = re.compile(r"^>\s*\[![^\]]*\][+-]?\s*")
_LEADING_MARKUP = re.compile(r"^(?:[-*+]|\d+\.)\s+|^>\s*")
_LEADING_DATE = re.compile(r"^\*{0,2}\d{4}-\d{2}-\d{2}\*{0,2}\s*[—–-]\s*")
_TRAILING_META = re.compile(r"(?:\s*_\(source:.*?\)_|\s+#[\w/-]+)\s*$")


def _strip_meta_tail(text: str) -> str:
    """Remove repeated trailing metadata (``#tags`` and ``_(source: ...)_``)."""
    previous = None
    while previous != text:
        previous = text
        text = _TRAILING_META.sub("", text).strip()
    return text

# Boilerplate that appears at the top of every generated page. A summary made
# of this text describes the generator, not the page, so it is skipped.
_BOILERPLATE_MARKERS = (
    "auto-maintained by",
    "generated by the",
    "promote durable",
    "conclusions into the curated wiki",
    "this note is auto-maintained",
)

# A memory page's callout header is the generated note under the H1; anything
# beginning with these is part of that header, never a summary.
_HEADER_LEADINS = (
    "auto-maintained by",
    "conclusions into the curated wiki",
    "promote durable",
)


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _collapse(text: str) -> str:
    return " ".join((text or "").split())


def _strip_markup(text: str) -> str:
    text = _WIKILINK_STRIP.sub(lambda m: m.group(2) or m.group(1), text)
    text = _MDLINK_STRIP.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    return _collapse(text)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1].rstrip()
    if " " in cut[max_chars // 2:]:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"


def summarize_page(page, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    """One-line, deterministic summary of a page.

    Preference order: frontmatter ``summary`` → ``description`` → first
    meaningful sentence of the body prose. Never raises.
    """
    try:
        return _summarize_page(page, max_chars)
    except Exception as exc:  # a bad page must never break catalog generation
        logger.debug("hwiki: summary failed for %s: %s",
                     getattr(page, "path", "?"), exc)
        return ""


def _is_boilerplate(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _BOILERPLATE_MARKERS)


def _summarize_page(page, max_chars: int) -> str:
    for key in ("summary", "description"):
        value = page.frontmatter.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate(_strip_markup(value), max_chars)

    body = (page.body or "")[:_SUMMARY_SCAN_CHARS]
    paragraph: List[str] = []
    in_fence = False
    for raw in body.splitlines():
        line = raw.rstrip()
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = line.strip()
        if not stripped:
            # A blank line ends the first prose paragraph — stop looking.
            if paragraph:
                break
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("---") or stripped.startswith("|"):
            continue
        if stripped.startswith("*") and stripped.endswith("*") and ":" in stripped:
            continue  # italic provenance line ("*Accepted ... via hwiki*")
        stripped = _CALLOUT_QUOTE.sub("", stripped)
        stripped = _LEADING_MARKUP.sub("", stripped)
        stripped = _LEADING_DATE.sub("", stripped)
        cleaned = _strip_meta_tail(_strip_markup(stripped))
        if not cleaned:
            continue
        lowered = cleaned.lower()
        # Skip the whole generated callout header: every line of it is a
        # lead-in fragment, so matching only the first one is not enough.
        if any(lowered.startswith(lead) for lead in _HEADER_LEADINS):
            continue
        if _is_boilerplate(cleaned) and not paragraph:
            continue
        paragraph.append(cleaned)

    text = _collapse(" ".join(paragraph))
    if not text:
        return ""
    # Cut at the first sentence end when it is not absurdly early.
    match = re.search(r"(?<=[.!?])\s", text)
    if match and match.start() >= 40:
        text = text[: match.start()]
    return _truncate(text, max_chars)


def _atomic_write(path: Path, text: str, *, root: Optional[Path] = None,
                  expected: object = None) -> bool:
    """Write ``text`` to ``path`` atomically. Returns False when it refuses.

    ``expected`` is the ownership state captured when the file was inspected
    (``None`` for absent/empty, else a ``(st_ino, st_mtime_ns, st_size)`` tuple). Just
    before the atomic replace the file is stat'ed again: if it changed — a person or
    another process wrote it in the meantime — nothing is replaced and ``False`` is
    returned. The temp file, the stat and the replace all go through a directory
    descriptor anchored under ``root``, so a swapped parent directory cannot move the
    write outside the vault.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if root is not None:
        parent_fd = _open_dir_beneath(root, parent)
    else:  # pragma: no cover - callers in this package always pass the vault root
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    tmp_name = f".hwiki-{os.getpid()}-{os.urandom(6).hex()}.tmp"
    try:
        fd = os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                     0o600, dir_fd=parent_fd)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        if root is not None:
            current = _stat_beneath(root, path)
        else:  # pragma: no cover
            current = _lstat(path)
        if _stat_state(current) != expected:
            return False
        os.replace(tmp_name, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        return True
    finally:
        try:
            os.unlink(tmp_name, dir_fd=parent_fd)
        except OSError:
            pass
        os.close(parent_fd)

def _lstat(path: Path):
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _stat_state(info) -> object:
    if info is None:
        return None
    return (info.st_ino, info.st_mtime_ns, info.st_size)


# ---------------------------------------------------------------------------
# index.md
# ---------------------------------------------------------------------------


def _group_label(rel_path: str) -> str:
    parent = Path(rel_path).parent.as_posix()
    return "" if parent in ("", ".") else parent


def _in_memory_zone(rel_path: str, memory_root: str) -> bool:
    if not memory_root:
        return False
    parent = Path(rel_path).parent.as_posix()
    return parent == memory_root or parent.startswith(memory_root + "/")


#: Characters a wikilink cannot carry in a target: ``|`` starts an alias, ``#`` a
#: heading, ``]]`` ends the link, and a newline ends the line.
_LINK_HOSTILE = ("|", "#", "]", "[", "\n", "\r")


def linkable_target(path: str) -> bool:
    """True when ``path`` can be written as a ``[[wikilink]]`` target and read back.

    A path containing one of :data:`_LINK_HOSTILE` cannot: the resolver (and
    Obsidian) would split it, so the catalog lists such pages as plain paths and
    ``lint`` does not ask them to be linked.
    """
    return not any(ch in path for ch in _LINK_HOSTILE)


def is_catalog_artifact(path: str, *, memory_dir: str = "") -> bool:
    """True only for the *generated* catalog files, by exact location.

    ``index.md`` and ``log.md`` live at the wiki root; the maintenance report lives at
    the top of the memory zone. Matching the basename anywhere instead made a curated
    page called ``maintenance.md`` (or a nested ``index.md``) invisible to the catalog
    demand check, so it could never be reported missing from the index.
    """
    rel = (path or "").replace("\\", "/").strip("/")
    if "/" not in rel and rel in (INDEX_FILENAME, LOG_FILENAME):
        return True
    zone = (memory_dir or "").strip("/")
    return bool(zone) and rel.lower() == f"{zone}/{MAINTENANCE_FILENAME}".lower()


def build_index(store: WikiStore, *, memory_dir: str = "",
                title: str = "Wiki Index",
                exclude_names: Sequence[str] = (INDEX_FILENAME, LOG_FILENAME),
                today: Optional[str] = None) -> str:
    """Render the full ``index.md`` text for ``store``."""
    stamp = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    memory_root = (memory_dir or store.memory_dir).strip("/")
    excluded = set(exclude_names)

    groups: Dict[str, List] = {}
    memory_pages: List = []
    # one enumeration for the whole render: resolving each emitted link by asking
    # ``store.resolve_page`` would re-walk the vault per page and turn the catalog
    # into an O(n^2) file-read loop on a large vault
    store_pages = list(store.iter_pages())
    if os.environ.get("HWIKI_DEBUG_CATALOG"):
        print("DBG store_pages:", [p.path for p in store_pages])
    for page in store_pages:
        name = Path(page.path).name
        if is_catalog_artifact(page.path, memory_dir=memory_root):
            continue
        if page.path in excluded or (name in excluded and "/" not in page.path):
            continue
        if _in_memory_zone(page.path, memory_root):
            memory_pages.append(page)
            continue
        groups.setdefault(_group_label(page.path), []).append(page)

    # A bare ``[[Stem]]`` is ambiguous once two folders hold the same filename, so
    # those are folder-qualified. Uniqueness of the stem is not enough by itself,
    # though: the catalog must emit a link that opens *this* page, so every candidate
    # is checked against the store's own resolver. ``Projects/index.md`` collides with
    # the root artifact ``index.md`` even though it is the only page with that stem;
    # a page whose link cannot be made to resolve to it is listed as a plain path.
    all_pages = [p for bucket in groups.values() for p in bucket] + memory_pages
    stems: Dict[str, int] = {}
    for page in all_pages:
        stem = Path(page.path).stem.lower()
        stems[stem] = stems.get(stem, 0) + 1

    # The catalog must emit a link whose meaning does not depend on directory
    # iteration order. A bare stem is used only when the resolver's own tiers name
    # *only* the intended page: an exact path (unique by construction) or a unique
    # case-folded path. Anything below that — filename slug, slugified slug, title —
    # keeps the first match, and "first" depends on glob order, so a bare link is used
    # only when that whole loose set is a singleton. ``a/Foo Bar.md`` and
    # ``b/foo-bar.md`` share a slugified stem, and ``Projects/index.md`` shares a slug
    # with the root artifact, so both are folder-qualified deterministically.
    exact_map: Dict[str, set] = {}
    folded_map: Dict[str, set] = {}
    loose_map: Dict[str, set] = {}

    def _record(mapping: Dict[str, set], key: str, page) -> None:
        if key:
            mapping.setdefault(key, set()).add(page.path)

    for other in store_pages:
        keys = [other.path]
        if other.path.lower().endswith(".md"):
            keys.append(other.path[:-3])
        for key in keys:
            _record(exact_map, key, other)
            _record(folded_map, key.lower(), other)
        _record(loose_map, other.slug, other)
        _record(loose_map, slugify(other.slug), other)
        if other.title:
            _record(loose_map, slugify(other.title), other)

    def _names_only(option: str, page) -> bool:
        option = option.split("|", 1)[0].split("#", 1)[0]
        exact = exact_map.get(option)
        if exact:
            return exact == {page.path}
        folded = folded_map.get(option.lower())
        if folded:
            return folded == {page.path}
        # both keys the slug tier compares against, unioned: a bare ``Foo Bar`` is
        # matched by ``b/foo-bar.md`` through ``slugify(page.slug)`` as well
        loose = (set(loose_map.get(option) or ())
                 | set(loose_map.get(slugify(option)) or ()))
        return loose == {page.path}

    def _link_target(page) -> Optional[str]:
        if not linkable_target(page.path):
            return None  # ``|``, ``#`` or ``]]`` in the path: no wikilink can carry it
        stem = Path(page.path).stem
        qualified = page.path[:-3] if page.path.lower().endswith(".md") else page.path
        options = []
        if stems.get(stem.lower(), 0) < 2:
            options.append(stem)
        if qualified != stem:
            options.append(f"{qualified}|{stem}")
        for option in options:
            if _names_only(option, page):
                return option
        return None

    unlinkable: List = []
    link_targets: Dict[str, str] = {}
    for bucket in list(groups.values()) + [memory_pages]:
        for page in list(bucket):
            target = _link_target(page)
            if target is None:
                bucket.remove(page)
                unlinkable.append(page)
            else:
                link_targets[page.path] = target

    groups = {label: bucket for label, bucket in groups.items() if bucket}

    lines: List[str] = [
        "---",
        f"title: {title}",
        "type: index",
        "status: auto",
        "source: hwiki-catalog",
        f"updated: {stamp}",
        "tags: [index]",
        "---",
        "",
        f"# {title}",
        "",
        "> Generated by the `hwiki` memory provider from the pages on disk.",
        "> Edits are overwritten on the next regeneration; change the pages instead.",
        "",
    ]

    total = sum(len(v) for v in groups.values())

    def _link(page) -> str:
        return f"[[{link_targets[page.path]}]]"
    # Wording is unchanged for ordinary vaults: only a vault that actually has a
    # path-only page gets the extra clause, so existing indexes and snapshots of this
    # line do not churn.
    if unlinkable:
        counted = (f"{total} linkable page(s) across {len(groups)} section(s), "
                   f"{len(memory_pages)} memory note(s), {len(unlinkable)} listed by "
                   f"path only")
    else:
        counted = (f"{total} curated page(s) across {len(groups)} section(s), "
                   f"{len(memory_pages)} memory note(s)")
    lines.append(f"*{counted} — "
                 f"{total + len(memory_pages) + len(unlinkable)} page(s) catalogued.*")
    lines.append("")

    def _emit(label: str, bucket: Iterable) -> None:
        items = sorted(bucket, key=lambda p: (p.title.lower(), p.path))
        if not items:
            return
        if label:
            lines.extend([f"## {label}", ""])
        for page in items:
            summary = summarize_page(page)
            suffix = f" — {summary}" if summary else ""
            if page.updated:
                suffix += f" *({page.updated})*"
            lines.append(f"- {_link(page)}{suffix}")
        lines.append("")

    for label in sorted(groups):
        _emit(label, groups[label])

    if memory_pages:
        lines.extend(["## Memory notes", ""])
        for page in sorted(memory_pages,
                           key=lambda p: (p.updated or "", p.path), reverse=True):
            summary = summarize_page(page, max_chars=110)
            kind = page.frontmatter.get("kind") or "note"
            suffix = f" — {summary}" if summary else ""
            lines.append(f"- {_link(page)} *({kind})*{suffix}")
        lines.append("")

    if unlinkable:
        lines.extend([
            "## Not linkable from an index",
            "",
            "These paths contain `|`, `#` or `]]`, which wikilink syntax cannot "
            "carry; open them directly.",
            "",
        ])
        lines.extend(f"- `{page.path}`" for page in sorted(unlinkable, key=lambda p: p.path))
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def write_index(store: WikiStore, *, path: Optional[Path] = None,
                memory_dir: str = "", title: str = "Wiki Index",
                today: Optional[str] = None, force: bool = False) -> Optional[Path]:
    """Regenerate ``index.md`` at the wiki root. Never raises.

    Refuses (returns None, logs) when the existing file is not hwiki's: an index a person
    wrote is not a derived artifact, and overwriting it would silently delete their
    navigation. ``force=True`` is the explicit opt-in.
    """
    target = path or (store.root / INDEX_FILENAME)
    try:
        store.safe_write_target(target)
    except ValueError as exc:
        logger.warning("hwiki: index write refused: %s", exc)
        return None
    try:
        with _FileLock(target, root=store.root):
            # Everything below happens under the lock, and the ownership state is
            # captured for the atomic write to verify again just before the replace:
            # a file that becomes hand-written in between is never overwritten.
            state = _owned_state(target)
            if not force and state is _NOT_OURS:
                logger.info("hwiki: %s was not generated by hwiki; leaving it alone",
                            target.name)
                return None
            expected = _stat_state(_lstat(target))
            text = build_index(store, memory_dir=memory_dir, title=title, today=today)
            if not _atomic_write(target, text, root=store.root, expected=expected):
                logger.warning("hwiki: %s changed while regenerating; left it alone",
                               target.name)
                return None
        return target
    except TimeoutError as exc:
        logger.warning("hwiki: index regeneration refused: %s", exc)
        return None
    except Exception as exc:  # a broken index must never break a write path
        logger.warning("hwiki: index regeneration failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# log.md
# ---------------------------------------------------------------------------

_LOG_HEADER = (
    "---\n"
    "title: Wiki Log\n"
    "type: log\n"
    "status: auto\n"
    "source: hwiki-catalog\n"
    "tags: [log]\n"
    "---\n\n"
    "# Wiki Log\n\n"
    "> Append-only record of what the memory provider did and when.\n"
    '> Entries are parseable: `grep "^## \\[" log.md | tail -5`.\n\n'
)


def append_log(store: WikiStore, kind: str, subject: str, *, detail: str = "",
               when: Optional[datetime] = None,
               path: Optional[Path] = None) -> Optional[Path]:
    """Append one ``## [date] kind | subject`` entry to ``log.md``. Never raises.

    Same ownership rule as ``index.md``: a hand-written (non-empty) log is not ours to
    append to. An empty one is adopted — a truncated file must not silence the trail.

    Append-only in the strict sense: existing bytes are never rewritten. If the file
    does not end with a newline (a crash mid-write), one separator newline is added so
    the new entry starts on its own line instead of merging into the broken one. The
    file is locked while appending so two turns cannot interleave or both write the
    header for a fresh log.
    """
    target = path or (store.root / LOG_FILENAME)
    # The log lives in the vault, which may be a shared directory: a leaf symlink, a
    # hardlink to an outside inode, or a symlinked parent directory would redirect the
    # append (and the lock) outside it. Refuse before touching anything; the checks
    # are repeated on the opened descriptor, so a swap after this point is caught too.
    try:
        store.safe_write_target(target)
    except ValueError as exc:
        logger.warning("hwiki: log append refused: %s", exc)
        return None
    if not owns_artifact(target):
        logger.info("hwiki: %s was not generated by hwiki; leaving it alone", target.name)
        return None
    if when is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    elif isinstance(when, datetime):
        stamp = when.strftime("%Y-%m-%d")
    else:
        # A wrong date is worse than no entry in an audit trail.
        logger.warning("hwiki: log entry refused: when=%r is not a datetime", when)
        return None
    kind = _CONTROL_RE.sub("", _collapse(
        kind if isinstance(kind, str) else "" if kind is None else str(kind)))
    subject = _CONTROL_RE.sub("", _collapse(
        subject if isinstance(subject, str) else "" if subject is None else str(subject)))
    detail = _CONTROL_RE.sub("", _collapse(
        detail if isinstance(detail, str) else "" if detail is None else str(detail)))
    kind = (kind or "event").split()[0]
    subject = subject[:200] or "(no subject)"
    detail = detail[:500]
    entry = f"## [{stamp}] {kind} | {subject}\n"
    if detail:
        entry += f"\n{detail}\n"
    entry += "\n"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        store.safe_write_target(target)  # the parent may now exist; re-check it
        with _FileLock(target, root=store.root):
            flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
            fd = _open_anchored(store.root, target, flags, create_parent=True)
            with os.fdopen(fd, "a+", encoding="utf-8", newline="\n") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
                    logger.warning(
                        "hwiki: log append refused: %s is not a plain, unshared file",
                        target)
                    return None
                # Ownership is checked again *under the lock*: another process may have
                # replaced our log with a hand-written file since the cheap pre-check.
                handle.seek(0)
                head = handle.read(_OWNER_SCAN_BYTES)
                if head and not _owned_head(head):
                    logger.warning(
                        "hwiki: %s was replaced by hand-written content; "
                        "leaving it alone", target.name)
                    return None
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                prefix = ""
                if size:
                    handle.seek(size - 1)
                    if handle.read(1) != "\n":
                        prefix = "\n"
                    handle.seek(0, os.SEEK_END)
                if size == 0:
                    handle.write(_LOG_HEADER)
                handle.write(prefix + entry)
        return target
    except Exception as exc:
        logger.warning("hwiki: log append failed: %s", exc)
        return None


def _tail_entries(target: Path, limit: int) -> List[Dict[str, str]]:
    """Parse the newest ``limit`` entries without reading the whole log.

    The log is append-only and grows for the life of the vault; reading it whole to
    return the last few entries made every status read cost a multi-megabyte
    allocation. The window starts small and grows only when the tail does not hold
    enough entries. When the window does not start at byte 0 its first line may be
    partial (or split a UTF-8 character), so it is dropped.
    """
    size = target.stat().st_size
    if size == 0:
        return []
    window = _TAIL_WINDOW
    while True:
        offset = max(0, size - window)
        with target.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(size - offset)
        lines = data.decode("utf-8", errors="replace").splitlines()
        needed = limit
        if offset > 0:
            lines = lines[1:]
            needed = limit + 1  # one real entry may have been dropped with the slice
        entries: List[Dict[str, str]] = []
        for line in lines:
            match = LOG_ENTRY_RE.match(line.strip())
            if match:
                entries.append({"date": match.group(1), "kind": match.group(2),
                                "subject": match.group(3)})
        if len(entries) >= needed or offset == 0:
            return entries[-limit:]
        window *= 4


def read_log_tail(store: WikiStore, limit: int = 10,
                  path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Parse the entries of ``log.md``, newest last. ``limit <= 0`` returns all.

    A short tail is read from the end of the file (see :func:`_tail_entries`); a
    malformed or partially written line is not an entry and is skipped.
    """
    target = path or (store.root / LOG_FILENAME)
    try:
        if limit > 0:
            return _tail_entries(target, limit)
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: List[Dict[str, str]] = []
    for line in text.splitlines():
        match = LOG_ENTRY_RE.match(line.strip())
        if match:
            out.append({"date": match.group(1), "kind": match.group(2),
                        "subject": match.group(3)})
    return out
