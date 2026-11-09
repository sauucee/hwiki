"""Filesystem model for a markdown wiki used as agent memory.

Pure stdlib. Responsible for:

* enumerating wiki pages (with include/exclude globs and a hard-excluded
  ``untrusted`` zone),
* parsing YAML-ish frontmatter without requiring PyYAML,
* extracting Obsidian ``[[wikilinks]]`` and building a link graph,
* chunking pages into heading-scoped sections for retrieval,
* appending durable notes into the *memory zone* under a write lock,
* refusing to persist anything that looks like a secret.

Design rule: this module NEVER writes outside ``<root>/<memory_dir>``.
The curated wiki is read-only to the memory provider — automatic writes
must not corrupt hand-maintained synthesis pages.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import stat
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:  # POSIX advisory locking; absent on Windows.
    import fcntl
except Exception:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)
_FENCE_OPEN_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_FENCE_CLOSE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*$")
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def strip_code(text: str) -> str:
    """Remove fenced blocks and inline code: a link inside them is literal text.

    Markdown (and Obsidian) do not make a ``[[wikilink]]`` inside a fence or a pair of
    backticks a link. Reading it as one invented edges, backlinks and dangling
    reports — a vault that documents wikilink syntax describes links it does not have.
    """
    kept: List[str] = []
    fence: Optional[Tuple[str, int]] = None
    for line in (text or "").splitlines(keepends=True):
        stripped = line.lstrip()
        if fence is None:
            opened = _FENCE_OPEN_RE.match(stripped)
            if opened:
                fence = (opened.group(1)[0], len(opened.group(1)))
                continue
            kept.append(line)
            continue
        closed = _FENCE_CLOSE_RE.match(stripped)
        # CommonMark: a closing fence is the same character and *at least* as long as
        # the opener. Comparing exact strings let ``` opened by ```` run to the next
        # three-backtick line, so the fence's samples became phantom edges.
        if closed and closed.group(1)[0] == fence[0] and len(closed.group(1)) >= fence[1]:
            fence = None
    return _INLINE_CODE_RE.sub("", "".join(kept))


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def parse_frontmatter(text: str) -> Tuple[Dict[str, object], str]:
    """Split ``text`` into (frontmatter dict, body).

    Uses PyYAML when importable, otherwise a deliberately small parser that
    handles the subset this wiki uses: ``key: scalar``, ``key: [a, b]`` and
    ``key:`` followed by ``- item`` lines. Unknown constructs degrade to the
    raw string rather than raising — a malformed page must never break recall.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw = match.group(1)
    body = text[match.end():]

    try:  # Prefer a real YAML parser when the host has one.
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items()}, body
    except Exception:
        pass

    data: Dict[str, object] = {}
    current_key: Optional[str] = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and current_key:
            existing = data.get(current_key)
            item = stripped[2:].strip().strip("'\"")
            if isinstance(existing, list):
                existing.append(item)
            else:
                data[current_key] = [item]
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        current_key = key
        if not value:
            data[key] = []
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            data[key] = [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
        else:
            data[key] = value.strip("'\"")
    return data, body


def _as_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


# ---------------------------------------------------------------------------
# Slugs
# ---------------------------------------------------------------------------

def _is_traversal(reference: str) -> bool:
    """True for an absolute or parent-escaping reference.

    Such a reference can never name a page: the store refuses paths outside the wiki
    root, so it is refused up front instead of falling through to a basename or title
    match that would resolve some unrelated page.
    """
    if reference.startswith(("/", "\\")):
        return True
    return any(part == ".." for part in re.split(r"[\\/]", reference))


def slugify(value: str, *, fallback: str = "note") -> str:
    """Filesystem-safe, lowercase, hyphenated slug."""
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    value = re.sub(r"[\s_]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:80] or fallback


# ---------------------------------------------------------------------------
# Secret guard
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: Sequence[Tuple[str, re.Pattern]] = (
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Scaleway secret key", re.compile(r"\bSCW[A-Z0-9]{8}[A-Z0-9-]{3,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("inline credential", re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|client[_-]?secret)\b"
        r"\s*[:=]\s*[\"']?[^\s\"',;]{8,}"
    )),
)


def scan_for_secrets(text: str) -> List[str]:
    """Return human-readable labels for any secret-shaped content found."""
    return [label for label, pattern in _SECRET_PATTERNS if pattern.search(text or "")]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A heading-scoped slice of a page — the unit of retrieval."""

    path: str
    heading: str
    text: str
    order: int


@dataclass
class Page:
    path: str                      # wiki-root-relative POSIX path
    abspath: Path
    title: str
    body: str
    frontmatter: Dict[str, object] = field(default_factory=dict)
    mtime: float = 0.0
    size: int = 0

    @property
    def slug(self) -> str:
        return Path(self.path).stem

    @property
    def spine(self) -> str:
        return _as_str(self.frontmatter.get("spine"))

    @property
    def doctype(self) -> str:
        return _as_str(self.frontmatter.get("type"))

    @property
    def status(self) -> str:
        return _as_str(self.frontmatter.get("status"))

    @property
    def updated(self) -> str:
        return _as_str(self.frontmatter.get("updated"))

    def links(self) -> List[str]:
        """Outbound ``[[wikilink]]`` targets, de-duplicated, order preserved."""
        seen: Dict[str, None] = {}
        for match in _WIKILINK_RE.finditer(strip_code(self.body)):
            target = match.group(1).strip()
            if target:
                seen.setdefault(target, None)
        return list(seen)

    def chunks(self, max_chars: int = 1200) -> List[Chunk]:
        """Split the body on H2/H3 boundaries, hard-wrapping long sections."""
        sections: List[Tuple[str, List[str]]] = [(self.title, [])]
        for line in self.body.splitlines():
            heading = _HEADING_RE.match(line)
            if heading and len(heading.group(1)) <= 3:
                sections.append((heading.group(2).strip(), []))
            else:
                sections[-1][1].append(line)

        out: List[Chunk] = []
        for heading, lines in sections:
            text = "\n".join(lines).strip()
            if not text:
                continue
            if len(text) <= max_chars:
                out.append(Chunk(self.path, heading, text, len(out)))
                continue
            buf: List[str] = []
            length = 0
            for para in text.split("\n\n"):
                if length + len(para) > max_chars and buf:
                    out.append(Chunk(self.path, heading, "\n\n".join(buf), len(out)))
                    buf, length = [], 0
                buf.append(para)
                length += len(para) + 2
            if buf:
                out.append(Chunk(self.path, heading, "\n\n".join(buf), len(out)))
        return out


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDES: Tuple[str, ...] = (
    ".git/*", ".obsidian/*", "node_modules/*", ".trash/*",
    "*/.git/*", "*/node_modules/*",
)


class WikiStore:
    """Read/write access to a markdown wiki rooted at ``root``.

    Parameters
    ----------
    root:
        Wiki root directory.
    include:
        Root-relative glob(s) selecting indexable pages. Defaults to ``**/*.md``.
    exclude:
        Root-relative glob(s) removed from the index.
    untrusted_dirs:
        Directories holding third-party source material. These are NEVER
        indexed and NEVER surfaced through prefetch — they are the natural
        home of prompt-injection payloads (customer PDFs, vendor advisories).
    memory_dir:
        Root-relative directory that the provider may write to.
    """

    def __init__(
        self,
        root: os.PathLike | str,
        *,
        include: Sequence[str] = ("**/*.md",),
        exclude: Sequence[str] = (),
        untrusted_dirs: Sequence[str] = ("raw",),
        memory_dir: str = "wiki/memory",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.include = tuple(include) or ("**/*.md",)
        self.exclude = tuple(exclude) + DEFAULT_EXCLUDES
        self.untrusted_dirs = tuple(d.strip("/") for d in untrusted_dirs if d.strip("/"))
        self.memory_dir = memory_dir.strip("/")

    # -- discovery ---------------------------------------------------------

    def exists(self) -> bool:
        return self.root.is_dir()

    def _is_untrusted(self, rel: str) -> bool:
        return any(rel == d or rel.startswith(d + "/") for d in self.untrusted_dirs)

    def _excluded(self, rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, pattern) for pattern in self.exclude)

    def iter_paths(self) -> Iterator[Path]:
        """Yield absolute paths of every indexable page."""
        if not self.exists():
            return
        seen: set[Path] = set()
        for pattern in self.include:
            for abspath in self.root.glob(pattern):
                if not abspath.is_file() or abspath in seen:
                    continue
                try:
                    rel = abspath.resolve().relative_to(self.root).as_posix()
                except ValueError:
                    continue  # symlink escaping the root
                if self._is_untrusted(rel) or self._excluded(rel):
                    continue
                seen.add(abspath)
                yield abspath

    def relpath(self, abspath: os.PathLike | str) -> str:
        return Path(abspath).resolve().relative_to(self.root).as_posix()

    def abspath(self, rel: str) -> Path:
        """Resolve a root-relative path, refusing traversal outside the root."""
        candidate = (self.root / rel.lstrip("/")).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"path escapes wiki root: {rel}")
        return candidate

    # -- reading -----------------------------------------------------------

    def load(self, abspath: os.PathLike | str) -> Optional[Page]:
        path = Path(abspath)
        try:
            stat = path.stat()
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        frontmatter, body = parse_frontmatter(text)
        title = _as_str(frontmatter.get("title"))
        if not title:
            for line in body.splitlines():
                heading = _HEADING_RE.match(line)
                if heading:
                    title = heading.group(2).strip()
                    break
        try:
            rel = self.relpath(path)
        except ValueError:
            return None
        return Page(
            path=rel,
            abspath=path,
            title=title or Path(rel).stem,
            body=body,
            frontmatter=frontmatter,
            mtime=stat.st_mtime,
            size=stat.st_size,
        )

    def load_rel(self, rel: str) -> Optional[Page]:
        """Load a root-relative page, refusing untrusted/excluded locations."""
        try:
            abspath = self.abspath(rel)
        except ValueError:
            return None
        try:
            normalized = self.relpath(abspath)
        except ValueError:
            return None
        if self._is_untrusted(normalized) or self._excluded(normalized):
            return None
        return self.load(abspath)

    def links_for(self, rel: str, *, cap: int = 8) -> List[str]:
        """Outbound ``[[wikilink]]`` targets of a page, capped.

        ``load_rel`` already refuses untrusted/excluded locations, so the
        raw/ boundary holds even if a backend ever returns a stray path.
        """
        page = self.load_rel(rel)
        if page is None:
            return []
        return page.links()[:cap]

    def iter_pages(self) -> Iterator[Page]:
        for abspath in self.iter_paths():
            page = self.load(abspath)
            if page is not None:
                yield page

    @staticmethod
    def _normalize_reference(reference: str) -> str:
        """Strip ``[[ ]]``, ``|alias`` and ``#heading`` from a link target."""
        reference = (reference or "").strip()
        if reference.startswith("[[") and reference.endswith("]]"):
            reference = reference[2:-2]
        return reference.split("|", 1)[0].split("#", 1)[0].strip()

    def _resolve_in(self, reference: str, pages: Sequence[Page]) -> Optional[Page]:
        """Resolve one reference against ``pages``. The single order of record.

        Every consumer (``resolve_page``, ``link_graph``, ``dangling_links``) goes
        through here, so a link the reader can follow is never reported as dangling
        and never keyed as a page that does not exist.
        """
        raw = (reference or "").strip()
        bracketed = raw.startswith("[[") and raw.endswith("]]")
        if bracketed:
            raw = raw[2:-2]
        # A literal page path is data, not wikilink syntax. The catalog lists pages
        # whose names contain ``#`` or ``|`` as plain paths, so the exact path has to
        # be matched *before* the target is parsed as "page, heading/alias" —
        # otherwise the catalog named a page nothing could open again.
        if not bracketed and raw:
            if _is_traversal(raw):
                return None
            for page in pages:
                if page.path == raw:
                    return page
                if page.path.lower().endswith(".md") and page.path[:-3] == raw:
                    return page
        reference = self._normalize_reference(raw)
        if not reference or _is_traversal(reference):
            return None
        for page in pages:
            if page.path == reference:
                return page
            if page.path.lower().endswith(".md") and page.path[:-3] == reference:
                return page
        if reference.lower().endswith(".md"):
            page = self.load_rel(reference)
            if page:
                return page
            # a guessed folder must not pick between twins
            stem = Path(reference).stem
            matches = [p for p in pages if p.slug == stem]
            return matches[0] if len(matches) == 1 else None
        folded = reference.lower()
        folded_matches = [
            p for p in pages
            if p.path.lower() == folded
            or (p.path.lower().endswith(".md") and p.path[:-3].lower() == folded)]
        if folded_matches:
            return folded_matches[0] if len(folded_matches) == 1 else None
        target = slugify(reference)
        fallback: Optional[Page] = None
        for page in pages:
            if page.slug == reference or slugify(page.slug) == target:
                return page
            if fallback is None and slugify(page.title) == target:
                fallback = page
        return fallback

    def resolve_page(self, reference: str) -> Optional[Page]:
        """Resolve a page by relative path, slug, wikilink target, or title.

        Order matters, and each step is narrower than the next:

        1. an exact, case-sensitive root-relative path (``Folder/Page`` or
           ``Folder/Page.md``) — the only way to name one of two pages whose paths
           differ only by case;
        2. a ``.md`` path that missed, falling back to a **unique** filename slug
           (an existing rule: a guessed folder must not resolve to a twin);
        3. a case-insensitive path, accepted only when exactly one page matches it —
           never a silent merge of ``Docs/Plan.md`` and ``docs/plan.md``;
        4. a bare slug or title, first match wins, unchanged so old links that name
           no folder keep resolving.

        The generated catalog form (``[[Folder/Page|Page]]``) and heading links
        (``Folder/Page#heading``) are accepted. Traversal (``../x``) and absolute
        references are refused instead of falling through to a basename or title.
        """
        return self._resolve_in(reference, list(self.iter_pages()))

    def link_graph(self, key: str = "slug") -> Tuple[Dict[str, List[str]],
                                                      Dict[str, List[str]]]:
        """Return (outlinks, backlinks) keyed by ``key``: ``"slug"`` or ``"path"``.

        ``"slug"`` is the default and the historical shape: callers read it by slug.
        It is lossy by construction — two pages that share a filename (``a/notes.md``
        and ``b/notes.md``) share the key ``notes``, so their backlinks merge and a
        page linking both lists the same target twice. That limit is documented, not
        fixed, because widening the default key space would break every existing
        caller; ``"path"`` is the opt-in view for callers that must attribute a link
        to exactly one page (``wiki_memory_read`` backlinks, orphan detection).

        Targets resolve exactly like ``resolve_page``: a link that resolves for the
        reader resolves here too, and one that does not keeps its raw target as the
        key instead of inventing an edge. Repeated targets collapse to one entry.
        """
        if key not in ("slug", "path"):
            raise ValueError(f"link_graph key must be 'slug' or 'path', not {key!r}")
        name = (lambda page: page.slug) if key == "slug" else (lambda page: page.path)
        outlinks: Dict[str, List[str]] = {}
        backlinks: Dict[str, List[str]] = {}
        pages = list(self.iter_pages())
        cache: Dict[str, Optional[Page]] = {}
        for page in pages:
            source = name(page)
            seen: List[str] = []
            for raw in page.links():
                lookup = raw.strip()
                if lookup not in cache:
                    cache[lookup] = self._resolve_in(lookup, pages)
                hit = cache[lookup]
                target = name(hit) if hit is not None else lookup
                if target in seen:
                    continue
                seen.append(target)
                backlinks.setdefault(target, []).append(source)
            outlinks[source] = seen
        return outlinks, backlinks

    def dangling_links(self, key: str = "slug") -> Dict[str, List[str]]:
        """Wikilinks pointing at pages that do not exist, keyed by ``key``.

        ``"slug"`` is the default and the historical shape; it is lossy — two sources
        that share a filename share one key, so one source's report appears under the
        other. ``"path"`` keys each report by the exact page to fix, and is the
        headroom-preserving opt-in: the default key space did not change under
        existing callers.

        A target counts as resolvable exactly when ``resolve_page`` resolves it, so
        this report can never disagree with what the reader can open. Traversal and
        absolute references are reported too — they never resolve, so silence would
        hide broken links.
        """
        if key not in ("slug", "path"):
            raise ValueError(f"dangling_links key must be 'slug' or 'path', not {key!r}")
        out: Dict[str, List[str]] = {}
        pages = list(self.iter_pages())
        cache: Dict[str, bool] = {}
        for page in pages:
            missing = []
            for raw in page.links():
                target = raw.strip()
                if target not in cache:
                    cache[target] = self._resolve_in(target, pages) is not None
                if not cache[target]:
                    missing.append(target)
            if missing:
                source = page.slug if key == "slug" else page.path
                out[source] = sorted(set(out.get(source, []) + missing))
        return out

    # -- writing (memory zone only) ---------------------------------------

    def safe_write_target(self, target: Path) -> Path:
        """Refuse a write target that is not a plain file inside the wiki root.

        Fail closed: the vault may be a shared directory, so a symlink (leaf or
        parent) or a hardlink is a path to a file this store has no business writing.
        Returns ``target`` unchanged when it is safe; raises ``ValueError`` otherwise.
        """
        root = os.path.realpath(self.root)
        parent = os.path.realpath(Path(target).parent)
        if parent != root and not parent.startswith(root + os.sep):
            raise ValueError(f"refusing to write outside the wiki root: {target}")
        if os.path.islink(target):
            raise ValueError(f"refusing to write through a symlink: {target}")
        try:
            info = os.lstat(target)
        except FileNotFoundError:
            return Path(target)
        except OSError as exc:
            raise ValueError(f"refusing to inspect write target {target}: {exc}") from None
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"refusing to write to a non-regular file: {target}")
        if info.st_nlink > 1:
            raise ValueError(f"refusing to write to a hardlinked file: {target}")
        return Path(target)

    def memory_root(self) -> Path:
        return self.root / self.memory_dir

    def memory_page_path(self, slug: str) -> Path:
        return self.memory_root() / f"{slugify(slug)}.md"

    def append_note(
        self,
        content: str,
        *,
        topic: str = "",
        page: str = "",
        tags: Sequence[str] = (),
        source: str = "",
        today: Optional[str] = None,
        allow_secrets: bool = False,
    ) -> Dict[str, object]:
        """Append a dated bullet to a memory-zone page, creating it if needed.

        Returns a result dict with ``path``, ``created`` and ``duplicate``.
        Raises ``ValueError`` when the content looks like a secret.
        """
        content = " ".join((content or "").split())
        if not content:
            raise ValueError("content is empty")

        if not allow_secrets:
            found = scan_for_secrets(content)
            if found:
                raise ValueError(
                    "refusing to write apparent secret material into the wiki "
                    f"({', '.join(found)}); store a pointer to the secret's "
                    "location instead"
                )

        slug = slugify(page or topic or "journal", fallback="journal")
        target = self.memory_page_path(slug)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.safe_write_target(target)

        stamp = today or date.today().isoformat()
        suffix = ""
        if tags:
            suffix += " " + " ".join(f"#{slugify(t)}" for t in tags if str(t).strip())
        if source:
            suffix += f" _(source: {source})_"
        entry = f"- **{stamp}** — {content}{suffix}"

        with _FileLock(target, root=self.root):
            created = not os.path.exists(target)
            # The open is anchored to the vault root and follows no symlink, so a leaf
            # or parent swapped for a symlink after ``safe_write_target`` is refused
            # rather than written through.
            flags = os.O_RDWR | os.O_CREAT
            fd = _open_anchored(self.root, target, flags, create_parent=True)
            with os.fdopen(fd, "r+", encoding="utf-8", newline="\n") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
                    raise ValueError(
                        f"refusing to write to {target}: not a plain, unshared file")
                if created:
                    header = (
                        "---\n"
                        f"title: {topic or slug.replace('-', ' ').title()}\n"
                        "spine: memory\n"
                        "type: memory\n"
                        "status: auto\n"
                        "source: agent-memory\n"
                        f"updated: {stamp}\n"
                        "---\n\n"
                        f"# {topic or slug.replace('-', ' ').title()}\n\n"
                        "> Auto-maintained by the `hwiki` memory provider. Promote durable\n"
                        "> conclusions into the curated wiki rather than editing in place.\n\n"
                        "## Log\n\n"
                    )
                    handle.seek(0)
                    handle.write(header + entry + "\n")
                    handle.truncate()
                    return {"path": self.relpath(target), "created": True,
                            "duplicate": False}

                existing = handle.read()
                if _normalize(content) in _normalize(existing):
                    return {"path": self.relpath(target), "created": False,
                            "duplicate": True}

                if "\n## Log" not in existing:
                    existing = existing.rstrip("\n") + "\n\n## Log\n"
                updated = existing.rstrip("\n") + "\n" + entry + "\n"
                updated = _bump_frontmatter_date(updated, stamp)
                handle.seek(0)
                handle.write(updated)
                handle.truncate()

        return {"path": self.relpath(target), "created": False, "duplicate": False}

    def content_hash(self) -> str:
        """Cheap fingerprint of the corpus (paths + mtimes + sizes)."""
        digest = hashlib.sha1()
        for abspath in sorted(self.iter_paths()):
            try:
                stat = abspath.stat()
            except OSError:
                continue
            digest.update(str(abspath).encode())
            digest.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
        return digest.hexdigest()


def _open_anchored(root: Path, target: Path, flags: int, mode: int = 0o644,
                   *, create_parent: bool = False) -> int:
    """Open ``target`` for writing under ``root`` without following any symlink.

    Every path component is opened relative to the previous directory descriptor with
    ``O_NOFOLLOW | O_DIRECTORY``, so a parent directory swapped for a symlink after a
    ``realpath`` check cannot redirect the open outside the vault. Paths are taken
    lexically from ``root``; anything not lexically below it is refused. The returned
    descriptor is owned by the caller.
    """
    root_abs = Path(os.path.abspath(root))
    target_abs = Path(os.path.abspath(target))
    try:
        rel = target_abs.relative_to(root_abs)
    except ValueError:
        raise ValueError(f"refusing to write outside the wiki root: {target}") from None
    if not rel.parts:
        raise ValueError(f"refusing to write to the wiki root itself: {target}")
    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    dir_fd = os.open(os.path.realpath(root), dir_flags)
    try:
        for part in rel.parts[:-1]:
            try:
                next_fd = os.open(part, dir_flags | nofollow, dir_fd=dir_fd)
            except FileNotFoundError:
                if not create_parent:
                    raise
                os.mkdir(part, 0o755, dir_fd=dir_fd)
                next_fd = os.open(part, dir_flags | nofollow, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = next_fd
        return os.open(rel.parts[-1], flags | nofollow, mode, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def _open_dir_beneath(root: Path, directory: Path) -> int:
    """Open ``directory`` (under ``root``) as a descriptor, following no symlinks."""
    root_abs = Path(os.path.abspath(root))
    dir_abs = Path(os.path.abspath(directory))
    try:
        rel = dir_abs.relative_to(root_abs)
    except ValueError:
        raise ValueError(f"refusing to open outside the wiki root: {directory}") from None
    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(os.path.realpath(root), dir_flags)
    for part in rel.parts:
        try:
            nxt = os.open(part, dir_flags | nofollow, dir_fd=fd)
        except OSError:
            os.close(fd)
            raise
        os.close(fd)
        fd = nxt
    return fd


def _stat_beneath(root: Path, target: Path) -> Optional[os.stat_result]:
    """``lstat`` ``target`` anchored under ``root`` (``None`` when it does not exist)."""
    root_abs = Path(os.path.abspath(root))
    target_abs = Path(os.path.abspath(target))
    try:
        rel = target_abs.relative_to(root_abs)
    except ValueError:
        raise ValueError(f"refusing to inspect outside the wiki root: {target}") from None
    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    dir_fd = os.open(os.path.realpath(root), dir_flags)
    try:
        for part in rel.parts[:-1]:
            next_fd = os.open(part, dir_flags | nofollow, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = next_fd
        return os.stat(rel.parts[-1], dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    finally:
        os.close(dir_fd)


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _bump_frontmatter_date(text: str, stamp: str) -> str:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return text
    head = match.group(0)
    if re.search(r"^updated:.*$", head, re.MULTILINE):
        new_head = re.sub(r"^updated:.*$", f"updated: {stamp}", head, count=1, flags=re.MULTILINE)
    else:
        new_head = head.rstrip("\n")
        new_head = new_head[: new_head.rfind("---")] + f"updated: {stamp}\n---\n"
    return new_head + text[match.end():]


class _FileLock:
    """Best-effort advisory lock so concurrent turns can't interleave writes.

    The lock is a separate file, and the file is **never deleted**. Unlinking it on
    release looks tidy but splits the lock: a waiter that already opened the old inode
    keeps blocking on a file no longer in the directory, a newcomer creates a new
    inode and takes "the lock" instantly, and both then hold it at once. A stale empty
    ``*.lock`` file is the cost of keeping one inode for the lock's whole life.

    The lockfile is opened with ``O_NOFOLLOW`` (and, when a vault ``root`` is given,
    through an anchored directory-descriptor walk), so a symlink — leaf or a parent
    swapped after a check — cannot redirect the lock outside the vault.

    Without advisory locking the lock **refuses to enter**: silently proceeding
    unlocked is exactly the interleaving this class exists to prevent.
    """

    def __init__(self, target: Path, timeout: float = 5.0, *,
                 root: Optional[Path] = None) -> None:
        self.lockfile = target.with_suffix(target.suffix + ".lock")
        self.timeout = timeout
        self.root = root
        self.locked = False  # True once the advisory lock is actually held
        self._handle = None

    def __enter__(self) -> "_FileLock":
        if fcntl is None:
            raise RuntimeError(
                "advisory file locking is unavailable on this platform; "
                "refusing to write without it")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        if self.root is not None:
            fd = _open_anchored(self.root, self.lockfile, flags,
                                create_parent=True)
        else:
            self.lockfile.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.lockfile, flags, 0o644)
        self._handle = os.fdopen(fd, "a+")
        deadline = time.time() + self.timeout
        while True:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.locked = True
                return self
            except OSError:
                if time.time() >= deadline:
                    # Fail closed: writing without the lock can interleave with
                    # another writer, so refuse the critical section instead — and
                    # close the descriptor we opened before raising.
                    try:
                        self._handle.close()
                    finally:
                        self._handle = None
                    raise TimeoutError(f"could not lock {self.lockfile} within "
                                       f"{self.timeout}s") from None
                time.sleep(0.05)

    def __exit__(self, *exc) -> None:
        if self._handle is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._handle.close()
        except Exception:
            pass
        # deliberately no unlink: see the class docstring (old-inode waiters)
