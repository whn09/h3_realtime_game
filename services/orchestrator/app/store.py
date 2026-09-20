"""Session persistence.

One JSON document per session, plus a plain-text copy of every compiled IR.

A single JSON file per session rather than a database: sessions are small (tens
of beats, a few KB each), there is exactly one writer process, and DESIGN.md
section 11 wants the premise and every IR on disk and auditable. A file tree you
can `cat` and `grep` satisfies that better than rows in SQLite, and there are no
migrations to run when the schema moves -- which it will, often, early on.

Writes are debounced: a beat changes status several times a second during a
pipeline, and there is no reason for each transition to hit the disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
from pathlib import Path

from .config import settings
from .schema import Session

log = logging.getLogger("h3game.store")

_FLUSH_DELAY_S = 0.5

# What `create_session` can produce, loosened to any of the same characters so a
# hand-made id from a test still works. Notably excludes `/`, `.` and `~`.
_SID_OK = re.compile(r"[A-Za-z0-9_-]{1,128}")


class Store:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.data_dir
        self._locks: dict[str, asyncio.Lock] = {}
        self._dirty: set[str] = set()
        self._cache: dict[str, Session] = {}
        self._flusher: asyncio.Task | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sessions").mkdir(parents=True, exist_ok=True)
        self._flusher = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        if self._flusher:
            self._flusher.cancel()
            try:
                await self._flusher
            except asyncio.CancelledError:
                pass
        await self.flush_all()

    # -- paths -------------------------------------------------------------- #

    def session_dir(self, sid: str) -> Path:
        return self.root / "sessions" / sid

    def _doc_path(self, sid: str) -> Path:
        return self.session_dir(sid) / "session.json"

    def lock(self, sid: str) -> asyncio.Lock:
        """Per-session mutation lock.

        The pipeline runs several coroutines against one session concurrently
        (a Director expanding while two beats generate), so every read-modify
        -write of the document has to hold this."""
        if sid not in self._locks:
            self._locks[sid] = asyncio.Lock()
        return self._locks[sid]

    # -- read / write ------------------------------------------------------- #

    def put(self, session: Session) -> None:
        self._cache[session.id] = session
        self.touch(session.id)

    def touch(self, sid: str) -> None:
        self._dirty.add(sid)
        if sid in self._cache:
            self._cache[sid].updated_at = time.time()

    def get(self, sid: str) -> Session | None:
        if sid in self._cache:
            return self._cache[sid]
        path = self._doc_path(sid)
        if not path.exists():
            return None
        try:
            session = Session.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.error("session %s on disk is unreadable: %s", sid, exc)
            return None
        self._cache[sid] = session
        return session

    def delete(self, sid: str) -> bool:
        """Erase a session: the document, its IR archive, its event log, and every
        rendered byte it produced.

        Three separate trees, because assets are laid out for serving rather than
        for deletion:

        * `DATA_DIR/sessions/<sid>/`   -- session.json, ir/, events.jsonl
        * `ASSETS_DIR/<sid>/`          -- keyframes (keyframe.py names them by sid)
        * `ASSETS_DIR/_h3/<sid>-*/`    -- the clips, one directory per beat, named
          by the GPU job id `f"{sid}-{beat.id}"`. `_fake/` for the ffmpeg backend.
          Globbing the prefix also catches the `-r` degraded retakes, which is the
          point: they are the same beat's earlier takes and nothing else refers to
          them.

        The clips are the whole reason this exists. A session document is a few KB;
        its clips are ~10MB per beat, and a browsed-around tree keeps the branches
        nobody watched. Deleting the row and leaving those behind would make the
        button a lie -- the disk would never go down.

        Dropped from the in-memory caches first, and discarded from `_dirty`
        before any file is touched: the flusher runs every 0.5s and a pending write
        would otherwise recreate `session.json` moments after it was removed, which
        is a session that exists on disk but is gone from every cache.
        """
        # The id arrives from a URL path segment and every path below is built by
        # joining it, so this is the one place in the service where an unchecked
        # `sid` would be an `rmtree` on a caller-chosen directory: `".."` alone
        # resolves `session_dir` to DATA_DIR itself. Real ids are
        # `<unix-seconds>-<6 hex>`; nothing outside this alphabet has ever been one.
        if not sid or not _SID_OK.fullmatch(sid):
            log.warning("refusing to delete implausible session id %r", sid[:80])
            return False

        self._dirty.discard(sid)
        self._cache.pop(sid, None)
        self._locks.pop(sid, None)

        doc_dir = self.session_dir(sid)
        existed = doc_dir.exists()
        targets = [doc_dir, settings.assets_dir / sid]
        for pool in ("_h3", "_fake"):
            base = settings.assets_dir / pool
            if base.is_dir():
                targets += sorted(base.glob(f"{sid}-*"))

        for path in targets:
            if not path.exists():
                continue
            try:
                shutil.rmtree(path)
            except OSError as exc:
                # Keep going. A half-deleted session is worse than one that lost a
                # single directory, and the caller has already been told it is
                # gone -- it is out of every cache and out of `list_sessions`.
                log.error("deleting %s: could not remove %s: %s", sid, path, exc)
        log.info("deleted session %s (%d paths)", sid, len(targets))
        return existed

    def list_sessions(self, limit: int = 50) -> list[dict[str, object]]:
        rows = []
        base = self.root / "sessions"
        if not base.exists():
            return rows
        for d in base.iterdir():
            if not (d / "session.json").exists():
                continue
            session = self.get(d.name)
            if not session:
                continue
            # A still from where the player actually stopped, falling back down
            # the chain: the cursor's own poster, then the opening keyframe. A row
            # the player cannot recognise on sight is not a history entry, it is a
            # list of ids -- and the id is the one thing about a past run nobody
            # remembers.
            cursor = session.beats.get(session.cursor or "")
            thumb = (cursor.poster_url if cursor else None) or session.opening_keyframe_url
            rows.append(
                {
                    "id": session.id,
                    "phase": session.phase.value,
                    "premise": session.premise[:120],
                    "genre": (session.bible.genre if session.bible else "") or session.genre,
                    "logline": session.bible.logline if session.bible else "",
                    "thumb_url": thumb,
                    "beats": len(session.beats),
                    # How far the story got, which is what the player remembers --
                    # not `beats`, which counts un-taken branches too and so makes
                    # a 2-choice run look like a 7-beat epic.
                    "path_length": len(session.path),
                    "act": cursor.state_after.act if cursor else 1,
                    "location": cursor.state_after.location if cursor else "",
                    "ended": session.phase.value == "ended",
                    "updated_at": session.updated_at,
                }
            )
        rows.sort(key=lambda r: r["updated_at"], reverse=True)
        return rows[:limit]

    # -- flushing ----------------------------------------------------------- #

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(_FLUSH_DELAY_S)
            try:
                await self.flush_all()
            except Exception as exc:  # noqa: BLE001
                log.error("flush failed: %s", exc)

    async def flush_all(self) -> None:
        for sid in list(self._dirty):
            self._dirty.discard(sid)
            session = self._cache.get(sid)
            if session:
                await asyncio.to_thread(self._write_doc, session)

    def _write_doc(self, session: Session) -> None:
        d = self.session_dir(session.id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "session.json.tmp"
        # Write-then-rename: a crash mid-write leaves the previous good document
        # rather than a truncated one.
        tmp.write_text(session.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self._doc_path(session.id))

    # -- audit trail -------------------------------------------------------- #

    def write_ir(self, sid: str, beat_id: str, prompt: str, meta: dict[str, object]) -> None:
        """Archive the exact prompt string sent to H3.

        This is the thing to look at when a beat comes out wrong: the IR is what
        the model actually saw, and it is otherwise buried inside the request."""
        d = self.session_dir(sid) / "ir"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{beat_id}.txt").write_text(prompt, encoding="utf-8")
        (d / f"{beat_id}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
