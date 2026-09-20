"""Per-session event log and fan-out, for the browser's SSE stream.

Two properties the frontend depends on:

* **Monotonic sequence numbers.** The client reconnects with `?after=<seq>` and
  receives everything it missed. A dropped connection mid-beat must not leave the
  UI stuck on a stale status, and polling the whole session document instead
  would be both chattier and racier.
* **Durable replay.** Events append to `events.jsonl`, so a client that
  reconnects after the process restarted still catches up, and the log doubles as
  the per-beat latency trace for tuning the time budget.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("h3game.events")

_QUEUE_MAX = 256


class Event(dict):
    """Plain dict so it serialises without ceremony."""


class SessionBus:
    def __init__(self, session_id: str, log_path: Path) -> None:
        self.session_id = session_id
        self.log_path = log_path
        self.seq = 0
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._recent: list[Event] = []
        self._load_tail()

    def _load_tail(self) -> None:
        """Recover the sequence counter and a replay window after a restart."""
        if not self.log_path.exists():
            return
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines[-400:]:
            line = line.strip()
            if not line:
                continue
            try:
                self._recent.append(Event(json.loads(line)))
            except ValueError:
                continue
        if self._recent:
            self.seq = int(self._recent[-1].get("seq", 0))

    def emit(self, type_: str, **payload: Any) -> Event:
        self.seq += 1
        event = Event(seq=self.seq, type=type_, ts=round(time.time(), 3), **payload)

        self._recent.append(event)
        if len(self._recent) > 400:
            del self._recent[: len(self._recent) - 400]

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("event log write failed for %s: %s", self.session_id, exc)

        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # A subscriber this far behind is a stalled browser tab. Drop it
                # rather than letting it apply backpressure to the pipeline; the
                # client will reconnect with `after` and replay from the log.
                log.warning("dropping a lagging subscriber on %s", self.session_id)
                self._subscribers.discard(q)
        return event

    def replay(self, after: int) -> list[Event]:
        return [e for e in self._recent if int(e.get("seq", 0)) > after]

    def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


class EventHub:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._buses: dict[str, SessionBus] = {}

    def bus(self, session_id: str) -> SessionBus:
        if session_id not in self._buses:
            path = self.root / "sessions" / session_id / "events.jsonl"
            self._buses[session_id] = SessionBus(session_id, path)
        return self._buses[session_id]
