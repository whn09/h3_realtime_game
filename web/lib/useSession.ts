"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { eventsUrl, getSession } from "./api";
import type { Beat, GameEvent, SessionView } from "./types";

/**
 * One session, kept live off the SSE stream.
 *
 * The document is fetched once and then only ever patched by events, which is
 * what lets the UI show a beat's status changing (`compiling` -> `waiting_frame`
 * -> `generating`) without polling a 14-second pipeline.
 *
 * Reconnection is manual rather than EventSource's own: its automatic retry
 * always re-opens the *original* URL, so it would replay from `after=0` forever.
 * We close on error and re-open at the last sequence number we actually saw, and
 * still de-duplicate on `seq` -- a replay window that overlaps by a few events is
 * normal, because the server subscribes before it replays so that nothing
 * emitted mid-replay is lost.
 */
export interface SessionStore {
  session: SessionView | null;
  events: GameEvent[];
  connected: boolean;
  error: string | null;
  /** Patch locally without waiting for the server to echo it back. */
  patchBeat: (id: string, patch: Partial<Beat>) => void;
  reload: () => Promise<void>;
}

const MAX_EVENT_LOG = 300;

export function useSession(sid: string | null): SessionStore {
  /**
   * The document, tagged with the session it belongs to.
   *
   * Tagged rather than bare, and that is the whole point of this shape: the
   * document used to be plain state, cleared by nothing, so switching runs left
   * the *previous* session's document in place until the new `GET` came back. For
   * that window -- one render, then however long the request takes -- `session`
   * described run A while `sid` said run B, and `Player` mounted against it and
   * did exactly what it is supposed to do with a session whose cursor has a clip:
   * played it. That is the "switched stories and the old one was still playing"
   * report, and it was audible as well as visible because the clip carries its
   * own score.
   *
   * Clearing it in an effect would not have fixed it; effects run after the
   * render that already handed the stale document to `Player`. So the mismatch is
   * resolved during render instead: `session` below is `null` unless the document
   * in hand is the one being asked for, which makes "loading a different run"
   * indistinguishable from "loading the first run" -- there is no state in which
   * a caller can be handed someone else's story.
   */
  const [loaded, setLoaded] = useState<{ sid: string; doc: SessionView } | null>(null);
  const [events, setEvents] = useState<GameEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const session = loaded && loaded.sid === sid ? loaded.doc : null;

  const lastSeq = useRef(0);
  const seen = useRef<Set<number>>(new Set());
  const source = useRef<EventSource | null>(null);
  const retry = useRef<ReturnType<typeof setTimeout> | null>(null);

  const reload = useCallback(async () => {
    if (!sid) return;
    try {
      const doc = await getSession(sid);
      setLoaded({ sid, doc });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [sid]);

  /**
   * Patch the document, but only if it is still the one this event came from. A
   * late event from the previous run's stream -- already closed, but a message can
   * be in flight past `close()` -- would otherwise be reduced into the new run's
   * document, inserting a beat that belongs to a different story.
   */
  const apply = useCallback((forSid: string, event: GameEvent) => {
    setLoaded((prev) =>
      prev && prev.sid === forSid ? { ...prev, doc: reduce(prev.doc, event) } : prev
    );
  }, []);

  useEffect(() => {
    if (!sid) return;
    let cancelled = false;
    lastSeq.current = 0;
    seen.current = new Set();
    // The event log belongs to a session, not to the app. Carrying the previous
    // run's events into this one makes the debug overlay describe the wrong story.
    setEvents([]);
    setError(null);

    const connect = () => {
      if (cancelled) return;
      const es = new EventSource(eventsUrl(sid, lastSeq.current));
      source.current = es;

      es.onopen = () => {
        setConnected(true);
        setError(null);
      };

      // Every event type shares one handler: the server sets the SSE `event:`
      // field, so `onmessage` alone would miss all of them.
      const onAny = (ev: MessageEvent<string>) => {
        let parsed: GameEvent;
        try {
          parsed = JSON.parse(ev.data) as GameEvent;
        } catch {
          return;
        }
        if (seen.current.has(parsed.seq)) return;
        seen.current.add(parsed.seq);
        lastSeq.current = Math.max(lastSeq.current, parsed.seq);
        apply(sid, parsed);
        setEvents((prev) => [...prev.slice(-(MAX_EVENT_LOG - 1)), parsed]);
      };

      for (const type of EVENT_TYPES) es.addEventListener(type, onAny as EventListener);
      es.onmessage = onAny;

      es.onerror = () => {
        setConnected(false);
        es.close();
        if (cancelled) return;
        // 1s is short enough that a restarted orchestrator is picked up while the
        // player is still looking at the same frame.
        retry.current = setTimeout(connect, 1000);
      };
    };

    void reload().then(connect);

    return () => {
      cancelled = true;
      if (retry.current) clearTimeout(retry.current);
      source.current?.close();
      source.current = null;
    };
  }, [sid, apply, reload]);

  const patchBeat = useCallback((id: string, patch: Partial<Beat>) => {
    setLoaded((prev) => {
      if (!prev?.doc.beats[id]) return prev;
      const doc = prev.doc;
      return {
        ...prev,
        doc: { ...doc, beats: { ...doc.beats, [id]: { ...doc.beats[id], ...patch } } },
      };
    });
  }, []);

  return useMemo(
    () => ({ session, events, connected, error, patchBeat, reload }),
    [session, events, connected, error, patchBeat, reload]
  );
}

const EVENT_TYPES = [
  "session.created",
  "session.failed",
  "worldsmith.started",
  "bible.ready",
  "beat.created",
  "beat.status",
  "beat.ir",
  "beat.ready",
  "beat.failed",
  "keyframe.ready",
  "options.ready",
  "cursor.moved",
  "custom.started",
  "summary.updated",
  "drift.reanchor",
  "reanchor.forced",
  "ending",
  "error",
] as const;

function withBeat(s: SessionView, beat: Beat): SessionView {
  const prev = s.beats[beat.id];
  return {
    ...s,
    // Merge rather than replace. `beat.created` and `beat.ready` are both full
    // summaries, but `cursor.moved` can arrive before a `beat.ready` that was
    // emitted for the same beat, and an out-of-order overwrite would drop the
    // video URL the player is about to need.
    beats: { ...s.beats, [beat.id]: prev ? { ...prev, ...beat } : beat },
  };
}

function reduce(s: SessionView, e: GameEvent): SessionView {
  switch (e.type) {
    case "bible.ready":
      return { ...s, bible: e.bible as SessionView["bible"], phase: "opening" };

    case "beat.created": {
      const beat = e.beat as Beat;
      const next = withBeat(s, beat);
      const parentId = e.parent_id as string | undefined;
      if (!parentId) return { ...next, root_id: beat.id, cursor: s.cursor ?? beat.id };
      return next;
    }

    case "beat.ready":
    case "cursor.moved": {
      const beat = e.beat as Beat;
      let next = withBeat(s, beat);
      if (e.type === "cursor.moved") {
        next = {
          ...next,
          cursor: beat.id,
          path: (e.path as string[] | undefined) ?? next.path,
        };
      }
      // The first finished beat flips the session out of its opening ritual.
      if (next.phase === "opening" || next.phase === "creating") {
        if (beat.status === "ready") next = { ...next, phase: "playing" };
      }
      return next;
    }

    case "beat.status": {
      const id = e.beat_id as string;
      if (!s.beats[id]) return s;
      return {
        ...s,
        beats: { ...s.beats, [id]: { ...s.beats[id], status: e.status as Beat["status"] } },
      };
    }

    case "beat.failed": {
      const id = e.beat_id as string;
      if (!s.beats[id]) return s;
      return {
        ...s,
        beats: {
          ...s.beats,
          [id]: { ...s.beats[id], status: "failed", error: String(e.detail ?? "") },
        },
      };
    }

    case "keyframe.ready": {
      const id = e.beat_id as string;
      if (!s.beats[id]) return s;
      return {
        ...s,
        beats: { ...s.beats, [id]: { ...s.beats[id], keyframe_url: e.url as string } },
      };
    }

    case "options.ready": {
      const id = e.beat_id as string;
      if (!s.beats[id]) return s;
      return {
        ...s,
        beats: {
          ...s.beats,
          [id]: {
            ...s.beats[id],
            narration: (e.narration as string) ?? s.beats[id].narration,
            options: (e.options as Beat["options"]) ?? [],
            children: (e.children as Record<string, string>) ?? {},
            predicted_choice: (e.predicted_choice as number) ?? 0,
          },
        },
      };
    }

    case "ending": {
      const id = e.beat_id as string;
      const beats = s.beats[id]
        ? { ...s.beats, [id]: { ...s.beats[id], is_ending: e.ending as Beat["is_ending"] } }
        : s.beats;
      return { ...s, beats, phase: "ended" };
    }

    case "session.failed":
      return { ...s, phase: "failed", error: String(e.detail ?? "unknown") };

    default:
      return s;
  }
}

/** root -> cursor, resolved from parent links so it survives a missing `path`. */
export function pathToCursor(s: SessionView): Beat[] {
  const out: Beat[] = [];
  let id = s.cursor;
  const guard = new Set<string>();
  while (id && s.beats[id] && !guard.has(id)) {
    guard.add(id);
    out.unshift(s.beats[id]);
    id = s.beats[id].parent_id;
  }
  return out;
}
