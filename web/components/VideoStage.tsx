"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

/**
 * A fixed pool of three `<video>` elements that are mounted once and never
 * unmounted.
 *
 * Three, not two, because both branches are pre-generated and both must be
 * buffered before the decision moment: current + branch A + branch B. Unmounting
 * to swap would tear down and rebuild a hardware decoder, which shows up as a
 * black flash of one or two frames -- exactly at the join we spent the whole
 * `fl2va` chain making invisible. So `src` is assigned imperatively through refs
 * and the swap is a CSS opacity change on elements that were already decoding.
 *
 * The pool is addressed by beat id, not by slot: callers say "buffer this beat"
 * and "show this beat", and the slot bookkeeping (including eviction, when a
 * player branches faster than the pool turns over) stays in here.
 */

export interface StageHandle {
  /** Point a slot at this clip and let it buffer. Idempotent per beat. */
  buffer: (beatId: string, url: string) => void;
  /** Make this beat visible and play it from the start. Buffers first if needed. */
  show: (beatId: string, url: string, fromStart?: boolean) => Promise<void>;
  /** Visible beat, or null before the first `show`. */
  visible: () => string | null;
  /** Current playback position of the visible clip, in seconds. */
  position: () => { time: number; duration: number };
  pause: () => void;
  resume: () => Promise<void>;
  setMuted: (muted: boolean) => void;
}

interface Props {
  onTimeUpdate?: (beatId: string, time: number, duration: number) => void;
  onEnded?: (beatId: string) => void;
  /**
   * Autoplay was refused. Browsers only allow audible playback after a user
   * gesture, and a gesture in one document does not always carry -- a reload
   * lands on a session that is mid-story with no fresh click behind it. The
   * parent turns this into a "click to continue" affordance rather than leaving
   * the player staring at a frozen frame.
   */
  onBlocked?: (beatId: string) => void;
  /**
   * The visible clip is running. Fired from the element's own `playing` event, so
   * it is the browser saying so rather than us inferring it from a resolved
   * promise.
   *
   * This is the counterpart to `onBlocked`, and the pair has to be symmetric: the
   * only honest way to retract "playback is blocked" is for playback to start.
   * Without it, `onBlocked` was a one-way door -- clicking "点击继续播放" started
   * the clip but left the panel sitting on top of it, because nothing told the
   * parent it had worked.
   */
  onPlaying?: (beatId: string) => void;
}

const SLOTS = [0, 1, 2] as const;

/**
 * Was this `play()` rejection the browser refusing us, or just a newer request
 * cancelling an older one?
 *
 * `HTMLMediaElement.play()` returns a promise that rejects for both, and the two
 * want opposite responses. `NotAllowedError` means audible playback was refused
 * for want of a user gesture -- the player needs to be told, because nothing will
 * happen until they click. `AbortError` means a `load()` or `pause()` landed after
 * the `play()` call and superseded it, which happens here as a matter of course:
 * `show` can be called again before the previous one's promise settles, and React
 * Strict Mode in development runs the effect that calls it twice on mount.
 *
 * Treating the second as the first is the whole bug: the clip was playing
 * perfectly and "点击继续播放" was on top of it, then vanished ten seconds later
 * when `onTimeUpdate` moved the mode to `deciding` near the end of the clip --
 * which is exactly the "goes away by itself after a while" that was reported.
 */
function isAutoplayRefusal(err: unknown): boolean {
  const name = err instanceof DOMException ? err.name : "";
  if (name === "AbortError") return false;
  // Anything else -- `NotAllowedError`, or something unforeseen -- counts, but the
  // callers below also check `el.paused` before reporting it. The overlay's one job
  // is to give a stalled player a way to start playback; if the clip is running,
  // there is nothing for it to offer and it must not appear.
  return true;
}

const VideoStage = forwardRef<StageHandle, Props>(function VideoStage(
  { onTimeUpdate, onEnded, onBlocked, onPlaying },
  ref
) {
  const els = useRef<(HTMLVideoElement | null)[]>([null, null, null]);
  // beat id -> slot, plus the reverse, so eviction can tell what it is throwing
  // away and never evicts the slot currently on screen.
  const slotOf = useRef<Map<string, number>>(new Map());
  const beatOf = useRef<(string | null)[]>([null, null, null]);
  const visible = useRef<string | null>(null);
  const useCount = useRef(0);
  const lastUsed = useRef<number[]>([0, 0, 0]);

  /**
   * Stop and release every element when the stage goes away.
   *
   * Taking a `<video>` out of the document does not stop it. A detached element
   * with a live `src` keeps decoding and keeps its audio on the output until it is
   * garbage collected, which is at the engine's convenience -- so leaving a
   * session left its soundtrack playing underneath the setup screen, and
   * underneath the *next* session if the player picked one, which is the second
   * half of the "old one is still playing" report. The first half was
   * `useSession` handing over a stale document; this is why it could still be
   * heard after that was fixed.
   *
   * `removeAttribute("src")` then `load()` is the documented way to make a media
   * element let go: pausing alone stops the sound but keeps the decoder and the
   * buffered data, and clearing `src` without `load()` is not guaranteed to
   * abort the fetch in progress.
   */
  useEffect(
    () => () => {
      for (const i of SLOTS) {
        const el = els.current[i];
        if (!el) continue;
        el.pause();
        el.removeAttribute("src");
        el.removeAttribute("data-src");
        el.load();
      }
    },
    []
  );

  const assign = (beatId: string, url: string): number => {
    const existing = slotOf.current.get(beatId);
    if (existing !== undefined) {
      lastUsed.current[existing] = ++useCount.current;
      return existing;
    }
    // Prefer an empty slot, then the least recently used one that is not on
    // screen. Evicting the visible slot would blank the picture mid-beat.
    const free = SLOTS.find((i) => beatOf.current[i] === null);
    const slot =
      free ??
      SLOTS.filter((i) => beatOf.current[i] !== visible.current).sort(
        (a, b) => lastUsed.current[a] - lastUsed.current[b]
      )[0];

    const evicted = beatOf.current[slot];
    if (evicted) slotOf.current.delete(evicted);
    beatOf.current[slot] = beatId;
    slotOf.current.set(beatId, slot);
    lastUsed.current[slot] = ++useCount.current;

    const el = els.current[slot];
    if (el && el.getAttribute("data-src") !== url) {
      el.setAttribute("data-src", url);
      el.src = url;
      // `preload="auto"` alone does not always start the fetch for a src set
      // after mount; load() makes it explicit.
      el.load();
    }
    return slot;
  };

  useImperativeHandle(
    ref,
    (): StageHandle => ({
      buffer: (beatId, url) => {
        assign(beatId, url);
      },
      show: async (beatId, url, fromStart = true) => {
        const slot = assign(beatId, url);
        const el = els.current[slot];
        if (!el) return;
        const previous = visible.current;
        visible.current = beatId;
        // Pause whatever was on screen only after the new one is visible, so the
        // compositor never has two frames' worth of nothing to show.
        for (const i of SLOTS) {
          const e = els.current[i];
          if (!e) continue;
          e.style.opacity = i === slot ? "1" : "0";
          e.style.zIndex = i === slot ? "2" : "1";
        }
        if (fromStart) el.currentTime = 0;
        let refused = false;
        try {
          await el.play();
        } catch (err) {
          refused = isAutoplayRefusal(err);
        }
        // Whatever became of the new clip, the old one has to stop. This used to be
        // behind an early `return` on the failure path, so a spurious rejection left
        // two clips running at once -- inaudible in the picture, obvious in the
        // sound.
        if (previous && previous !== beatId) {
          const prevSlot = slotOf.current.get(previous);
          if (prevSlot !== undefined) els.current[prevSlot]?.pause();
        }
        // Only report a block if this call is still the current one and the element
        // really is sitting still. A `show` that has been superseded says nothing:
        // the newer call owns the screen and will report for itself.
        if (refused && visible.current === beatId && el.paused) onBlocked?.(beatId);
      },
      visible: () => visible.current,
      position: () => {
        const id = visible.current;
        const slot = id ? slotOf.current.get(id) : undefined;
        const el = slot !== undefined ? els.current[slot] : null;
        return {
          time: el?.currentTime ?? 0,
          duration: Number.isFinite(el?.duration) ? (el?.duration as number) : 0,
        };
      },
      pause: () => {
        const id = visible.current;
        const slot = id ? slotOf.current.get(id) : undefined;
        if (slot !== undefined) els.current[slot]?.pause();
      },
      resume: async () => {
        const id = visible.current;
        const slot = id ? slotOf.current.get(id) : undefined;
        const el = slot !== undefined ? els.current[slot] : null;
        if (!el) return;
        try {
          await el.play();
        } catch (err) {
          if (id && isAutoplayRefusal(err) && visible.current === id && el.paused) {
            onBlocked?.(id);
          }
        }
      },
      setMuted: (muted) => {
        for (const i of SLOTS) {
          const e = els.current[i];
          if (e) e.muted = muted;
        }
      },
    })
  );

  return (
    <div className="stage">
      {SLOTS.map((i) => (
        <video
          key={i}
          ref={(el) => {
            // Keep the element on detach instead of storing the `null` React
            // passes. Passive effect cleanup runs *after* refs are detached, so
            // nulling here would leave the teardown above with nothing to stop --
            // and a detached-but-playing <video> is precisely what it exists to
            // silence. Nothing reads these after unmount, so holding a stale
            // reference for the length of a teardown costs nothing.
            if (el) els.current[i] = el;
          }}
          className="stage-video"
          style={{ opacity: 0, zIndex: 1 }}
          preload="auto"
          playsInline
          // No `controls`: a scrubber invites the player to seek, and seeking
          // breaks the illusion that this is one continuous film rather than a
          // chain of 14-second clips.
          onTimeUpdate={(ev) => {
            const id = beatOf.current[i];
            if (!id || id !== visible.current) return;
            const el = ev.currentTarget;
            onTimeUpdate?.(id, el.currentTime, Number.isFinite(el.duration) ? el.duration : 0);
          }}
          onPlaying={() => {
            const id = beatOf.current[i];
            // Buffered slots never play, but a slot that *was* visible can emit a
            // late `playing` as it is being paused; gate on the current beat so a
            // clip leaving the screen cannot speak for the one arriving.
            if (id && id === visible.current) onPlaying?.(id);
          }}
          onEnded={() => {
            const id = beatOf.current[i];
            if (id && id === visible.current) onEnded?.(id);
          }}
        />
      ))}
    </div>
  );
});

export default VideoStage;
