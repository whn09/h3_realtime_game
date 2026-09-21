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
 *
 * The other half of "never unmount" is that an element carries its *previous*
 * clip's picture into its next one. Assigning a new `src` does not clear what the
 * compositor holds: the element keeps painting the last frame it decoded until
 * the new media produces one. So revealing a slot the moment its `src` changes
 * shows the clip that used to live there -- the sibling branch, or the beat
 * before -- for the handful of frames until the first real frame decodes. That is
 * the reported "有的时候前几帧是其他画面", and it is a playback artefact only: the
 * mp4s themselves were checked frame by frame against the image each was
 * conditioned on (`bench/frame_continuity_check.py`) and every one starts where
 * it should. Hence `hasFrame` below, and hence `show` waiting on it.
 */

export interface StageHandle {
  /**
   * Point a slot at this clip and let it buffer. Idempotent per beat.
   *
   * `poster` is the clip's own first frame (`poster.jpg`, ~70KB against the clip's
   * 1.7MB). It is what the element paints until it has decoded a frame of its own,
   * and it is the answer to the black screen: before this, a slot revealed before
   * its first frame arrived had nothing at all to show.
   */
  buffer: (beatId: string, url: string, poster?: string | null) => void;
  /** Make this beat visible and play it from the start. Buffers first if needed. */
  show: (
    beatId: string,
    url: string,
    fromStart?: boolean,
    poster?: string | null
  ) => Promise<void>;
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
  /**
   * The element gave up on this clip.
   *
   * Nothing used to listen for `error`, and that was the one failure mode the
   * player could not see through: the fetch reads 200 in devtools, the decoder
   * rejects the media or the transfer breaks, the element goes quiet, and `show`
   * is left awaiting a `play()` promise that never settles. On screen that is a
   * frozen picture with no overlay, no toast, and nothing in the log -- "视频加载
   * 成功了，但是就是播不出来". The element knows exactly what went wrong; it just
   * had no way to say so.
   *
   * The string is diagnostic, not prose: the browser's own code and message plus
   * `readyState` / `networkState` / how many bytes it managed to buffer, which is
   * what distinguishes "this file is not decodable" from "the tunnel died at 40%".
   */
  onMediaError?: (beatId: string, detail: string) => void;
  /**
   * Playback is wanted and the picture is not advancing (`waiting`), or it is.
   *
   * Worth surfacing here specifically because the clip comes down an ssh tunnel
   * that has been measured at both 107KB/s and 670KB/s within a minute: at the
   * bad end a 1.7MB clip cannot arrive in real time, and "it is still loading"
   * and "it is broken" look identical to the player unless one of them says so.
   */
  onBuffering?: (beatId: string, buffering: boolean) => void;
}

const SLOTS = [0, 1, 2] as const;

/**
 * How long `show` will hold the outgoing picture waiting for the incoming clip to
 * have a frame, before revealing it anyway.
 *
 * Generous on purpose. What is on screen during the wait is *correct* -- the
 * previous clip's final frame, which for a continuous beat is pixel-identical to
 * the image the incoming clip was conditioned on, possibly with the freeze <img>
 * and 「世界在回应你」 over it -- so waiting costs nothing but a slightly later cut,
 * while not waiting shows the wrong shot. The clip is ~1.1MB and normally already
 * buffered, so this bound is only reached when the fetch is genuinely in trouble;
 * revealing then is the lesser evil, because a black rectangle is worse than a
 * stale frame that is about to be replaced (DESIGN.md section 7: 绝不黑屏).
 */
const REVEAL_WAIT_MS = 4000;

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

/** `MediaError.code` -> the constant's name, because the number alone means nothing. */
const MEDIA_ERR = ["", "ABORTED", "NETWORK", "DECODE", "SRC_NOT_SUPPORTED"] as const;

/** What the element will admit to, in one line. */
function mediaDetail(el: HTMLVideoElement): string {
  const err = el.error;
  const buffered = el.buffered.length ? el.buffered.end(el.buffered.length - 1) : 0;
  const code = err ? `${MEDIA_ERR[err.code] ?? err.code} (${err.code})` : "no MediaError";
  const msg = err?.message ? ` ${err.message}` : "";
  return (
    `${code}${msg} · readyState=${el.readyState} networkState=${el.networkState}` +
    ` buffered=${buffered.toFixed(1)}s/${Number.isFinite(el.duration) ? el.duration.toFixed(1) : "?"}s`
  );
}

const VideoStage = forwardRef<StageHandle, Props>(function VideoStage(
  { onTimeUpdate, onEnded, onBlocked, onPlaying, onMediaError, onBuffering },
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
   * Does the element in this slot hold a decoded frame of the clip *currently*
   * assigned to it? Cleared when a new `src` is set and set again on
   * `loadeddata`, which is the event that means readyState reached
   * HAVE_CURRENT_DATA -- i.e. there is a frame at the current position. The gap
   * between those two is the window in which the element still paints the
   * previous beat, which is the bug this exists to close.
   */
  const hasFrame = useRef<boolean[]>([false, false, false]);
  /** Resolvers for `show` calls parked on a slot's first frame. */
  const waiters = useRef<(() => void)[][]>([[], [], []]);
  /**
   * The id `show` is currently working towards. Distinct from `visible`, which
   * must keep naming the clip that is actually on screen until the swap really
   * happens -- otherwise the outgoing clip's `ended` and `timeupdate` are dropped
   * on the floor while we wait, and the beat that was playing never reports that
   * it finished.
   */
  const wanted = useRef<string | null>(null);

  const wake = (slot: number) => {
    const parked = waiters.current[slot];
    waiters.current[slot] = [];
    for (const resolve of parked) resolve();
  };

  const markFrame = (slot: number) => {
    hasFrame.current[slot] = true;
    wake(slot);
  };

  const awaitFrame = (slot: number): Promise<void> =>
    new Promise((resolve) => {
      if (hasFrame.current[slot]) return resolve();
      let settled = false;
      const done = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve();
      };
      const timer = setTimeout(done, REVEAL_WAIT_MS);
      waiters.current[slot].push(done);
    });

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

  const assign = (beatId: string, url: string, poster?: string | null): number => {
    const existing = slotOf.current.get(beatId);
    if (existing !== undefined) {
      lastUsed.current[existing] = ++useCount.current;
      return existing;
    }
    // Prefer an empty slot, then the least recently used one that is neither on
    // screen nor about to be. Evicting the visible slot would blank the picture
    // mid-beat; evicting `wanted` would be worse -- `show` is parked on that
    // slot's first frame, and it would wake up to a different clip and reveal it.
    // That matters because the two are briefly different: `show` now holds the
    // outgoing picture while the incoming clip decodes, and the player's children
    // start buffering inside that window.
    const protectedIds = new Set([visible.current, wanted.current].filter(Boolean));
    const free = SLOTS.find((i) => beatOf.current[i] === null);
    const byAge = (a: number, b: number) => lastUsed.current[a] - lastUsed.current[b];
    const slot =
      free ??
      SLOTS.filter((i) => !protectedIds.has(beatOf.current[i])).sort(byAge)[0] ??
      // Cannot happen at three slots and at most two protected beats, but a pool
      // that ever shrinks must still return something rather than `undefined`.
      SLOTS.filter((i) => beatOf.current[i] !== visible.current).sort(byAge)[0];

    const evicted = beatOf.current[slot];
    if (evicted) slotOf.current.delete(evicted);
    beatOf.current[slot] = beatId;
    slotOf.current.set(beatId, slot);
    lastUsed.current[slot] = ++useCount.current;

    const el = els.current[slot];
    if (el && el.getAttribute("data-src") !== url) {
      el.setAttribute("data-src", url);
      // Before the fetch, not after: from here until `loadeddata` this element is
      // still showing the clip it is being taken away from.
      hasFrame.current[slot] = false;
      // Poster before src, because `load()` is what makes the element paint it: an
      // element with no media data shows its poster, and from here until
      // `loadeddata` that is the only picture of *this* clip that exists. 70KB
      // against 1.7MB, so over the tunnel it lands roughly ten seconds sooner, and
      // it is frame 0 of exactly this shot -- not a neighbour's, not black.
      if (poster) el.poster = poster;
      else el.removeAttribute("poster");
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
      buffer: (beatId, url, poster) => {
        assign(beatId, url, poster);
      },
      show: async (beatId, url, fromStart = true, poster) => {
        const slot = assign(beatId, url, poster);
        const el = els.current[slot];
        if (!el) return;
        wanted.current = beatId;
        // Hold the outgoing picture until this element can actually paint *this*
        // clip. A beat that was buffered while the previous one played clears this
        // instantly; the ones that do not are exactly the ones that used to flash
        // the wrong shot -- a branch whose clip finished after the cursor had
        // already moved to it, or a jump from the story tree to a beat no slot was
        // holding.
        if (!hasFrame.current[slot]) {
          await awaitFrame(slot);
          // A newer `show` took over while we waited -- it owns the screen now, and
          // revealing this slot would put the beat the player just left back on it.
          if (wanted.current !== beatId) return;
        }
        const previous = visible.current;
        visible.current = beatId;
        // Rewind before revealing, not after. A slot that has already been watched
        // -- the story tree jumping back to a beat still in the pool -- is sitting
        // on its *last* frame, and showing it first and seeking second puts the end
        // of the shot on screen for the length of the seek.
        if (fromStart && el.currentTime !== 0) el.currentTime = 0;
        // Pause whatever was on screen only after the new one is visible, so the
        // compositor never has two frames' worth of nothing to show.
        for (const i of SLOTS) {
          const e = els.current[i];
          if (!e) continue;
          e.style.opacity = i === slot ? "1" : "0";
          e.style.zIndex = i === slot ? "2" : "1";
        }
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
          // `loadeddata` = readyState reached HAVE_CURRENT_DATA = there is a frame
          // of *this* src to paint, which is the precondition `show` waits on.
          onLoadedData={() => markFrame(i)}
          // And the other direction: `load()` empties the element. Belt and braces
          // with the explicit clear in `assign`, and it also covers the teardown
          // path, so the flag can never outlive the media it describes.
          onEmptied={() => {
            hasFrame.current[i] = false;
          }}
          onTimeUpdate={(ev) => {
            const id = beatOf.current[i];
            if (!id || id !== visible.current) return;
            const el = ev.currentTarget;
            // The clock moved, so whatever it was waiting for, it got it. This is
            // the clear that cannot lie: `canplay` is the decoder's opinion about
            // the future, `timeupdate` is the picture actually advancing.
            onBuffering?.(id, false);
            onTimeUpdate?.(id, el.currentTime, Number.isFinite(el.duration) ? el.duration : 0);
          }}
          onPlaying={() => {
            const id = beatOf.current[i];
            // Buffered slots never play, but a slot that *was* visible can emit a
            // late `playing` as it is being paused; gate on the current beat so a
            // clip leaving the screen cannot speak for the one arriving.
            if (id && id === visible.current) {
              onPlaying?.(id);
              onBuffering?.(id, false);
            }
          }}
          // A slot that has failed will never produce a frame, so anything parked
          // on one must be let go now rather than sitting out REVEAL_WAIT_MS for a
          // frame that is not coming. Revealing a failed element is still the right
          // move: it keeps painting the previous clip's picture, which beats black.
          onError={(ev) => {
            const el = ev.currentTarget;
            const id = beatOf.current[i];
            wake(i);
            const detail = mediaDetail(el);
            // Console too, not only the UI: the toast is one line and the player
            // may well have clicked it away before telling anyone.
            console.warn(`[stage] media error on ${id ?? `slot ${i}`}: ${detail}`, el.currentSrc);
            if (id) onMediaError?.(id, detail);
          }}
          // `waiting` is the only event that raises it: it means playback is wanted
          // and stopped for want of the next frame. `stalled` used to raise it too
          // and that was wrong -- `stalled` is about the *download* going quiet,
          // which Chrome also reports for a clip that is already playable, so the
          // notice stuck on screen over a clip that had finished downloading in a
          // few seconds. The element's state goes to the console at the same moment,
          // because "readyState=4, buffered 14.4s, still waiting" and "readyState=1,
          // buffered 0.6s" are the same picture and completely different bugs.
          onWaiting={(ev) => {
            const id = beatOf.current[i];
            if (!id || id !== visible.current) return;
            console.warn(`[stage] waiting on ${id}: ${mediaDetail(ev.currentTarget)}`);
            onBuffering?.(id, true);
          }}
          onCanPlay={() => {
            const id = beatOf.current[i];
            if (id && id === visible.current) onBuffering?.(id, false);
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
