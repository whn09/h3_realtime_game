/**
 * Mirrors the orchestrator's wire shapes: `session_view`, `_beat_summary`,
 * `_option_summary`, `_state_summary` in services/orchestrator/app/engine.py.
 *
 * Kept hand-written rather than generated. The surface is small and stable, and
 * a hand-written mirror is where the *browser's* reading of a field belongs --
 * `last_frame_url` being pixel-identical to the clip's final frame is the fact
 * the whole decision overlay is built on, and that belongs in a comment here
 * rather than in a generator's output.
 */

export type BeatStatus =
  | "pending"
  | "compiling"
  | "waiting_frame"
  | "queued"
  | "generating"
  | "ready"
  | "failed";

export type SessionPhase = "creating" | "opening" | "playing" | "ended" | "failed";

export type Transition = "continuous" | "cut" | "timeskip";

export type ShotType = "wide" | "medium" | "closeup" | "pov" | "tracking" | "aerial";

export interface OptionSummary {
  label: string;
  consequence_hint: string;
  transition: Transition;
  shot_type: ShotType;
  references_state: string;
}

export interface StateSummary {
  beat_index: number;
  act: number;
  location: string;
  time_of_day: string;
  elapsed_in_world: string;
  inventory: string[];
  stats: Record<string, number>;
  flags: Record<string, boolean>;
  tension: number;
  present_characters: string[];
}

export interface Ending {
  kind: string;
  title: string;
  epilogue: string;
}

export interface Beat {
  id: string;
  parent_id: string | null;
  index: number;
  origin: string;
  label: string;
  status: BeatStatus;
  error: string | null;
  degraded: boolean;
  video_url: string | null;
  poster_url: string | null;
  /**
   * The clip's final decoded frame, extracted server-side. Verified bit-exact
   * against the frame the decoder leaves on screen at `ended`, which is what
   * makes freezing on this `<img>` invisible rather than a visible jump.
   */
  last_frame_url: string | null;
  keyframe_url: string | null;
  /**
   * The image prompt that produced `keyframe_url`, or `""` on a beat that
   * inherited its first frame from its parent -- which is most of them. Empty is
   * therefore information, not a gap: it means nothing re-drew the cast here.
   */
  keyframe_prompt: string;
  duration_ms: number | null;
  has_audio: boolean;
  narration: string;
  transition: Transition;
  shot_type: ShotType;
  ir_source: string;
  /** `ir_validator` rules this beat's IR broke. Empty on the healthy path. */
  ir_violations: string[];
  options: OptionSummary[];
  /** option index, as a string key -- it survives a JSON round trip that way. */
  children: Record<string, string>;
  predicted_choice: number;
  is_ending: Ending | null;
  drift: Record<string, number> | null;
  timings: Record<string, number>;
  state: StateSummary;
}

/**
 * A character as the bible freezes them. `appearance` is reused *verbatim* in
 * every IR that features them and `appearance_en` is what re-draws them at a
 * cut, so the debug panel shows both in full rather than truncated: the whole
 * point of reading them is to check they are specific enough to reproduce.
 */
export interface BibleCharacter {
  id: string;
  name: string;
  appearance: string;
  appearance_en: string;
  voice: string;
  arc: string;
}

/**
 * The frozen world bible, exactly as `WorldBible.model_dump()` emits it. Written
 * out in full rather than left as an index signature: this shape was wrong in
 * three places (`characters[].role`, `outline[].goal`, `outline[].turn` do not
 * exist server-side; `appearance_en` and `style_anchor_en` were missing) and the
 * index signature is what let it stay wrong, because every misspelling typed as
 * `unknown` instead of failing.
 */
export interface WorldBible {
  premise: string;
  genre: string;
  logline: string;
  style_anchor: string;
  /** English, for the image model only. `""` when the Worldsmith skipped it. */
  style_anchor_en: string;
  music_bible: string;
  ambience: string;
  pov: "first" | "third";
  protagonist_id: string;
  characters: BibleCharacter[];
  world_rules: string[];
  outline: { act: number; milestone: string; target_beats: number }[];
  stat_names: string[];
  opening: {
    type: ShotType;
    subject: string;
    action: string;
    setting: string;
    mood: string;
    dialogue: { speaker: string; line: string }[];
    sfx_focus: string;
  } | null;
  opening_keyframe_prompt: string;
}

/** `GET /sessions/{sid}/beats/{beat_id}/ir` -- the exact string H3 was given. */
export interface IrView {
  prompt: string;
  meta: {
    source?: string;
    attempts?: number;
    violations?: string[];
    intent?: Record<string, unknown>;
  };
}

export interface SessionView {
  id: string;
  phase: SessionPhase;
  error: string | null;
  premise: string;
  genre: string;
  pov: "first" | "third";
  bible: WorldBible | null;
  /**
   * What the server had to invent because the Worldsmith left it out -- default
   * stats, a missing `appearance_en`. Each line is a reason this run behaves
   * unlike its premise, so they are the first thing to read when it does.
   */
  bible_notes: string[];
  /** Session-scoped latencies. Only the Worldsmith lands here; beats have their own. */
  timings: Record<string, number>;
  opening_keyframe_url: string | null;
  root_id: string | null;
  cursor: string | null;
  path: string[];
  beats: Record<string, Beat>;
  created_at: number;
  updated_at: number;
}

export interface Preset {
  id: string;
  title: string;
  premise: string;
  genre: string;
}

export interface PresetsResponse {
  presets: Preset[];
  beat_seconds: number;
  branch_count: number;
}

/**
 * One row of `GET /sessions`. Mirrors `Store.list_sessions`, which reads the
 * persisted `session.json` rather than live memory -- so a run survives an
 * orchestrator restart and is still listed here.
 *
 * `path_length` and `beats` are both counts and they are not the same thing:
 * `beats` includes every un-taken branch that was pre-generated, so it runs
 * roughly 3x ahead of the story the player actually saw.
 */
export interface SessionSummary {
  id: string;
  phase: SessionPhase;
  premise: string;
  genre: string;
  logline: string;
  thumb_url: string | null;
  beats: number;
  path_length: number;
  act: number;
  location: string;
  ended: boolean;
  /** Unix seconds, from the server clock. */
  updated_at: number;
}

/**
 * SSE frames. Every event carries `seq` and `type`; the rest is per-type. Typed
 * as a discriminated union only for the events the UI actually branches on --
 * the others are still delivered and logged, which is what the debug rail shows.
 */
export type GameEvent = { seq: number; type: string; ts?: number } & Record<string, unknown>;

export function isBeatEvent(
  e: GameEvent
): e is GameEvent & { beat: Beat } {
  return typeof e.beat === "object" && e.beat !== null;
}
