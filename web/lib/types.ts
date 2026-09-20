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
  duration_ms: number | null;
  has_audio: boolean;
  narration: string;
  transition: Transition;
  shot_type: ShotType;
  ir_source: string;
  options: OptionSummary[];
  /** option index, as a string key -- it survives a JSON round trip that way. */
  children: Record<string, string>;
  predicted_choice: number;
  is_ending: Ending | null;
  drift: Record<string, number> | null;
  timings: Record<string, number>;
  state: StateSummary;
}

export interface WorldBible {
  logline: string;
  genre: string;
  style_anchor: string;
  music_bible: string;
  characters: { name: string; appearance: string; role?: string }[];
  outline: { act: number; goal: string; turn: string }[];
  [key: string]: unknown;
}

export interface SessionView {
  id: string;
  phase: SessionPhase;
  error: string | null;
  premise: string;
  genre: string;
  pov: "first" | "third";
  bible: WorldBible | null;
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
