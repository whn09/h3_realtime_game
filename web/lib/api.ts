import type { PresetsResponse, SessionSummary, SessionView } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://127.0.0.1:8100";

/**
 * How long before the clip ends the options appear. Not a time limit -- picture
 * and sound keep running underneath, so this only buys the player a head start
 * on reading the two cards.
 */
export const DECISION_LEAD_S = Number(process.env.NEXT_PUBLIC_DECISION_LEAD ?? 4);

/**
 * Seconds to wait *after* the clip ends before committing the Director's
 * predicted choice on the player's behalf. `0` disables it: the last frame holds
 * indefinitely and nothing is decided until the player decides.
 *
 * Off by default, and that is a correction rather than a preference. Before, the
 * only window was `DECISION_LEAD_S` -- the last 4s of the clip -- so a player who
 * was still reading got a choice made for them. Auto-advance suits a demo you
 * want to run unattended, not someone actually playing, and the cost of holding
 * is nothing: both branches are already generated, so the GPUs are not waiting.
 */
export const AUTO_ADVANCE_S = Number(process.env.NEXT_PUBLIC_AUTO_ADVANCE ?? 0);

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    // FastAPI puts the message in `detail`; fall back to the status line so a
    // proxy error (which has no JSON body at all) still says something useful.
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

export const getPresets = () => call<PresetsResponse>("/presets");

export const listSessions = () =>
  call<{ sessions: SessionSummary[] }>("/sessions?limit=24");

export const getSession = (sid: string) => call<SessionView>(`/sessions/${sid}`);

export const createSession = (body: {
  premise?: string;
  genre?: string;
  pov?: "first" | "third";
  preset_id?: string;
}) => call<SessionView>("/sessions", { method: "POST", body: JSON.stringify(body) });

/**
 * Returns as soon as the child beat exists, not when its video does. The clip
 * arrives on the event stream -- often already finished, because both children
 * were pre-generated while the player was still watching the parent.
 */
export const choose = (
  sid: string,
  body: { option_index?: number; custom_action?: string }
) =>
  call<{ beat_id: string; status: string }>(`/sessions/${sid}/choose`, {
    method: "POST",
    body: JSON.stringify(body),
  });

export const seek = (sid: string, beatId: string) =>
  call<{ beat_id: string; status: string }>(`/sessions/${sid}/seek`, {
    method: "POST",
    body: JSON.stringify({ beat_id: beatId }),
  });

export const getIr = (sid: string, beatId: string) =>
  call<{ prompt: string; meta: Record<string, unknown> }>(
    `/sessions/${sid}/beats/${beatId}/ir`
  );

export const eventsUrl = (sid: string, after: number) =>
  `${API_BASE}/sessions/${sid}/events?after=${after}`;
