"use client";

import type { Beat, SessionView } from "@/lib/types";

/**
 * The state readout. Exists because the consequence system is invisible
 * otherwise: flags and stats change what the Director writes, and a player who
 * cannot see them reads the branching as arbitrary.
 */

interface Props {
  session: SessionView;
  shown: Beat | null;
  connected: boolean;
  muted: boolean;
  treeOpen: boolean;
  onToggleMute: () => void;
  onToggleTree: () => void;
  onExit: () => void;
}

export default function Hud({
  session,
  shown,
  connected,
  muted,
  treeOpen,
  onToggleMute,
  onToggleTree,
  onExit,
}: Props) {
  const state = shown?.state;
  return (
    <div className="hud">
      <div className="hud-left">
        {/* Leaving abandons nothing: the session is on disk and listed under
            history, so this is a way back to the shelf, not a quit. */}
        <button className="hud-title hud-home" onClick={onExit} title="回到开头（这局会留在历史里）">
          {session.bible?.genre || session.genre || "故事"}
        </button>
        {state ? (
          <>
            <span className="hud-chip">第 {state.act} 幕</span>
            <span className="hud-chip">{state.location}</span>
            <span className="hud-chip">{state.time_of_day}</span>
            <span className="hud-chip" title="张力">
              张力 {state.tension}/10
            </span>
            {state.inventory.length ? (
              <span className="hud-chip" title="随身">
                {state.inventory.slice(0, 3).join("·")}
              </span>
            ) : null}
            {Object.entries(state.stats)
              .slice(0, 3)
              .map(([k, v]) => (
                <span key={k} className="hud-chip">
                  {k} {v}
                </span>
              ))}
          </>
        ) : null}
      </div>
      <div className="hud-right">
        {shown?.degraded ? <span className="hud-warn">降级重试</span> : null}
        {shown?.transition === "continuous" ? (
          <span className="hud-chip hud-dim" title="这一拍从上一拍的末帧续上">
            接续
          </span>
        ) : null}
        <button className="hud-btn" onClick={onToggleMute}>
          {muted ? "取消静音" : "静音"}
        </button>
        <button className={treeOpen ? "hud-btn hud-btn-on" : "hud-btn"} onClick={onToggleTree}>
          故事树
        </button>
        <span
          className={connected ? "hud-live" : "hud-live hud-live-off"}
          title={connected ? "事件流已连接" : "事件流断开，正在重连"}
        />
      </div>
    </div>
  );
}
