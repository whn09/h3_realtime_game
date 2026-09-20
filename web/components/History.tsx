"use client";

import type { SessionSummary } from "@/lib/types";

/**
 * Past runs, newest first.
 *
 * Resuming is free and total: every beat of every branch is on disk, including
 * the ones the player never took, so an old session comes back with its whole
 * tree intact and the story tree is immediately walkable. Nothing is regenerated
 * to get back in -- the only GPU work a resumed session triggers is whatever the
 * frontier decides is missing *ahead* of where the player stopped.
 *
 * The rows come from the persisted `session.json`, not from live memory, which is
 * why a run survives an orchestrator restart and still appears here.
 */

interface Props {
  rows: SessionSummary[];
  onResume: (sid: string) => void;
}

const PHASE_LABEL: Record<string, string> = {
  creating: "还在开场",
  opening: "还在开场",
  playing: "进行中",
  ended: "已完结",
  failed: "中断了",
};

/**
 * Relative, because the absolute time is never what the player is asking. They
 * are asking "is this the one I was just playing", and "3 分钟前" answers that
 * while a timestamp makes them do arithmetic.
 */
function ago(unixSeconds: number): string {
  const s = Math.max(0, Date.now() / 1000 - unixSeconds);
  if (s < 60) return "刚刚";
  if (s < 3600) return `${Math.floor(s / 60)} 分钟前`;
  if (s < 86400) return `${Math.floor(s / 3600)} 小时前`;
  if (s < 86400 * 30) return `${Math.floor(s / 86400)} 天前`;
  return new Date(unixSeconds * 1000).toLocaleDateString("zh-CN");
}

export default function History({ rows, onResume }: Props) {
  return (
    <section className="history">
      <h2>
        回到之前的故事 <span className="dim">（分支全都留着，进去就能接着走）</span>
      </h2>
      <div className="history-list">
        {rows.map((r) => (
          <button key={r.id} className="hrow" onClick={() => onResume(r.id)}>
            {r.thumb_url ? (
              <img className="hthumb" src={r.thumb_url} alt="" loading="lazy" />
            ) : (
              <span className="hthumb hthumb-blank" />
            )}
            <span className="hbody">
              <span className="htitle">
                {/* The logline is what the Worldsmith wrote for *this* run; the
                    premise is what the player typed, which for a preset start is
                    identical across every run of that preset and so cannot tell
                    two of them apart. */}
                {r.logline || r.premise || "（无名的故事）"}
              </span>
              <span className="hmeta">
                {r.genre ? <span>{r.genre}</span> : null}
                <span>第 {r.act} 幕</span>
                {r.location ? <span>{r.location}</span> : null}
                <span>{r.path_length} 段</span>
                <span className={r.phase === "failed" ? "err" : ""}>
                  {PHASE_LABEL[r.phase] ?? r.phase}
                </span>
                <span className="dim">{ago(r.updated_at)}</span>
              </span>
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
