"use client";

import { useState } from "react";
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
  /** Erase a run for good. Resolves once the orchestrator has removed it; the
   *  caller drops the row. Rejecting leaves the row where it is and the message
   *  is shown in place. */
  onDelete: (sid: string) => Promise<void>;
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

export default function History({ rows, onResume, onDelete }: Props) {
  /**
   * Which row's delete is armed, if any. One id rather than a set: arming a
   * second row disarms the first, so there is never more than one live "confirm"
   * on screen and a mis-aimed second click cannot land on a button that was
   * already waiting for one.
   *
   * Two steps rather than `confirm()`, and rather than one click. One click is
   * wrong because this is irreversible and the target is a small control sitting
   * next to the much larger "resume" surface -- a slip costs the run. A modal is
   * wrong because the list is where the decision is made: the player is looking
   * at the logline and the thumbnail that tell them whether this is the run they
   * want gone, and a dialog covers exactly that.
   */
  const [armed, setArmed] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [failed, setFailed] = useState<Record<string, string>>({});

  const remove = async (sid: string) => {
    setArmed(null);
    setBusy(sid);
    setFailed((f) => {
      const { [sid]: _drop, ...rest } = f;
      return rest;
    });
    try {
      await onDelete(sid);
      // No state reset on success: the row is gone from `rows`, so this component
      // stops rendering it and `busy` is never read again.
    } catch (e) {
      setBusy(null);
      setFailed((f) => ({ ...f, [sid]: e instanceof Error ? e.message : String(e) }));
    }
  };

  return (
    <section className="history">
      <h2>
        回到之前的故事 <span className="dim">（分支全都留着，进去就能接着走）</span>
      </h2>
      <div className="history-list">
        {rows.map((r) => (
          <div key={r.id} className="hrow-wrap">
            {/* The row and the delete control are siblings, not nested: a button
                inside a button is invalid, and more practically the whole row is
                the resume target, so a delete inside it would be a click the
                parent also hears. */}
            <button
              className="hrow"
              onClick={() => onResume(r.id)}
              disabled={busy === r.id}
            >
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
                  {failed[r.id] ? (
                    <span className="err">删不掉：{failed[r.id]}</span>
                  ) : null}
                </span>
              </span>
            </button>

            {busy === r.id ? (
              <span className="hdel-busy dim">
                <span className="spinner" /> 正在删
              </span>
            ) : armed === r.id ? (
              <span className="hdel-confirm">
                {/* Says what goes, because the row does not show it: the disk cost
                    of a run is its clips, and "3 段影像" is the part a player might
                    not want to lose. */}
                <span className="dim">连 {r.path_length} 段影像一起删？</span>
                <button className="hdel-yes" onClick={() => void remove(r.id)}>
                  删除
                </button>
                <button className="hdel-no" onClick={() => setArmed(null)}>
                  取消
                </button>
              </span>
            ) : (
              <button
                className="hdel"
                title="删除这一局"
                aria-label="删除这一局"
                onClick={() => setArmed(r.id)}
              >
                ×
              </button>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
