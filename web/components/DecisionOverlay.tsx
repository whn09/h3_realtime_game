"use client";

import { useEffect, useRef, useState } from "react";
import type { Beat, BeatStatus } from "@/lib/types";

const SHOT_LABEL: Record<string, string> = {
  wide: "远景",
  medium: "中景",
  closeup: "特写",
  pov: "主观",
  tracking: "跟拍",
  aerial: "俯瞰",
};

const TRANSITION_LABEL: Record<string, string> = {
  continuous: "接续",
  cut: "硬切",
  timeskip: "跳时",
};

const READINESS: Record<BeatStatus, string> = {
  pending: "排队中",
  compiling: "编译提示词",
  waiting_frame: "等待末帧",
  queued: "等待 GPU",
  generating: "生成中",
  ready: "已就绪",
  failed: "失败",
};

interface Props {
  beat: Beat;
  /**
   * Auto-advance progress: 0 -> the clip just ended, 1 -> expired. `null` when
   * nothing is counting, which is the default -- and then no ring is drawn at
   * all, because a ring that is not counting anything still reads as a deadline.
   */
  countdown: number | null;
  /** The clip has ended and the picture is holding on its last frame. */
  held: boolean;
  /** Status of each option's pre-generated child, by option index. */
  childStatus: (index: number) => BeatStatus | null;
  chosen: number | null;
  busy: boolean;
  onChoose: (index: number) => void;
  onCustom: (action: string) => void;
}

export default function DecisionOverlay({
  beat,
  countdown,
  held,
  childStatus,
  chosen,
  busy,
  onChoose,
  onCustom,
}: Props) {
  const [custom, setCustom] = useState("");
  const [customOpen, setCustomOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (customOpen) inputRef.current?.focus();
  }, [customOpen]);

  const R = 26;
  const C = 2 * Math.PI * R;
  const remaining = countdown === null ? 1 : Math.max(0, 1 - countdown);

  return (
    <div className="decision">
      <div className="decision-top">
        {beat.narration ? <p className="narration">{beat.narration}</p> : null}
        {countdown !== null ? (
          <div className="ring" aria-hidden>
            <svg viewBox="0 0 64 64" width="64" height="64">
              <circle className="ring-track" cx="32" cy="32" r={R} />
              <circle
                className="ring-arc"
                cx="32"
                cy="32"
                r={R}
                strokeDasharray={C}
                strokeDashoffset={C * (1 - remaining)}
              />
            </svg>
          </div>
        ) : held ? (
          // Says the quiet part out loud. Without it, a picture that has stopped
          // moving with no timer anywhere is ambiguous between "waiting for you"
          // and "stuck", and those call for opposite reactions from the player.
          <p className="held-note">画面在等你</p>
        ) : null}
      </div>

      <div className="cards">
        {beat.options.map((opt, i) => {
          const status = childStatus(i);
          const ready = status === "ready";
          const failed = status === "failed";
          const predicted = i === beat.predicted_choice;
          return (
            <button
              key={i}
              className={[
                "card",
                chosen === i ? "card-chosen" : "",
                predicted ? "card-predicted" : "",
                failed ? "card-failed" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              // Deliberately clickable even when the child is not ready. The
              // player's decisiveness should never be punished by our scheduler:
              // choosing early is recorded and the swap happens the moment the
              // clip exists, with the freeze-frame covering the gap.
              disabled={busy}
              onClick={() => onChoose(i)}
            >
              <span className="card-label">{opt.label}</span>
              {opt.consequence_hint ? (
                <span className="card-hint">{opt.consequence_hint}</span>
              ) : null}
              <span className="card-meta">
                <span>{SHOT_LABEL[opt.shot_type] ?? opt.shot_type}</span>
                <span>{TRANSITION_LABEL[opt.transition] ?? opt.transition}</span>
                <span className={ready ? "dot dot-ready" : failed ? "dot dot-bad" : "dot"}>
                  {status ? READINESS[status] : "未生成"}
                </span>
              </span>
            </button>
          );
        })}
      </div>

      <div className="custom">
        {customOpen ? (
          <form
            onSubmit={(ev) => {
              ev.preventDefault();
              const action = custom.trim();
              if (action) onCustom(action);
            }}
          >
            <input
              ref={inputRef}
              value={custom}
              maxLength={200}
              placeholder="或者……你想做什么？（这条路没有预生成，要等一会儿）"
              onChange={(ev) => setCustom(ev.target.value)}
              disabled={busy}
            />
            <button type="submit" disabled={busy || !custom.trim()}>
              去
            </button>
          </form>
        ) : (
          <button className="custom-open" onClick={() => setCustomOpen(true)} disabled={busy}>
            或者……（自由行动）
          </button>
        )}
      </div>
    </div>
  );
}
