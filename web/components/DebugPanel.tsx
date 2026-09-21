"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { getIr } from "@/lib/api";
import type { Beat, GameEvent, IrView, SessionView } from "@/lib/types";

/**
 * The instrument panel. Five things that are otherwise invisible from a browser:
 *
 *   * **时间** -- where the seconds went. Every stage already measures itself into
 *     `timings`, but the numbers only existed in the JSON, so "生成慢" could not
 *     be attributed to the Director, to PromptIR, to the GPU, or to the network.
 *   * **这一拍** -- the exact string H3 was given, fetched from the on-disk
 *     archive, plus the keyframe prompt when this beat re-drew its own cast.
 *     These two are the answer to every "为什么画面是这样" question.
 *   * **世界观** -- the frozen bible, including the fields nothing else renders:
 *     the style anchor that is appended to every IR verbatim, and each
 *     character's appearance, which is the whole of the consistency mechanism.
 *   * **播放器** -- the three `<video>` elements as the browser sees them. The one
 *     tab that is not about the pipeline at all: everything above answers "what did
 *     we generate", this one answers "why is that not on screen".
 *   * **事件** -- the SSE log, newest first.
 *
 * Read-only by construction. It fetches one endpoint that the player's own path
 * never touches (`/ir`) and otherwise renders state that was already in memory,
 * so opening it cannot change what the run does -- which is the only way a debug
 * view is worth trusting when the thing being debugged is timing.
 */

interface Props {
  sid: string;
  session: SessionView;
  shown: Beat | null;
  events: GameEvent[];
  onClose: () => void;
}

type Tab = "time" | "beat" | "player" | "bible" | "events";

const TABS: { id: Tab; label: string }[] = [
  { id: "time", label: "时间" },
  { id: "beat", label: "这一拍" },
  { id: "player", label: "播放器" },
  { id: "bible", label: "世界观" },
  { id: "events", label: "事件" },
];

function fmtMs(ms: number): string {
  if (!Number.isFinite(ms)) return "—";
  if (ms >= 10_000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${Math.round(ms)}ms`;
}

/**
 * The stage breakdown, in pipeline order. Lifted out of the beat's `timings` by
 * key, and every key *not* in this list is still printed below it raw -- a
 * measurement added server-side shows up without anyone remembering to edit this
 * file, because the whole failure mode of a hand-curated debug view is silently
 * not showing the number you just added.
 *
 * `total` marks the two roll-ups that contain other rows: they get no bar,
 * because drawing them next to their own components reads as double counting.
 */
const STAGES: { key: string; label: string; hint?: string; total?: boolean }[] = [
  { key: "director_total_ms", label: "导演出岔路", hint: "Haiku 4.5，写两个分支" },
  { key: "promptir_wall_ms", label: "提示词编译", hint: "PromptIR，含校验与重修" },
  // Normally only beat 0 has this. A drawn image shares no pixels with what the
  // player is looking at, so every other beat starts from the previous clip's last
  // frame instead -- `MIDSTORY_KEYFRAMES=1` brings the old behaviour back.
  { key: "keyframe_ms", label: "关键帧", hint: "SD3.5，通常只有开场那一拍才有" },
  { key: "gpu_upload_ms", label: "上传首帧", hint: "H3_TRANSPORT=http 时恒为 0：H3 自己来拉" },
  { key: "gpu_server_ms", label: "H3 生成", hint: "SGLang 服务端自己报的耗时" },
  { key: "gpu_sglang_ms", label: "H3 提交+轮询", hint: "HTTP 往返，含排队" },
  { key: "gpu_download_ms", label: "下载成片", hint: "取回 mp4，内网 HTTP 约 6ms" },
  // `gpu_onbox_ms` is the submit-and-poll wall clock, which is `gpu_server_ms`
  // plus queueing on the box. It is not the last-frame extraction, whatever an
  // earlier label here said.
  { key: "gpu_onbox_ms", label: "GPU 机上墙钟", hint: "提交到完成，比服务端耗时多出的是排队" },
  { key: "gpu_postprocess_ms", label: "本地后处理", hint: "ffmpeg 抽帧/探测" },
  { key: "gpu_total_ms", label: "GPU 段合计", total: true },
  { key: "beat_wall_ms", label: "这一拍总墙钟", total: true },
  {
    key: "prepared_ahead_ms",
    label: "其中提前做掉",
    hint: "两次选择之前就编译好的 IR / 画好的关键帧，不占这一拍的墙钟",
    total: true,
  },
];

const KNOWN = new Set(STAGES.map((s) => s.key));

function Bars({ timings }: { timings: Record<string, number> }) {
  const rows = STAGES.filter((s) => timings[s.key] !== undefined);
  // Scale against the largest *component*, not against the roll-up: if the total
  // sets the scale then every real stage is a sliver and the chart says nothing.
  const scale = Math.max(
    1,
    ...rows.filter((r) => !r.total).map((r) => timings[r.key] ?? 0)
  );
  // Everything the server measured that this file does not know about.
  const extra = Object.entries(timings).filter(([k]) => !KNOWN.has(k));

  return (
    <>
      <div className="dbg-bars">
        {rows.map((r) => (
          <div key={r.key} className={r.total ? "dbg-bar dbg-bar-total" : "dbg-bar"}>
            <span className="dbg-bar-label" title={r.hint ?? r.key}>
              {r.label}
            </span>
            <span className="dbg-bar-track">
              {r.total ? null : (
                <i style={{ width: `${Math.min(100, ((timings[r.key] ?? 0) / scale) * 100)}%` }} />
              )}
            </span>
            <span className="dbg-bar-value">{fmtMs(timings[r.key] ?? 0)}</span>
          </div>
        ))}
        {rows.length === 0 ? <p className="dbg-empty">这一拍还没有计时数据。</p> : null}
      </div>
      {extra.length ? (
        <div className="dbg-kv dbg-kv-tight">
          {extra.map(([k, v]) => (
            <div key={k}>
              <b>{k}</b>
              <span>{k.endsWith("_ms") ? fmtMs(v) : v}</span>
            </div>
          ))}
        </div>
      ) : null}
    </>
  );
}

function median(xs: number[]): number {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * Median rather than mean, across every beat that finished.
 *
 * One cold start, one failed attempt that was retried, or one beat that queued
 * behind another moves a mean of six samples by seconds; the median answers the
 * question actually being asked, which is "what does a beat normally cost".
 * min/max are printed next to it so a fluke is still visible rather than hidden
 * by the statistic that ignores it.
 */
function Aggregate({ beats }: { beats: Beat[] }) {
  const rows = useMemo(() => {
    const keys = ["beat_wall_ms", "gpu_server_ms", "gpu_total_ms", "director_total_ms", "promptir_wall_ms", "keyframe_ms"];
    return keys
      .map((key) => {
        const xs = beats.map((b) => b.timings[key]).filter((v): v is number => typeof v === "number");
        const label = STAGES.find((s) => s.key === key)?.label ?? key;
        return { key, label, n: xs.length, med: median(xs), lo: Math.min(...xs), hi: Math.max(...xs) };
      })
      .filter((r) => r.n > 0);
  }, [beats]);

  if (!rows.length) return null;
  return (
    <table className="dbg-table">
      <thead>
        <tr>
          <th>阶段</th>
          <th>样本</th>
          <th>中位</th>
          <th>最快</th>
          <th>最慢</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.key}>
            <td>{r.label}</td>
            <td>{r.n}</td>
            <td>{fmtMs(r.med)}</td>
            <td>{fmtMs(r.lo)}</td>
            <td>{fmtMs(r.hi)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** The two media state machines, by value -- the numbers on their own are unreadable. */
const READY_STATE = ["NOTHING", "METADATA", "CURRENT", "FUTURE", "ENOUGH"] as const;
const NETWORK_STATE = ["EMPTY", "IDLE", "LOADING", "NO_SOURCE"] as const;
const MEDIA_ERR = ["", "ABORTED", "NETWORK", "DECODE", "SRC_UNSUPPORTED"] as const;

interface SlotRow {
  slot: number;
  beat: string;
  visible: boolean;
  ready: string;
  network: string;
  buffered: string;
  clock: string;
  note: string;
}

/**
 * The three `<video>` elements exactly as the browser sees them, polled.
 *
 * This tab exists because of a debugging session that should have taken a minute
 * and took an afternoon: a clip would not play, devtools showed the fetch
 * succeeding, and every explanation -- dead tunnel, undecodable file, refused
 * autoplay, a reveal that raced the first frame -- looks identical from outside the
 * element. The element itself knows which one it is; `readyState` and
 * `networkState` say so in two words. `IDLE` with `NOTHING` means nothing was ever
 * fetched; `LOADING` with `NOTHING` means it is on the way; `ENOUGH` while paused
 * means the data is all here and the problem is `play()`.
 *
 * Read from the DOM rather than through a handle on purpose: a probe that shares no
 * state with `VideoStage` cannot be fooled by `VideoStage`'s own bookkeeping being
 * wrong, which is precisely the thing one wants to check.
 *
 * Also worth knowing while reading it: a fresh load of a mid-story session assigns
 * **three** clips, not one -- the beat on screen plus both branches, prefetched so
 * the decision moment has something to cut to. Three mp4 fetches in the network tab
 * is the pool working. Fewer usually means the browser served one from cache (no row
 * at all, until a hard reload) or has not started it yet.
 */
function StageProbe() {
  const [rows, setRows] = useState<SlotRow[]>([]);

  useEffect(() => {
    const read = () => {
      const els = Array.from(document.querySelectorAll<HTMLVideoElement>("video.stage-video"));
      setRows(
        els.map((el, slot) => {
          const src = el.currentSrc || el.getAttribute("data-src") || "";
          // .../_h3/<sid>-<beat>/beat.mp4 -> <beat>
          const dir = src.split("/").at(-2) ?? "";
          const buffered = el.buffered.length ? el.buffered.end(el.buffered.length - 1) : 0;
          const err = el.error;
          return {
            slot,
            beat: dir ? dir.replace(/^\d+-[0-9a-z]+-/, "") : "—",
            visible: el.style.opacity === "1",
            ready: `${READY_STATE[el.readyState] ?? el.readyState}`,
            network: `${NETWORK_STATE[el.networkState] ?? el.networkState}`,
            buffered: `${buffered.toFixed(1)}s`,
            clock: `${el.currentTime.toFixed(1)}s / ${
              Number.isFinite(el.duration) ? el.duration.toFixed(1) : "?"
            }s`,
            note: err
              ? `${MEDIA_ERR[err.code] ?? err.code}${err.message ? `: ${err.message}` : ""}`
              : el.paused
                ? "paused"
                : "playing",
          };
        })
      );
    };
    read();
    // 400ms: fast enough to watch a load progress, slow enough that the panel is
    // not itself a load on the thing it is measuring.
    const timer = setInterval(read, 400);
    return () => clearInterval(timer);
  }, []);

  if (rows.length === 0) return <p className="dbg-empty">舞台还没有挂载。</p>;
  return (
    <table className="dbg-table">
      <thead>
        <tr>
          <th>slot</th>
          <th>beat</th>
          <th>readyState</th>
          <th>networkState</th>
          <th>已缓冲</th>
          <th>进度</th>
          <th>状态</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.slot}>
            <td>
              {r.slot}
              {r.visible ? " ●" : ""}
            </td>
            <td>
              <code>{r.beat}</code>
            </td>
            <td>{r.ready}</td>
            <td>{r.network}</td>
            <td>{r.buffered}</td>
            <td>{r.clock}</td>
            <td>{r.note}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function DebugPanel({ sid, session, shown, events, onClose }: Props) {
  const [tab, setTab] = useState<Tab>("time");
  const [ir, setIr] = useState<IrView | null>(null);
  const [irError, setIrError] = useState<string | null>(null);
  // Keyed by beat, so flipping tabs or re-rendering on an event does not refetch
  // a string that cannot change once written.
  const cache = useRef<Map<string, IrView>>(new Map());

  const beatId = shown?.id ?? null;

  useEffect(() => {
    if (tab !== "beat" || !beatId) return;
    const hit = cache.current.get(beatId);
    if (hit) {
      setIr(hit);
      setIrError(null);
      return;
    }
    let live = true;
    setIr(null);
    setIrError(null);
    getIr(sid, beatId)
      .then((v) => {
        cache.current.set(beatId, v);
        if (live) setIr(v);
      })
      .catch((e: unknown) => {
        if (live) setIrError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      live = false;
    };
  }, [sid, beatId, tab]);

  const readyBeats = useMemo(
    () => Object.values(session.beats).filter((b) => b.status === "ready"),
    [session.beats]
  );
  const bible = session.bible;
  /**
   * Both fields are new, and the orchestrator and this bundle are restarted
   * separately -- so a page held open across a deploy, or a browser cache held
   * across one, can be handed a document without them. The types say they are
   * always there and the server says so too; these two fallbacks are about the
   * minutes in between, where the alternative is a blank panel thrown by the one
   * view you opened to find out what is wrong.
   */
  const notes = session.bible_notes ?? [];
  const sessionTimings = session.timings ?? {};

  return (
    <aside className="dbg" aria-label="调试面板">
      <header className="dbg-head">
        <nav>
          {TABS.map((t) => (
            <button
              key={t.id}
              className={tab === t.id ? "dbg-tab dbg-tab-on" : "dbg-tab"}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <button className="dbg-close" onClick={onClose} title="关闭（D）">
          ✕
        </button>
      </header>

      <div className="dbg-body">
        {tab === "time" ? (
          <>
            <h4>开场（每局一次）</h4>
            {sessionTimings.worldsmith_total_ms !== undefined ? (
              <div className="dbg-kv">
                <div>
                  <b>世界圣经</b>
                  <span>{fmtMs(sessionTimings.worldsmith_total_ms)}</span>
                </div>
                <div>
                  <b>首 token</b>
                  <span>{fmtMs(sessionTimings.worldsmith_ttft_ms ?? NaN)}</span>
                </div>
                <div>
                  <b>输出 token</b>
                  <span>{sessionTimings.worldsmith_out_tokens ?? "—"}</span>
                </div>
                <div>
                  <b>调用次数</b>
                  <span title="大于 1 说明第一次返回被截断或校验失败，重跑了一次">
                    {sessionTimings.worldsmith_attempts ?? 1}
                  </span>
                </div>
              </div>
            ) : (
              <p className="dbg-empty">还没有世界圣经的计时（旧的存档不会有）。</p>
            )}

            <h4>
              这一拍{shown ? <code>{shown.id}</code> : null}
            </h4>
            {shown ? <Bars timings={shown.timings} /> : <p className="dbg-empty">还没有在播的片段。</p>}

            <h4>全局（{readyBeats.length} 个已完成的片段）</h4>
            <Aggregate beats={readyBeats} />
          </>
        ) : null}

        {tab === "beat" ? (
          shown ? (
            <>
              <div className="dbg-kv">
                <div>
                  <b>beat</b>
                  <span>
                    <code>{shown.id}</code> · 第 {shown.index} 拍
                  </span>
                </div>
                <div>
                  <b>状态</b>
                  <span>
                    {shown.status}
                    {shown.degraded ? "（降级路径）" : ""}
                  </span>
                </div>
                <div>
                  <b>转场 / 机位</b>
                  <span>
                    {shown.transition} · {shown.shot_type}
                  </span>
                </div>
                <div>
                  <b>IR 来源</b>
                  <span title="llm=模型直出；repaired=校验失败后重修；template=完全兜底">
                    {shown.ir_source || "—"}
                  </span>
                </div>
                <div>
                  <b>时长 / 声音</b>
                  <span>
                    {shown.duration_ms ? `${(shown.duration_ms / 1000).toFixed(2)}s` : "—"} ·{" "}
                    {shown.has_audio ? "有" : "无"}
                  </span>
                </div>
                {shown.drift ? (
                  <div>
                    <b>漂移</b>
                    <span title="与本链锚点帧的差距，越大越像换了个世界">
                      {Object.entries(shown.drift)
                        .map(([k, v]) => `${k} ${v.toFixed(3)}`)
                        .join(" · ")}
                    </span>
                  </div>
                ) : null}
              </div>

              {shown.ir_violations.length ? (
                <>
                  <h4>IR 校验告警</h4>
                  <ul className="dbg-notes">
                    {shown.ir_violations.map((v) => (
                      <li key={v}>{v}</li>
                    ))}
                  </ul>
                </>
              ) : null}

              {/* Only present on a beat that re-drew its own first frame. On a
                  continuous beat the face came from the previous clip, and saying
                  so is more useful than an empty box. */}
              <h4>关键帧提示词（英文，给 SD3.5）</h4>
              {shown.keyframe_prompt ? (
                <pre className="dbg-pre">{shown.keyframe_prompt}</pre>
              ) : (
                <p className="dbg-empty">
                  这一拍接续上一拍的末帧，没有重新画关键帧——长相是继承来的。
                </p>
              )}

              <h4>IR（送进 H3 的完整提示词）</h4>
              {ir ? (
                <>
                  <pre className="dbg-pre">{ir.prompt}</pre>
                  <div className="dbg-kv dbg-kv-tight">
                    <div>
                      <b>字数</b>
                      <span>{ir.prompt.length}</span>
                    </div>
                    <div>
                      <b>source</b>
                      <span>{ir.meta.source ?? "—"}</span>
                    </div>
                    <div>
                      <b>attempts</b>
                      <span>{ir.meta.attempts ?? "—"}</span>
                    </div>
                  </div>
                </>
              ) : irError ? (
                <p className="dbg-empty">读不到 IR：{irError}</p>
              ) : (
                <p className="dbg-empty">
                  <span className="spinner" /> 正在读 IR 存档……
                </p>
              )}
            </>
          ) : (
            <p className="dbg-empty">还没有在播的片段。</p>
          )
        ) : null}

        {tab === "player" ? (
          <>
            <h4>三个 video 元素（● 是正在显示的那个）</h4>
            <StageProbe />
            <p className="dbg-empty">
              一次载入会给三条片子赋 src：在播的这一拍，加上预取的两个分支。
              <code>IDLE</code> + <code>NOTHING</code> = 根本没去取；<code>LOADING</code> +{" "}
              <code>NOTHING</code> = 在路上；<code>ENOUGH</code> 还 paused = 数据齐了，卡在
              play()；<code>EMPTY</code> + beat 显示 <code>—</code> = 这个元素被清空了，连 src
              都没有（此时哪怕写着 playing 也是假的：对没有 src 的元素调 play() 不会报错，只会永远
              不返回）。
            </p>
          </>
        ) : null}

        {tab === "bible" ? (
          bible ? (
            <>
              {notes.length ? (
                <>
                  <h4>补过的地方</h4>
                  <ul className="dbg-notes dbg-notes-warn">
                    {notes.map((n) => (
                      <li key={n}>{n}</li>
                    ))}
                  </ul>
                </>
              ) : null}

              <div className="dbg-kv">
                <div>
                  <b>题材</b>
                  <span>{bible.genre || "—"}</span>
                </div>
                <div>
                  <b>视角</b>
                  <span>{bible.pov === "first" ? "第一人称" : "第三人称"}</span>
                </div>
                <div>
                  <b>数值</b>
                  <span>{bible.stat_names.join(" · ") || "—"}</span>
                </div>
              </div>

              <h4>一句话故事</h4>
              <p className="dbg-text">{bible.logline || "—"}</p>

              <h4>前提</h4>
              <p className="dbg-text">{bible.premise || "—"}</p>

              {/* The anchor is appended to every IR verbatim in code, so a weak one
                  is a weak look in every single beat -- which is exactly why it is
                  printed in full and printed first among the frozen fields. */}
              <h4>风格锚点（逐字追加到每段 IR）</h4>
              <p className="dbg-text">{bible.style_anchor || "—"}</p>
              <h4>风格锚点·英文（只给文生图）</h4>
              <p className="dbg-text">
                {bible.style_anchor_en || <em className="dbg-miss">缺失，关键帧退回用中文锚点</em>}
              </p>

              <h4>音乐语法</h4>
              <p className="dbg-text">{bible.music_bible || "—"}</p>
              <h4>环境音底噪</h4>
              <p className="dbg-text">{bible.ambience || "—"}</p>

              <h4>角色（{bible.characters.length}）</h4>
              {bible.characters.map((c) => (
                <div key={c.id} className="dbg-card">
                  <b>
                    {c.name}
                    <code>{c.id}</code>
                    {c.id === bible.protagonist_id ? <i className="dbg-pill">主角</i> : null}
                  </b>
                  <p className="dbg-text">{c.appearance || "—"}</p>
                  <p className="dbg-text dbg-en">
                    {c.appearance_en || <em className="dbg-miss">缺 appearance_en，硬切时少一层约束</em>}
                  </p>
                  <p className="dbg-sub">
                    音色：{c.voice || "—"}　弧光：{c.arc || "—"}
                  </p>
                </div>
              ))}

              <h4>世界规则</h4>
              <ul className="dbg-notes">
                {bible.world_rules.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>

              <h4>三幕大纲</h4>
              <ul className="dbg-notes">
                {bible.outline.map((a) => (
                  <li key={a.act}>
                    第 {a.act} 幕（{a.target_beats} 拍）：{a.milestone}
                  </li>
                ))}
              </ul>

              <h4>开场关键帧提示词</h4>
              <pre className="dbg-pre">{bible.opening_keyframe_prompt || "—"}</pre>
            </>
          ) : (
            <p className="dbg-empty">世界圣经还没写完。</p>
          )
        ) : null}

        {tab === "events" ? (
          events.length ? (
            <ol className="dbg-events">
              {[...events].reverse().map((e) => {
                const { seq, type, ts, ...rest } = e;
                return (
                  <li key={seq}>
                    <span className="dbg-seq">{seq}</span>
                    <span className="dbg-etype">{type}</span>
                    <span className="dbg-edetail">{summarise(rest)}</span>
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="dbg-empty">还没有事件。</p>
          )
        ) : null}
      </div>
    </aside>
  );
}

/**
 * One line per event, and short enough to scan a hundred of them.
 *
 * Whole payloads are not printable here: `beat.ready` carries a full beat
 * summary and `bible.ready` carries the entire bible, so `JSON.stringify` of a
 * few events would fill the panel and bury the sequence, which is the only
 * reason to read a log in order. Objects collapse to `{…}` and strings are
 * clipped; the tabs above are where the full values live.
 */
function summarise(rest: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const [k, v] of Object.entries(rest)) {
    if (v === null || v === undefined) continue;
    let s: string;
    if (typeof v === "object") s = Array.isArray(v) ? `[${v.length}]` : "{…}";
    else s = String(v);
    if (s.length > 48) s = `${s.slice(0, 48)}…`;
    parts.push(`${k}=${s}`);
  }
  return parts.join("  ");
}
