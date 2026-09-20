"use client";

import { useEffect, useState } from "react";
import { createSession, getPresets, listSessions } from "@/lib/api";
import type { Preset, SessionSummary } from "@/lib/types";
import History from "./History";

/**
 * Scene and worldview setup.
 *
 * The presets and the free-text box are not alternatives: the orchestrator treats
 * a preset as a starting point and lets a typed premise win, which is how "pick
 * 修仙, then describe your own corner of it" works. So both are live at once and
 * the button never disables one.
 *
 * This screen also carries a second job: its submit click is the user gesture the
 * browser requires before audible playback. Without it the first clip would load
 * and then silently refuse to start.
 */

interface Props {
  onCreated: (sid: string) => void;
  /** Resume a past run. Same handler as `onCreated` -- a session id is a session
   *  id, whether it was made a second ago or yesterday. */
  onResume: (sid: string) => void;
}

export default function Setup({ onCreated, onResume }: Props) {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [history, setHistory] = useState<SessionSummary[]>([]);
  const [presetId, setPresetId] = useState("");
  const [premise, setPremise] = useState("");
  const [genre, setGenre] = useState("");
  const [pov, setPov] = useState<"first" | "third">("third");
  const [beatSeconds, setBeatSeconds] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPresets()
      .then((r) => {
        setPresets(r.presets);
        setBeatSeconds(r.beat_seconds);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    // Failure here is deliberately silent. An empty history is the normal state
    // on a first run, and surfacing a fetch error for it next to the presets
    // would make a working setup screen look broken.
    listSessions()
      .then((r) => setHistory(r.sessions))
      .catch(() => undefined);
  }, []);

  const selected = presets.find((p) => p.id === presetId);

  const submit = async () => {
    if (busy) return;
    if (!presetId && !premise.trim()) {
      setError("先选一个开局，或者自己写一句。");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const s = await createSession({
        preset_id: presetId || undefined,
        premise: premise.trim() || undefined,
        genre: genre.trim() || undefined,
        pov,
      });
      onCreated(s.id);
    } catch (e) {
      setBusy(false);
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="setup">
      <header>
        <h1>生成式互动影像</h1>
        <p>你说一个世界，它拍给你看。每一次选择都会即时生成下一段影像。</p>
      </header>

      <section>
        <h2>选一个开局</h2>
        <div className="presets">
          {presets.map((p) => (
            <button
              key={p.id}
              className={presetId === p.id ? "preset preset-on" : "preset"}
              onClick={() => setPresetId(presetId === p.id ? "" : p.id)}
            >
              <span className="preset-title">{p.title}</span>
              <span className="preset-genre">{p.genre}</span>
              <span className="preset-premise">{p.premise}</span>
            </button>
          ))}
          {presets.length === 0 ? <p className="dim">正在读取开局……</p> : null}
        </div>
      </section>

      <section>
        <h2>
          或者自己写 <span className="dim">（和上面的开局可以叠加，写了就以你的为准）</span>
        </h2>
        <textarea
          value={premise}
          maxLength={4000}
          rows={4}
          placeholder={
            selected
              ? `在「${selected.title}」里，你想从哪一刻开始？`
              : "例如：我在一列永不停站的夜班地铁上醒来，车厢里所有人都戴着我的脸。"
          }
          onChange={(e) => setPremise(e.target.value)}
        />
        <div className="row">
          <label>
            题材
            <input
              value={genre}
              maxLength={120}
              placeholder={selected?.genre || "留空由 AI 判断"}
              onChange={(e) => setGenre(e.target.value)}
            />
          </label>
          <label>
            视角
            <select value={pov} onChange={(e) => setPov(e.target.value as "first" | "third")}>
              <option value="third">第三人称（看着主角）</option>
              <option value="first">第一人称（主观镜头）</option>
            </select>
          </label>
        </div>
      </section>

      <footer>
        <button className="go" onClick={() => void submit()} disabled={busy}>
          {busy ? "正在开场……" : "开始"}
        </button>
        <p className="dim">
          {beatSeconds
            ? `每段影像 ${beatSeconds.toFixed(3)} 秒，带声音。开场要等世界圣经和第一个镜头，约半分钟。`
            : "开场要等世界圣经和第一个镜头。"}
        </p>
        {error ? <p className="err">{error}</p> : null}
      </footer>

      {/* Last, not first: a returning player scrolls to it, while a new one is
          not asked to walk past a list of runs that are not theirs. */}
      {history.length > 0 ? <History rows={history} onResume={onResume} /> : null}
    </div>
  );
}
