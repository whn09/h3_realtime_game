"use client";

import { useMemo } from "react";
import type { Beat, SessionView } from "@/lib/types";

/**
 * The branch DAG, laid out as one column per beat index.
 *
 * Worth its own panel because of what the pre-generation scheduler leaves behind:
 * the branch the player *didn't* take was generated anyway and is still on disk,
 * so walking back to a fork and taking the other path costs one seek and no GPU
 * time at all. Without somewhere to see that, those clips are simply thrown away.
 *
 * Laid out with CSS grid and drawn with one SVG overlay rather than a graph
 * library: the graph is a shallow tree with a handful of nodes per column, and a
 * layout engine would be more code than the arithmetic it replaces.
 */

const COL_W = 168;
const ROW_H = 76;
const NODE_W = 140;
const NODE_H = 56;

interface Props {
  session: SessionView;
  onSeek: (beatId: string) => void;
  onClose: () => void;
}

interface Placed {
  beat: Beat;
  col: number;
  row: number;
}

export default function StoryTree({ session, onSeek, onClose }: Props) {
  const { placed, byId, cols, rows } = useMemo(() => {
    const beats = Object.values(session.beats);
    const columns = new Map<number, Beat[]>();
    for (const b of beats) {
      const list = columns.get(b.index) ?? [];
      list.push(b);
      columns.set(b.index, list);
    }
    const placedList: Placed[] = [];
    const index = new Map<string, Placed>();
    const sortedCols = [...columns.keys()].sort((a, b) => a - b);
    let maxRows = 1;
    for (const col of sortedCols) {
      // Stable order inside a column: by parent's row first, so sibling pairs sit
      // next to each other and the edges do not cross for the common case.
      const list = (columns.get(col) ?? []).sort((a, b) => {
        const pa = a.parent_id ? index.get(a.parent_id)?.row ?? 0 : 0;
        const pb = b.parent_id ? index.get(b.parent_id)?.row ?? 0 : 0;
        return pa - pb || a.id.localeCompare(b.id);
      });
      list.forEach((beat, row) => {
        const p: Placed = { beat, col: sortedCols.indexOf(col), row };
        placedList.push(p);
        index.set(beat.id, p);
      });
      maxRows = Math.max(maxRows, list.length);
    }
    return { placed: placedList, byId: index, cols: sortedCols.length, rows: maxRows };
  }, [session.beats]);

  const width = Math.max(cols * COL_W, 320);
  const height = Math.max(rows * ROW_H, 120);
  const cx = (p: Placed) => p.col * COL_W + NODE_W / 2;
  const cy = (p: Placed) => p.row * ROW_H + NODE_H / 2;
  const onPath = new Set(session.path);

  return (
    <div className="tree" role="dialog" aria-label="故事树">
      <div className="tree-head">
        <span>故事树 · {placed.length} 拍</span>
        <span className="tree-hint">未走过的分支也已生成，点击即可回到岔路口</span>
        <button onClick={onClose}>关闭</button>
      </div>
      <div className="tree-scroll">
        <div className="tree-canvas" style={{ width, height }}>
          <svg className="tree-edges" width={width} height={height}>
            {placed.map((p) => {
              if (!p.beat.parent_id) return null;
              const parent = byId.get(p.beat.parent_id);
              if (!parent) return null;
              const x1 = cx(parent) + NODE_W / 2 - 8;
              const y1 = cy(parent);
              const x2 = cx(p) - NODE_W / 2 + 8;
              const y2 = cy(p);
              const mid = (x1 + x2) / 2;
              const live = onPath.has(p.beat.id) && onPath.has(parent.beat.id);
              return (
                <path
                  key={p.beat.id}
                  className={live ? "edge edge-live" : "edge"}
                  d={`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`}
                />
              );
            })}
          </svg>
          {placed.map((p) => {
            const b = p.beat;
            return (
              <button
                key={b.id}
                className={[
                  "node",
                  session.cursor === b.id ? "node-cursor" : "",
                  onPath.has(b.id) ? "node-path" : "",
                  b.status === "ready" ? "node-ready" : "",
                  b.status === "failed" ? "node-failed" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                style={{ left: p.col * COL_W, top: p.row * ROW_H, width: NODE_W, height: NODE_H }}
                disabled={b.status !== "ready"}
                onClick={() => onSeek(b.id)}
                title={`${b.label} · ${b.shot_type} · ${b.transition} · ${b.status}`}
              >
                {b.poster_url ? <img src={b.poster_url} alt="" /> : <span className="node-blank" />}
                <span className="node-label">{b.label || `第 ${b.index} 拍`}</span>
                {b.status !== "ready" ? <span className="node-badge">{b.status}</span> : null}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
