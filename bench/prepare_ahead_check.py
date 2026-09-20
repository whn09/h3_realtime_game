"""Does preparing ahead actually take the IR and the keyframe off the critical path?

Plays three beats against `FAKE_GPU=1`, which is the only way to answer this
without a GPU: the fake backend returns a clip in ~1s, so `beat_wall_ms` is
dominated by exactly the two Bedrock calls this change is about, and a beat that
was prepared should come back an order of magnitude faster than one that was not.

Checks four things, and the last two are the ones that would silently not work:

  1. the grandchildren of the cursor get `prepared_ahead_ms` -- the work happened
     at depth 2 at all;
  2. a beat that reaches production already prepared reports `beat_wall_ms` well
     below its own `promptir_wall_ms` + `keyframe_ms`, which is the definition of
     "this happened while the player was watching something else";
  3. no beat compiled its IR twice -- `promptir_calls` counts attempts, so a
     double compile shows up as a beat whose archived meta says otherwise;
  4. nothing was prepared for a `continuous` beat's keyframe, since that decision
     does not exist yet at prepare time.

Usage (on the box, against a throwaway server):
    python bench/prepare_ahead_check.py http://127.0.0.1:8109 cultivation
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8109"
PRESET = sys.argv[2] if len(sys.argv) > 2 else "cultivation"


def req(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        f"{BASE}{path}", data=data, method=method,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=180) as resp:
        return json.load(resp)


def wait_for(sid: str, pred, label: str, timeout: float = 300.0) -> dict:
    """Poll the session document. Polling rather than the SSE stream because this
    asks about state, not about ordering, and state is what `GET /sessions` has."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        s = req("GET", f"/sessions/{sid}")
        if pred(s):
            return s
        now = f"{s['phase']} cursor={s.get('cursor')}"
        if now != last:
            print(f"    … {now}", flush=True)
            last = now
        time.sleep(2.0)
    raise SystemExit(f"timed out waiting for {label}")


def beats_of(s: dict) -> dict[str, dict]:
    return s["beats"]


def children_of(s: dict, beat_id: str) -> list[str]:
    return list(beats_of(s)[beat_id].get("children", {}).values())


print(f"creating a session from preset {PRESET} …", flush=True)
sess = req("POST", "/sessions", {"preset_id": PRESET})
sid = sess["id"]
print(f"  session {sid}", flush=True)

# The opening: bible (~90s on Sonnet 5) then one clip.
s = wait_for(
    sid,
    lambda s: any(b["status"] == "ready" for b in beats_of(s).values()),
    "the first beat",
)
root = next(b for b in beats_of(s).values() if b["status"] == "ready")
print(f"  opening beat {root['id']} ready", flush=True)

trace: list[tuple[str, dict]] = [(root["id"], root)]

for step in range(2):
    # Options for the cursor arrive with the beat; its children's children are
    # what `prepare_ahead` is supposed to be working on right now.
    s = wait_for(
        sid,
        lambda s: len(children_of(s, s["cursor"])) >= 2,
        "options on the cursor",
    )
    cursor_id = s["cursor"]
    kids = children_of(s, cursor_id)

    # Give the look-ahead and the prepares time to land. Not a race we can join
    # from out here -- this is the window the player spends watching the clip.
    print(f"  cursor {cursor_id} -> {kids}; waiting on depth-2 prepares …", flush=True)
    s = wait_for(
        sid,
        lambda s: all(
            "prepared_ahead_ms" in beats_of(s).get(gc, {}).get("timings", {})
            for k in children_of(s, cursor_id)
            for gc in children_of(s, k)
        ) and any(children_of(s, k) for k in children_of(s, cursor_id)),
        "prepared grandchildren",
        timeout=180.0,
    )
    for k in children_of(s, cursor_id):
        for gc in children_of(s, k):
            b = beats_of(s)[gc]
            t = b["timings"]
            print(
                f"    prepared {gc:>18} transition={b['transition']:<10} "
                f"prepared_ahead={t['prepared_ahead_ms']:>8.0f}ms "
                f"ir={t.get('promptir_wall_ms', 0):>7.0f}ms "
                f"kf={t.get('keyframe_ms', 0):>7.0f}ms",
                flush=True,
            )

    print(f"  choosing option 0 (step {step + 1}) …", flush=True)
    picked = req("POST", f"/sessions/{sid}/choose", {"option_index": 0})["beat_id"]
    s = wait_for(
        sid, lambda s: beats_of(s)[picked]["status"] in ("ready", "failed"), f"beat {picked}"
    )
    b = beats_of(s)[picked]
    if b["status"] == "failed":
        raise SystemExit(f"beat {picked} failed: {b.get('error')}")
    trace.append((picked, b))

print("\n--- what each beat cost ------------------------------------------------")
print(f"{'beat':>18} {'trans':<11} {'wall':>9} {'ir':>9} {'kf':>9} {'prepared':>9}  verdict")
for bid, b in trace:
    t = b["timings"]
    wall = t.get("beat_wall_ms", 0.0)
    ir = t.get("promptir_wall_ms", 0.0)
    kf = t.get("keyframe_ms", 0.0)
    pre = t.get("prepared_ahead_ms")
    # `max` and not the sum: the two run concurrently, so what production would
    # have had to wait for is whichever finished last.
    off = max(ir, kf)
    if pre is None:
        verdict = "not prepared (expected for the opening)"
    elif wall < off:
        verdict = f"OK, {off - wall:.0f}ms of model time moved off the beat"
    else:
        verdict = f"** wall {wall:.0f}ms still covers the {off:.0f}ms it should not"
    print(f"{bid:>18} {b['transition']:<11} {wall:>9.0f} {ir:>9.0f} {kf:>9.0f} "
          f"{(pre if pre is not None else float('nan')):>9.0f}  {verdict}")

print("\n--- invariants ---------------------------------------------------------")
s = req("GET", f"/sessions/{sid}")
bad = 0
for bid, b in beats_of(s).items():
    t = b["timings"]
    # A prepared keyframe on a beat the Director marked continuous means the
    # gate in `_wants_early_keyframe` is not holding, and 36% of those images
    # would be thrown away by the rewrite to `cut`.
    if b["transition"] == "continuous" and b.get("parent_id") and "prepared_ahead_ms" in t:
        if t.get("keyframe_ms") and b["status"] != "ready":
            print(f"** {bid}: keyframe drawn at prepare time for a continuous beat")
            bad += 1
for bid, b in trace:
    ir = req("GET", f"/sessions/{sid}/beats/{bid}/ir")
    if ir["meta"].get("attempts", 1) > 2:
        print(f"** {bid}: IR archive says attempts={ir['meta']['attempts']}")
        bad += 1
print("clean" if not bad else f"{bad} problem(s)")
print(f"\nsession {sid}")
