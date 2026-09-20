"""Does `_produce` join an in-flight `_prepare` instead of racing it?

The end-to-end check cannot reach this: there, prepares finish during the 14s the
player spends watching, so production always finds the work already done. The race
only happens when the cursor advances onto a beat mid-prepare -- a fast double
choice, or a slow Bedrock call -- and it is the one case where the change could do
damage rather than merely fail to help: two SD3.5 images would be drawn, and only
one of them would end up in `keyframe_path`, leaving a paid-for picture that
nothing points at.

So this drives the two coroutines directly against stubs that count calls and take
long enough to overlap. No Bedrock, no GPU, no network.

    python bench/prepare_race_check.py
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "orchestrator"))

from app import engine as eng  # noqa: E402
from app.schema import (  # noqa: E402
    Beat, BeatStatus, BranchIntent, IRSections, Session, SessionPhase, ShotSpec, WorldBible,
)

CALLS: dict[str, int] = {"ir": 0, "kf": 0}


@dataclass
class _Compiled:
    ir: IRSections
    source: str = "llm"
    attempts: int = 1
    violations: tuple = ()
    timings: dict | None = None


class StubPromptIR:
    async def compile(self, **kw):
        CALLS["ir"] += 1
        await asyncio.sleep(0.3)
        return _Compiled(
            ir=IRSections(description="d", soundscape="s", music="m"),
            violations=[], timings={"promptir_wall_ms": 300.0},
        )


@dataclass
class _Frame:
    path: Path
    url: str
    source: str = "stub"
    elapsed_ms: float = 400.0


class StubKeyframer:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp

    async def generate(self, sid, name, prompt, seed=0):
        CALLS["kf"] += 1
        await asyncio.sleep(0.4)
        # A real file, because `_have_keyframe` checks the filesystem and a stub
        # that skipped that would test a different function than the one shipping.
        p = self.tmp / f"{name}-{CALLS['kf']}.png"
        p.write_bytes(b"\x89PNG stub")
        return _Frame(path=p, url=f"http://x/{p.name}")


class StubGpu:
    def prefetch(self, path):  # noqa: D102
        pass


class StubStore:
    def touch(self, sid): pass
    def write_ir(self, sid, bid, prompt, meta): pass


class StubBus:
    def emit(self, *a, **kw): pass


def make(tmp: Path):
    e = eng.Engine.__new__(eng.Engine)
    e.store = StubStore()
    e.promptir = StubPromptIR()
    e.keyframer = StubKeyframer(tmp)
    e.gpu = StubGpu()

    shot = ShotSpec(subject="s", action="a", setting="t")
    bible = WorldBible(premise="p", style_anchor="anchor")
    beat = Beat(id="b", intent=BranchIntent(label="l", shot=shot, transition="cut"), parent_id="p0")
    session = Session(id="s", premise="p", phase=SessionPhase.PLAYING, bible=bible)
    session.beats[beat.id] = beat

    rt = eng.Runtime(session=session, bus=StubBus())
    return e, rt, beat


async def main() -> int:
    tmp = Path("/tmp/prepare_race")
    tmp.mkdir(parents=True, exist_ok=True)
    bad = 0

    # --- the race: production starts 100ms into a prepare --------------------
    CALLS.update(ir=0, kf=0)
    e, rt, beat = make(tmp)

    async def produce_soon():
        await asyncio.sleep(0.1)
        # Only the part of `_produce` under test: the join, then the two steps that
        # are supposed to be no-ops. Calling the real `_produce` would need a GPU.
        waited = rt.preparing.get(beat.id)
        if waited is not None:
            await waited.wait()
        kf_task = None
        if e._wants_early_keyframe(beat) and not eng._have_keyframe(beat):
            kf_task = asyncio.create_task(e._fresh_keyframe(rt, beat))
        await e._compile_ir(rt, beat)
        return await e._conditioning_frame(rt, beat, kf_task)

    frame, _ = await asyncio.gather(produce_soon(), e._prepare(rt, beat.id))
    print(f"overlapping prepare+produce: ir_calls={CALLS['ir']} kf_calls={CALLS['kf']}")
    if CALLS["ir"] != 1:
        print(f"** IR compiled {CALLS['ir']}x, expected once"); bad += 1
    if CALLS["kf"] != 1:
        print(f"** keyframe drawn {CALLS['kf']}x, expected once -- a paid image is orphaned"); bad += 1
    if frame != beat.keyframe_path:
        print(f"** conditioning frame {frame} is not the recorded {beat.keyframe_path}"); bad += 1
    else:
        print(f"  conditioning frame is the prepared one: {Path(frame).name}")

    # --- the sequential case: prepare finishes long before production --------
    CALLS.update(ir=0, kf=0)
    e, rt, beat = make(tmp)
    await e._prepare(rt, beat.id)
    await e._compile_ir(rt, beat)
    reused = await e._conditioning_frame(rt, beat, None)
    print(f"prepare then produce:        ir_calls={CALLS['ir']} kf_calls={CALLS['kf']}")
    if (CALLS["ir"], CALLS["kf"]) != (1, 1):
        print("** prepared work was redone"); bad += 1
    if reused != beat.keyframe_path:
        print("** the prepared keyframe was not reused"); bad += 1
    if beat.status is not BeatStatus.PLANNED:
        print(f"** prepare moved the status to {beat.status}"); bad += 1
    else:
        print("  status still 'planned' after prepare, as the UI expects")

    # --- a stale path: the file is gone after a restart ---------------------
    CALLS.update(ir=0, kf=0)
    e, rt, beat = make(tmp)
    await e._prepare(rt, beat.id)
    Path(beat.keyframe_path).unlink()
    redrawn = await e._conditioning_frame(rt, beat, None)
    print(f"prepared image deleted:      kf_calls={CALLS['kf']} (expect 2, it re-draws)")
    if CALLS["kf"] != 2 or not Path(redrawn).exists():
        print("** a missing prepared keyframe was not re-drawn"); bad += 1

    # --- continuous: nothing to draw ahead of time -------------------------
    CALLS.update(ir=0, kf=0)
    e, rt, beat = make(tmp)
    beat.intent.transition = "continuous"
    await e._prepare(rt, beat.id)
    print(f"continuous beat prepared:    ir_calls={CALLS['ir']} kf_calls={CALLS['kf']} (expect 1, 0)")
    if (CALLS["ir"], CALLS["kf"]) != (1, 0):
        print("** a continuous beat had its keyframe drawn on a guess"); bad += 1

    print("\nclean" if not bad else f"\n{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
