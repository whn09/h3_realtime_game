"""The beat pipeline and the session state machine.

## The dependency graph, which is the whole point

A beat's video and its *children's* options do not depend on each other, so they
run concurrently:

    beat N's intent known
      ├─ PromptIR(N) ─ keyframe or parent's last frame ─ H3(N) ─ last frame(N)
      └─ Director(N) ─ two child intents ─ PromptIR(N+1a/b) ─┐
                                                             └─ H3(N+1a/b)
                                                                (blocked only on
                                                                 last frame(N))

The Director and both children's IR compilation therefore overlap with beat N's
generation entirely, and the children's GPU work starts the instant N's last frame
lands. This is what the time budget in DESIGN.md section 2 is actually buying:
only the strictly serial edge -- last frame(N) -> H3(N+1) -- has to fit inside a
beat's playback.

## Why the tree is grown from the cursor, not recursively

Expanding every beat as soon as it exists would grow 2^N videos. Expansion is
driven from the player's cursor with a bounded lookahead (`pregen_depth`), so the
tree only ever holds the beats that have been visited plus one ply of unplayed
siblings -- which is exactly the set the story-tree panel can offer for free
backtracking (section 6.4).

## Nothing here raises at the player

Every failure path lands on a playable state: a failed IR becomes a templated IR,
a failed Director becomes generic options, a failed generation becomes a `failed`
beat the UI covers with a freeze frame and narration. The session's job is to keep
going (section 7).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import gpu, keyframe as kf
from .config import settings
from .director import Director
from .events import EventHub, SessionBus
from .llm import LLM, LLMError
from .promptir import PromptIR
from .schema import (
    Beat,
    BeatStatus,
    BranchIntent,
    Session,
    SessionPhase,
    StateDelta,
    WorldState,
)
from .store import Store
from .worldsmith import Worldsmith

log = logging.getLogger("h3game.engine")


class EngineError(RuntimeError):
    pass


@dataclass
class Runtime:
    session: Session
    bus: SessionBus
    frame_ready: dict[str, asyncio.Event] = field(default_factory=dict)
    inflight: set[str] = field(default_factory=set)
    # Beats whose Director call is in flight, so a second caller can *wait* for
    # the expansion instead of returning early -- see `_expand`.
    expanding: dict[str, asyncio.Event] = field(default_factory=dict)
    # The same trick for `_prepare`. Needed for a different reason than
    # `expanding`: `prepare:{id}` and `produce:{id}` are distinct `_spawn` keys,
    # so the idempotence guard there does not see them as the same work, and the
    # cursor can advance onto a beat whose prepare is still running. Without a
    # join, that beat pays for a second Haiku call and a second SD3.5 image --
    # and then one of the two images loses the race to set `keyframe_path`, so it
    # is not just waste but a picture that nothing points at.
    preparing: dict[str, asyncio.Event] = field(default_factory=dict)
    tasks: set[asyncio.Task] = field(default_factory=set)

    def frame_event(self, beat_id: str) -> asyncio.Event:
        if beat_id not in self.frame_ready:
            self.frame_ready[beat_id] = asyncio.Event()
        return self.frame_ready[beat_id]


class Engine:
    def __init__(self, store: Store, hub: EventHub, client: httpx.AsyncClient) -> None:
        self.store = store
        self.hub = hub
        self.client = client
        self.llm = LLM()
        self.worldsmith = Worldsmith(self.llm)
        self.director = Director(self.llm)
        self.promptir = PromptIR(self.llm)
        self.keyframer = kf.Keyframer()
        self.gpu = gpu.GpuPool(client)
        self.runtimes: dict[str, Runtime] = {}

    # -- plumbing ----------------------------------------------------------- #

    def runtime(self, sid: str) -> Runtime:
        if sid in self.runtimes:
            return self.runtimes[sid]
        session = self.store.get(sid)
        if not session:
            raise EngineError(f"session {sid} not found")
        rt = Runtime(session=session, bus=self.hub.bus(sid))
        # A session reloaded from disk has beats whose frames already exist; mark
        # them ready so a resumed session does not deadlock waiting for a
        # generation that finished before the restart.
        for beat in session.beats.values():
            if beat.last_frame_path or beat.status is BeatStatus.READY:
                rt.frame_event(beat.id).set()
        self.runtimes[sid] = rt
        return rt

    def _spawn(self, rt: Runtime, key: str, coro) -> None:
        """Launch `coro` once per key. Idempotent, so it is safe to re-drive the
        frontier on every cursor move without duplicating work."""
        if key in rt.inflight:
            coro.close()
            return
        rt.inflight.add(key)

        async def runner() -> None:
            try:
                await coro
            except Exception as exc:  # noqa: BLE001
                log.exception("task %s failed: %s", key, exc)
                rt.bus.emit("error", where=key, detail=str(exc)[:400])
            finally:
                rt.inflight.discard(key)

        task = asyncio.create_task(runner(), name=f"{rt.session.id}:{key}")
        rt.tasks.add(task)
        task.add_done_callback(rt.tasks.discard)

    def _set_status(self, rt: Runtime, beat: Beat, status: BeatStatus, **extra) -> None:
        beat.status = status
        self.store.touch(rt.session.id)
        rt.bus.emit("beat.status", beat_id=beat.id, status=status.value, **extra)

    # -- session creation --------------------------------------------------- #

    async def create_session(
        self, premise: str, genre: str = "", pov: str = ""
    ) -> Session:
        sid = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
        session = Session(
            id=sid,
            phase=SessionPhase.CREATING,
            premise=premise.strip(),
            genre=genre.strip(),
            pov="first" if pov == "first" else "third",
        )
        self.store.put(session)
        rt = Runtime(session=session, bus=self.hub.bus(sid))
        self.runtimes[sid] = rt
        rt.bus.emit("session.created", session_id=sid, premise=session.premise)
        self._spawn(rt, "bootstrap", self._bootstrap(rt))
        return session

    async def _bootstrap(self, rt: Runtime) -> None:
        session = rt.session
        rt.bus.emit("worldsmith.started")
        try:
            bible, state, notes, timings = await self.worldsmith.build(
                session.premise, session.genre, session.pov
            )
        except LLMError as exc:
            session.phase = SessionPhase.FAILED
            session.error = str(exc)
            self.store.touch(session.id)
            rt.bus.emit("session.failed", detail=str(exc)[:400])
            return

        session.bible = bible
        session.genre = bible.genre or session.genre
        session.pov = bible.pov
        session.phase = SessionPhase.OPENING
        # Kept on the session, not just emitted: this one call is the largest
        # single number in the whole opening -- 73s of the measured 99s to first
        # picture -- and until it was stored here that had to be inferred from the
        # gap between `session.created_at` and the root beat's `created_at`. It is
        # also the only latency in the service that no amount of pre-generation can
        # hide, because the root beat's keyframe prompt comes out of this call.
        session.timings = dict(timings)
        session.bible_notes = list(notes)
        self.store.touch(session.id)
        rt.bus.emit(
            "bible.ready",
            bible=bible.model_dump(mode="json"),
            notes=notes,
            timings=timings,
        )

        assert bible.opening is not None  # _ensure_playable guarantees this
        root = Beat(
            id=f"b0-{uuid.uuid4().hex[:6]}",
            parent_id=None,
            index=0,
            origin="opening",
            label="开场",
            intent=BranchIntent(
                label="开场",
                transition="cut",  # nothing to continue from
                shot=bible.opening,
                state_delta=StateDelta(summary_append="故事开始"),
            ),
            state_after=state.model_copy(deep=True),
        )
        root.state_after.recent_shot_types = [bible.opening.type]
        session.beats[root.id] = root
        session.root_id = root.id
        session.cursor = root.id
        session.path = [root.id]
        self.store.touch(session.id)
        rt.bus.emit("beat.created", beat=_beat_summary(root), is_root=True)

        # The opening beat blocks the player by definition -- nothing is playing
        # to hide it behind.
        self._spawn(rt, f"produce:{root.id}", self._produce(rt, root.id, gpu.PRIORITY_BLOCKING))
        self._spawn(rt, "frontier", self._ensure_frontier(rt))

    # -- the frontier ------------------------------------------------------- #

    async def _ensure_frontier(self, rt: Runtime) -> None:
        """Make sure everything reachable within `pregen_depth` of the cursor is
        either generating or done. Safe to call repeatedly."""
        session = rt.session
        if session.phase is SessionPhase.ENDED:
            return
        cursor = session.cursor_beat
        if not cursor:
            return

        await self._expand(rt, cursor.id)
        cursor = session.cursor_beat
        if not cursor or cursor.is_ending:
            return

        children = list(cursor.children.values())
        for child_id in children:
            self._spawn(rt, f"produce:{child_id}", self._produce(rt, child_id, gpu.PRIORITY_PREGEN))

        # Expand the children *now*, while the player is still watching the
        # cursor. This is the biggest latency lever in the whole loop and it
        # costs no GPU time at all -- only two extra Director calls.
        #
        # Measured on a real 6-beat playthrough: `director_total` is 27-49s
        # while a beat's own generation is 18-24s. Expanding a beat only when
        # the cursor arrived at it put both of those *after* the player
        # committed, in series: 45-73s of dead air against 14.375s of playback.
        # That is the stall that reads as "generation is slow", and almost none
        # of it was generation.
        #
        # Expanding ahead is sound because a beat's options depend on its
        # `intent` and `state_after`, both of which exist the moment the beat is
        # created by `_make_child` -- not on its video. So the Director for
        # every reachable next beat can run concurrently with the current one's
        # playback, leaving only the child's own generation on the critical path.
        for child_id, res in zip(
            children,
            await asyncio.gather(
                *(self._expand(rt, cid) for cid in children), return_exceptions=True
            ),
        ):
            if isinstance(res, BaseException):
                # A look-ahead failure is not fatal: the beat is still playable
                # and arriving at it will retry the expansion on the slow path.
                log.warning("look-ahead expand of %s failed: %s", child_id, res)

        # Depth 2, but only the parts that cost no GPU: compile the grandchildren's
        # IR, and draw the keyframe for the ones that already know they open on a
        # cut. The expansion that made them exist just happened in the loop above.
        #
        # This is the same argument as the Director look-ahead, applied one step
        # further. PromptIR and the image model read `intent` and `state_after` and
        # nothing else -- no video, no last frame -- so they can run as soon as
        # `_make_child` has run, which for a grandchild is now. Doing them here
        # rather than inside `_produce` takes max(4.5s, 6.3s) off the critical path
        # of the beat the player is about to choose, which is the whole of the
        # difference between a beat and the 14.375s it has to fit in.
        #
        # Deliberately *not* `pregen_depth >= 2`, and the distinction matters: that
        # setting is off because six clips against two non-preemptible slots makes
        # the player wait behind speculation. Nothing here takes a slot. The cost
        # is Bedrock calls on branches that are never played, which is money and
        # not latency -- so it degrades the bill, never the game.
        if settings.prepare_ahead:
            for child_id in children:
                child = session.beats.get(child_id)
                if not child:
                    continue
                for gc in child.children.values():
                    self._spawn(rt, f"prepare:{gc}", self._prepare(rt, gc))

        if settings.pregen_depth >= 2:
            # Speculative depth-2: pre-generate the children of the branch the
            # Director expects the player to take, covering the choice *after* this
            # one (DESIGN.md section 2.3 lever 4). The expansion it needs already
            # happened in the loop above.
            #
            # Off by default, because at two replicas it loses: it puts six clips in
            # flight against two slots and the beats the player actually needs end
            # up queued behind speculation, which no priority can fix once a
            # non-preemptible 13s generation has the slot. Measured numbers are in
            # `config.pregen_depth`. This path is kept because it becomes correct
            # the moment there are slots for it -- it is gated on capacity, not
            # wrong in principle.
            predicted_id = cursor.children.get(str(cursor.predicted_choice))
            predicted = session.beats.get(predicted_id or "")
            if predicted:
                for gc in predicted.children.values():
                    self._spawn(
                        rt, f"produce:{gc}",
                        self._produce(rt, gc, gpu.PRIORITY_SPECULATIVE),
                    )

    async def _expand(self, rt: Runtime, beat_id: str) -> None:
        """Run the Director for `beat_id` and materialise its child beats.

        Joinable, not fire-and-forget: a caller that finds an expansion already
        in flight waits for it rather than returning immediately. That matters
        because `_ensure_frontier` reads `cursor.children` on the line after
        awaiting this -- and now that children are expanded ahead of time, the
        cursor's own expansion is usually already running when the player
        arrives. Returning early there would leave the frontier looking at an
        un-expanded beat and silently skip the pre-generation it exists to
        schedule, which is worse than the duplicate call the guard prevents.
        """
        session = rt.session
        beat = session.beats.get(beat_id)
        if not beat or beat.options or beat.is_ending:
            return
        running = rt.expanding.get(beat_id)
        if running is not None:
            await running.wait()
            return
        done = rt.expanding[beat_id] = asyncio.Event()
        try:
            assert session.bible is not None
            out, issues, timings = await self.director.expand(session.bible, beat)

            beat.narration = out.narration
            beat.options = out.options
            beat.predicted_choice = out.predicted_choice
            beat.is_ending = out.is_ending
            beat.timings.update(timings)

            for i, opt in enumerate(out.options):
                child = self._make_child(session, beat, opt, str(i), origin="choice")
                rt.bus.emit("beat.created", beat=_beat_summary(child), parent_id=beat.id, option=i)

            self.store.touch(session.id)
            rt.bus.emit(
                "options.ready",
                beat_id=beat.id,
                narration=beat.narration,
                predicted_choice=beat.predicted_choice,
                options=[_option_summary(o) for o in beat.options],
                children=dict(beat.children),
                issues=issues,
                timings=timings,
            )
            if beat.is_ending:
                session.phase = SessionPhase.ENDED
                self.store.touch(session.id)
                rt.bus.emit("ending", beat_id=beat.id, ending=beat.is_ending.model_dump(mode="json"))

            # Off the critical path: keep the Director's context from growing
            # without bound.
            nxt = beat.state_after
            if nxt.beat_index and nxt.beat_index % settings.resummarise_every == 0:
                self._spawn(rt, f"summary:{beat.id}", self._resummarise(rt, beat.id))
        finally:
            # Released even on failure or cancellation: waiters then see
            # `beat.options` still empty, which is the same state they would see
            # after an error on the slow path and which every caller tolerates.
            rt.expanding.pop(beat_id, None)
            done.set()

    def _make_child(
        self, session: Session, parent: Beat, intent: BranchIntent, key: str, origin: str
    ) -> Beat:
        state = parent.state_after.apply(intent.state_delta)
        state.recent_shot_types = (parent.state_after.recent_shot_types + [intent.shot.type])[-6:]
        child = Beat(
            id=f"b{parent.index + 1}-{uuid.uuid4().hex[:6]}",
            parent_id=parent.id,
            index=parent.index + 1,
            origin=origin,  # type: ignore[arg-type]
            label=intent.label,
            intent=intent,
            state_after=state,
        )
        session.beats[child.id] = child
        parent.children[key] = child.id
        return child

    async def _resummarise(self, rt: Runtime, beat_id: str) -> None:
        session = rt.session
        beat = session.beats.get(beat_id)
        if not beat or not session.bible:
            return
        summary = await self.director.compress_summary(session.bible, beat.state_after)
        if not summary:
            return
        # Apply to this beat and to every descendant already created, so the
        # compression is not lost the moment the player moves on.
        beat.state_after.summary = summary
        for child_id in beat.children.values():
            child = session.beats.get(child_id)
            if child:
                child.state_after.summary = summary
        self.store.touch(session.id)
        rt.bus.emit("summary.updated", beat_id=beat_id, summary=summary)

    # -- preparing one beat, without the GPU -------------------------------- #

    def _chains_from_parent(self, beat: Beat) -> bool:
        """Will this beat's first frame be the previous clip's last frame?

        The one place that answer is computed, because two things read it and they
        must not disagree: `_compile_ir` writes the IR's continuity clause from it,
        and `_wants_early_keyframe` decides whether to spend $0.08 drawing. An IR
        that promises an unbroken shot alongside a drawn establishing frame is
        exactly the artefact this all came from.

        It is the early answer, taken before the parent's clip exists. Under the
        default it is also the final one. With `midstory_keyframes` on, a parent that
        turns out to have drifted can still flip a continuous beat to a cut inside
        `_conditioning_frame`, after this has been read -- so that path gets an IR
        written for the transition the beat used to have.
        """
        if not beat.parent_id:
            return False
        if not settings.midstory_keyframes:
            return True
        return beat.intent.transition == "continuous"

    def _wants_early_keyframe(self, beat: Beat) -> bool:
        """Whether this beat needs an image drawn for it at all, early or late.

        Under the default (`settings.midstory_keyframes` off) the answer is yes for
        exactly one beat per session: the opening, which has no previous frame to
        continue from. Every other beat chains, whatever the Director called its
        transition -- see `_conditioning_frame`.

        With mid-story keyframes on, this is also the early/late question it used to
        be. It is yes only for a beat the Director already marked as a cut: a
        `continuous` beat is excluded because `_conditioning_frame` may rewrite it
        once the parent's drift is known -- 36% of them did -- and that answer does
        not exist yet, so drawing for it would be a coin flip on a $0.08 image. The
        rewrite only ever went continuous -> cut and never back, so a beat that
        reads `cut` here still reads `cut` at production time.

        The inverse of `_chains_from_parent` by construction, not by coincidence: a
        beat draws its own opening frame exactly when it has no previous frame to
        continue from.
        """
        return not self._chains_from_parent(beat)

    async def _compile_ir(self, rt: Runtime, beat: Beat) -> None:
        """Compile this beat's IR into `beat.ir`, exactly once.

        A no-op if it is already there, which is what lets `_prepare` and
        `_produce` both call it unconditionally: whichever runs first pays, the
        other returns immediately.
        """
        if beat.ir is not None:
            return
        session = rt.session
        assert session.bible is not None
        compiled = await self.promptir.compile(
            bible=session.bible,
            state=beat.state_after,
            intent=beat.intent,
            chained=self._chains_from_parent(beat),
        )
        beat.ir = compiled.ir
        beat.ir_source = compiled.source  # type: ignore[assignment]
        beat.ir_violations = compiled.violations
        beat.timings.update(compiled.timings)
        self.store.write_ir(
            session.id, beat.id, compiled.ir.to_prompt(),
            {
                "source": compiled.source,
                "attempts": compiled.attempts,
                "violations": compiled.violations,
                "intent": beat.intent.model_dump(mode="json"),
            },
        )
        rt.bus.emit(
            "beat.ir", beat_id=beat.id, source=compiled.source,
            violations=compiled.violations, description=compiled.ir.description,
        )

    async def _prepare(self, rt: Runtime, beat_id: str) -> None:
        """Do everything a beat needs except the video: its IR, and its keyframe
        if that is already decided. Touches no GPU slot.

        Runs for grandchildren of the cursor, two choices out, so that by the time
        the player picks, the only work left is the 9.9s of H3 -- see
        `settings.prepare_ahead` for the arithmetic.

        Never calls `_set_status`. The beat stays `planned` throughout, which is
        what it is: nothing is being generated for it and the UI should not say
        otherwise. It also keeps `_produce`'s own status guard meaningful, since
        a prepared beat is indistinguishable from an untouched one as far as
        scheduling is concerned.
        """
        session = rt.session
        beat = session.beats.get(beat_id)
        # `planned` and not anything else: a beat already producing, ready, or
        # failed either does not need this or is past caring.
        if not beat or beat.status is not BeatStatus.PLANNED:
            return
        wants_kf = self._wants_early_keyframe(beat) and not beat.keyframe_path
        if beat.ir is not None and not wants_kf:
            return
        if beat_id in rt.preparing:
            return

        done = rt.preparing[beat_id] = asyncio.Event()
        started = time.perf_counter()
        try:
            jobs = [self._compile_ir(rt, beat)]
            if wants_kf:
                jobs.append(self._fresh_keyframe(rt, beat))
            for res in await asyncio.gather(*jobs, return_exceptions=True):
                if isinstance(res, BaseException):
                    # Not fatal, and not even logged as an error: `_produce` redoes
                    # whatever is still missing on its own critical path, which is
                    # exactly the behaviour before this method existed. A failed
                    # prepare costs the latency it was meant to save, nothing more.
                    log.warning("prepare of %s fell back to the slow path: %s", beat_id, res)
            # Recorded so the debug panel can say *why* `beat_wall_ms` is smaller
            # than the stages inside it. Without this marker a prepared beat looks
            # like a measurement bug -- 10s total containing a 4.5s compile and a
            # 6.3s image -- when it is the entire point: this many milliseconds
            # happened while the player was watching something else.
            beat.timings["prepared_ahead_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            self.store.touch(session.id)
        finally:
            done.set()
            rt.preparing.pop(beat_id, None)

    # -- producing one beat ------------------------------------------------- #

    async def _produce(self, rt: Runtime, beat_id: str, priority: int) -> None:
        session = rt.session
        beat = session.beats.get(beat_id)
        if not beat or beat.status in (BeatStatus.READY, BeatStatus.GENERATING):
            return
        # The cursor can arrive on a beat whose `_prepare` is still in flight.
        # Joining it is strictly better than racing it: the work is the work either
        # way, and duplicating it would also fork `keyframe_path`.
        in_prepare = rt.preparing.get(beat_id)
        if in_prepare is not None:
            await in_prepare.wait()
        assert session.bible is not None
        started = time.perf_counter()

        try:
            # 0. start the keyframe, if this beat already knows it starts fresh --
            # The image model costs ~7.6s and the keyframe depends on the shot
            # intent, not on the IR, so the two have no reason to be serial. A
            # `continuous` beat may not join in: whether it re-anchors is only
            # known once the parent's frame arrives, and that answer rewrites
            # `transition`, which rewrites the IR's continuity clause. Compiling
            # against a transition that is about to change would produce an IR
            # promising to continue a shot it is actually cutting away from.
            kf_task: asyncio.Task[str] | None = None
            if self._wants_early_keyframe(beat) and not _have_keyframe(beat):
                kf_task = asyncio.create_task(self._fresh_keyframe(rt, beat))

            # 1. compile the IR ------------------------------------------------
            # Both of these are usually already done by `_prepare`, in which case
            # the beat goes straight from `planned` to `queued`. The status is only
            # announced when there is really something to wait for, so `compiling`
            # in the UI keeps meaning "a model is running".
            if beat.ir is None:
                self._set_status(rt, beat, BeatStatus.COMPILING)
            await self._compile_ir(rt, beat)
            assert beat.ir is not None

            # 2. conditioning frame -------------------------------------------
            first_frame = await self._conditioning_frame(rt, beat, kf_task)
            # If what we are conditioning on is literally the parent's last
            # frame, that image may already be sitting on the replica that
            # produced it, in which case landing there costs no upload at all.
            # Guarded by an equality check rather than by `transition`, because
            # `_conditioning_frame` can silently downgrade a continuous beat to a
            # fresh keyframe -- and a fresh keyframe is nowhere on any box yet.
            parent = session.beats.get(beat.parent_id or "")
            first_frame_remote = (
                dict(parent.remote_last_frame)
                if parent and parent.last_frame_path == first_frame
                else {}
            )

            # 3. generate ------------------------------------------------------
            self._set_status(rt, beat, BeatStatus.QUEUED, priority=priority)
            req = gpu.GpuRequest(
                job_id=f"{session.id}-{beat.id}",
                ir=beat.ir,
                seconds=settings.beat_seconds,
                first_frame=first_frame,
                first_frame_remote=first_frame_remote,
                task="fl2va",
                seed=abs(hash((session.id, beat.id))) % 2_147_483_647,
            )
            result = await self.gpu.generate(
                req, priority=priority,
                on_slot=lambda ep: self._set_status(rt, beat, BeatStatus.GENERATING, endpoint=ep),
            )

            # 4. record --------------------------------------------------------
            beat.video_url = result.video_url
            beat.poster_url = result.poster_url
            beat.last_frame_url = result.last_frame_url
            beat.last_frame_path = result.last_frame_path
            beat.remote_last_frame = result.remote_last_frame
            beat.duration_ms = result.duration_ms
            beat.has_audio = result.has_audio
            beat.frame_stats = result.frame_stats
            beat.degraded = result.degraded
            beat.timings.update(result.timings)
            beat.timings["beat_wall_ms"] = round((time.perf_counter() - started) * 1000.0, 1)

            self._check_drift(rt, beat)
            # Start moving this frame to the GPU boxes now rather than when a
            # child needs it. On the real backend the conditioning frame has to
            # exist on the server's filesystem before the request is accepted, and
            # that upload is otherwise a second of dead time on the critical path
            # of every continuous beat.
            self.gpu.prefetch(beat.last_frame_path)
            beat.status = BeatStatus.READY
            self.store.touch(session.id)
            if session.phase is SessionPhase.OPENING:
                session.phase = SessionPhase.PLAYING
            rt.bus.emit("beat.ready", beat=_beat_summary(beat))

        except Exception as exc:  # noqa: BLE001
            beat.status = BeatStatus.FAILED
            beat.error = str(exc)[:400]
            self.store.touch(session.id)
            log.exception("beat %s failed", beat_id)
            rt.bus.emit("beat.failed", beat_id=beat.id, detail=beat.error)
        finally:
            # A keyframe still in flight means the beat failed before it was
            # needed; nobody will ever read it.
            if kf_task and not kf_task.done():
                kf_task.cancel()
            # Unblock children unconditionally. Without a last frame they fall
            # back to a fresh keyframe, which is a visible cut -- far better than
            # a subtree that waits forever.
            rt.frame_event(beat.id).set()

    async def _conditioning_frame(
        self, rt: Runtime, beat: Beat, kf_task: asyncio.Task[str] | None = None
    ) -> str:
        """Resolve the image this beat starts from.

        By default every beat but the opening waits for its parent's last frame --
        the single serial edge in the pipeline -- and starts from it. The opening
        draws, and only because it has nothing to continue from.

        The Director's `transition` does not change that, which is the whole of the
        change: a drawn image and the frame the player is looking at have never seen
        each other, so splicing them is a visible discontinuity no matter how good
        the image is. `transition` now describes what happens to the *story* and the
        IR renders it as camera movement inside one unbroken shot -- see
        `promptir._continuity_block`. `settings.midstory_keyframes` restores the old
        behaviour for comparison.
        """
        session = rt.session
        assert session.bible is not None

        # `_chains_from_parent`, not `beat.parent_id`: a beat that is going to draw
        # its own opening frame has no reason to wait for the parent's, and waiting
        # anyway would put the one serial edge in the pipeline in front of work that
        # does not depend on it. Under the default this is every beat but the
        # opening; with `midstory_keyframes` on it is the continuous ones, which is
        # what the wait was guarded by before.
        if self._chains_from_parent(beat):
            assert beat.parent_id is not None
            self._set_status(rt, beat, BeatStatus.WAITING_FRAME, parent_id=beat.parent_id)
            await rt.frame_event(beat.parent_id).wait()
            parent = session.beats.get(beat.parent_id)
            have_parent_frame = bool(
                parent and parent.last_frame_path and Path(parent.last_frame_path).exists()
            )
            # The re-anchor decision is read *here*, from the parent, rather than
            # from this beat's own copied flag. The Director expands a beat while
            # that beat is still generating, so children are created before the
            # parent's drift is measured -- a flag copied at creation time is
            # always one beat stale. This is the latest possible moment, and by
            # construction it is after the parent's `_check_drift`.
            wants_reanchor = settings.midstory_keyframes and bool(
                parent and parent.state_after.needs_reanchor
            )
            if wants_reanchor:
                log.info("beat %s forced to cut: parent needs re-anchoring", beat.id)
                # Late enough to be past `_compile_ir`, so the IR it is about to be
                # sent with was written for the transition it *had*. That is the
                # cost of correcting drift and the reason this path is now opt-in.
                beat.intent.transition = "cut"
                rt.bus.emit("reanchor.forced", beat_id=beat.id, parent_id=parent.id)  # type: ignore[union-attr]
            elif have_parent_frame:
                assert parent is not None and parent.last_frame_path is not None
                return parent.last_frame_path
            else:
                # Not a story decision, a missing file: the parent failed or its
                # frame was swept. Drawing is the only way to produce anything.
                log.warning(
                    "beat %s has no parent frame to continue from; drawing instead", beat.id
                )
                beat.intent.transition = "cut"

        # Three ways to have an opening image, in descending order of how early it
        # was started: a task `_produce` launched before compiling, one `_prepare`
        # finished two choices ago, or -- for a beat that only just discovered it
        # has to cut -- one generated now, serially, on the critical path.
        if kf_task:
            return await kf_task
        if _have_keyframe(beat):
            assert beat.keyframe_path is not None
            return beat.keyframe_path
        return await self._fresh_keyframe(rt, beat)

    async def _fresh_keyframe(self, rt: Runtime, beat: Beat) -> str:
        """Generate this beat's own opening image and return its local path.

        Called either speculatively (concurrently with PromptIR, for a beat that
        already knows it starts fresh) or on the serial path (for a continuous
        beat whose parent turned out to need re-anchoring), so it must not touch
        anything the IR compilation reads. It does not: `transition` is decided
        by the caller, and the two flags it writes are read only by the Director,
        one beat later.
        """
        session = rt.session
        assert session.bible is not None

        # The opening has a prompt written for it. `opening_keyframe_prompt` is
        # 70-130 English words the Worldsmith composed for this one frame, with the
        # whole bible in front of it -- and it was being thrown away in favour of
        # the generic assembly below, on the single frame the player stares at
        # while the first clip renders. The generic path is still the fallback,
        # since nothing guarantees the field came back.
        opening_prompt = session.bible.opening_keyframe_prompt.strip()
        if beat.index == 0 and opening_prompt:
            prompt = f"{opening_prompt}, cinematic still frame, 16:9, no text"[:2000]
        else:
            prompt = kf.build_prompt(session.bible, beat.intent.shot, beat.state_after)
        # Logged because this is the only place a face is decided, and "why does
        # she look different after the cut" is otherwise unanswerable after the
        # fact: the image is on disk but the sentence that produced it is not.
        log.info("keyframe prompt for %s: %s", beat.id, prompt)
        beat.keyframe_prompt = prompt
        frame = await self.keyframer.generate(
            session.id, f"kf-{beat.id}", prompt,
            seed=abs(hash(beat.id)) % 2_147_483_647,
        )
        beat.keyframe_url = frame.url
        beat.keyframe_path = str(frame.path)
        beat.timings["keyframe_ms"] = frame.elapsed_ms
        # Same reason as the last-frame prefetch in `_produce`: when this ran
        # concurrently with the IR there is real time left before generation, and
        # the upload is otherwise on the critical path.
        self.gpu.prefetch(str(frame.path))
        # A fresh keyframe *is* the re-anchor, so the counters reset here.
        beat.state_after.needs_reanchor = False
        beat.state_after.beats_since_anchor = 0
        if beat.index == 0:
            session.opening_keyframe_url = frame.url
        rt.bus.emit(
            "keyframe.ready", beat_id=beat.id, url=frame.url, source=frame.source,
            elapsed_ms=frame.elapsed_ms,
        )
        return str(frame.path)

    def _check_drift(self, rt: Runtime, beat: Beat) -> None:
        """Compare against the beat's *anchor*, not its parent and not beat 0.

        The parent is wrong because per-hop deltas stay under any sensible
        threshold indefinitely while the accumulated look walks away entirely.
        Beat 0 is wrong for a different reason, measured: with real keyframes
        every beat past index 1 breached on `d_luma` by 0.03-0.30 against a
        threshold of 0.06 -- and those beats had each been generated from their
        own fresh keyframe, so there was no chain behind them to have drifted at
        all. What beat 0 actually measures is "does this shot look like the
        opening shot", and a closeup is legitimately brighter and 4x sharper than
        a wide establishing shot. Answering that question with a re-anchoring cut
        made every beat cut, which removed the continuity the mechanism exists to
        protect.

        So the reference is the last fresh-keyframe beat on this branch, and a
        beat that started fresh reports no drift and becomes the new reference.
        """
        session = rt.session
        stats = beat.frame_stats
        if not stats:
            return
        if session.drift_baseline is None:
            # Kept as the fallback anchor and as the session's reference look.
            session.drift_baseline = dict(stats)

        # Whether an image was drawn, not what the transition says. Those were the
        # same question when every cut drew; now a `cut` beat normally chains from
        # its parent like any other, and reading `transition` here would reset the
        # reference on a beat that never left the chain -- which is the one way to
        # make a drift meter that can never report drift. `keyframe_path` is written
        # by `_fresh_keyframe` and by nothing else, so it is the physical fact.
        if _have_keyframe(beat):
            beat.drift_anchor = dict(stats)
            return

        parent = session.beats.get(beat.parent_id or "")
        anchor = (parent.drift_anchor if parent else None) or session.drift_baseline
        beat.drift_anchor = anchor
        beat.drift = gpu.drift_from(anchor, stats)
        breached = gpu.drift_exceeded(beat.drift)
        due = beat.state_after.beats_since_anchor >= settings.reanchor_every
        if breached or due:
            beat.state_after.needs_reanchor = True
            rt.bus.emit(
                "drift.reanchor", beat_id=beat.id, drift=beat.drift,
                breached=breached, interval_due=due,
            )

    # -- player actions ----------------------------------------------------- #

    async def choose(
        self, sid: str, option_index: int | None = None, custom_action: str | None = None
    ) -> Beat:
        rt = self.runtime(sid)
        session = rt.session
        cursor = session.cursor_beat
        if not cursor:
            raise EngineError("session has no current beat yet")
        if session.phase is SessionPhase.ENDED:
            raise EngineError("story has ended; seek to an earlier beat to branch again")

        if custom_action and custom_action.strip():
            child = await self._custom_child(rt, cursor, custom_action.strip())
        else:
            if not cursor.options:
                raise EngineError("options are not ready yet")
            idx = 0 if option_index is None else int(option_index)
            child_id = cursor.children.get(str(idx))
            if not child_id:
                raise EngineError(f"option {idx} does not exist")
            child = session.beats[child_id]

        session.cursor = child.id
        session.path = session.path + [child.id]
        self.store.touch(session.id)

        # The player is now staring at a freeze frame waiting for this clip, so it
        # jumps the queue ahead of its speculative sibling.
        bumped = await self.gpu.bump(f"{session.id}-{child.id}")
        self._spawn(rt, f"produce:{child.id}", self._produce(rt, child.id, gpu.PRIORITY_BLOCKING))

        rt.bus.emit(
            "cursor.moved", beat_id=child.id, beat=_beat_summary(child),
            path=session.path, bumped=bumped,
        )
        self._spawn(rt, f"frontier:{child.id}", self._ensure_frontier(rt))
        return child

    async def _custom_child(self, rt: Runtime, cursor: Beat, action: str) -> Beat:
        """The free-text path (DESIGN.md section 6.3).

        Cannot be pre-generated by definition, so the wait is real. The design
        accepts that: the player deliberately stepped off the offered path, and
        that is where the open-world feeling comes from."""
        session = rt.session
        assert session.bible is not None
        rt.bus.emit("custom.started", beat_id=cursor.id, action=action[:200])
        intent, narration, issues, timings = await self.director.custom(
            session.bible, cursor, action
        )
        key = f"custom:{len([k for k in cursor.children if k.startswith('custom')])}"
        child = self._make_child(session, cursor, intent, key, origin="custom")
        child.narration = narration
        child.timings.update(timings)
        self.store.touch(session.id)
        rt.bus.emit(
            "beat.created", beat=_beat_summary(child), parent_id=cursor.id,
            option=None, custom=True, narration=narration, issues=issues,
        )
        return child

    async def seek(self, sid: str, beat_id: str) -> Beat:
        """Jump to any beat already in the tree -- the free backtracking that
        double-branch pre-generation pays for (DESIGN.md section 6.4)."""
        rt = self.runtime(sid)
        session = rt.session
        beat = session.beats.get(beat_id)
        if not beat:
            raise EngineError(f"beat {beat_id} not found")

        path: list[str] = []
        node: Beat | None = beat
        while node:
            path.append(node.id)
            node = session.beats.get(node.parent_id) if node.parent_id else None
        session.path = list(reversed(path))
        session.cursor = beat.id
        if session.phase is SessionPhase.ENDED and not beat.is_ending:
            session.phase = SessionPhase.PLAYING
        self.store.touch(session.id)
        rt.bus.emit("cursor.moved", beat_id=beat.id, beat=_beat_summary(beat), path=session.path)
        self._spawn(rt, f"frontier:{beat.id}", self._ensure_frontier(rt))
        return beat

    async def delete(self, sid: str) -> bool:
        """Cancel everything this session has running, then erase it.

        Order matters. A session being deleted can easily have two clips on the
        GPUs and a Director call in flight, and every one of those tasks ends by
        writing to the document and touching the store -- so deleting the files
        first just means they come back, written by a task that does not know its
        session is gone.

        The cancellation is not awaited to completion, and deliberately so: a beat
        already handed to a replica takes up to 9s to come back, and the player who
        clicked delete is not waiting for that. `_produce` releases its slot in a
        `finally`, so the GPU is freed either way; what the cancelled task will
        find, if it gets far enough to write, is a session missing from the cache,
        and `store.touch` on an uncached id is a no-op. One `sleep(0)` to let each
        cancellation actually be delivered before the runtime disappears.
        """
        rt = self.runtimes.pop(sid, None)
        if rt is not None:
            for task in list(rt.tasks):
                task.cancel()
            await asyncio.sleep(0)
        existed = self.store.delete(sid)
        self.hub.drop(sid)
        return existed

    # -- introspection ------------------------------------------------------ #

    async def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "fake_gpu": settings.fake_gpu,
            # Which way the bytes go, and the URL the GPU boxes were told to fetch
            # frames from. Worth a line here because the wrong answer to the second
            # one fails at submit time on a box we do not control, and this is the
            # cheapest place to read what we actually advertised.
            "transport": settings.h3_transport,
            "assets_for_gpu": getattr(self.gpu.backend, "internal_base", None),
            "slots": self.gpu.slots,
            "pool": self.gpu.snapshot(),
            "endpoints": await self.gpu.health(),
            "active_sessions": len(self.runtimes),
            "models": {
                "worldsmith": settings.model_worldsmith,
                "director": settings.model_director,
                "promptir": settings.model_promptir,
                "keyframe": settings.keyframe_model,
            },
        }

    async def shutdown(self) -> None:
        for rt in self.runtimes.values():
            for task in list(rt.tasks):
                task.cancel()
        await asyncio.sleep(0)


def _have_keyframe(beat: Beat) -> bool:
    """Whether this beat's own opening image is on disk and usable *now*.

    The `exists` check is not paranoia. The path can be recorded and the file be
    gone in two ordinary cases: a session resumed after `ASSETS_DIR` was cleared,
    and a prepared keyframe outliving a restart of this process. Returning True
    for either would hand a missing path to the uploader, which fails the beat --
    where re-drawing costs 6.3s and works.
    """
    return bool(beat.keyframe_path) and Path(beat.keyframe_path or "").exists()


# --------------------------------------------------------------------------- #
# Event payload shaping                                                       #
# --------------------------------------------------------------------------- #


def _option_summary(intent: BranchIntent) -> dict[str, object]:
    return {
        "label": intent.label,
        "consequence_hint": intent.consequence_hint,
        "transition": intent.transition,
        "shot_type": intent.shot.type,
        "references_state": intent.references_state,
    }


def _beat_summary(beat: Beat) -> dict[str, object]:
    """What the browser needs, and nothing more.

    Local filesystem paths deliberately do not appear here -- they are useless to
    a browser and they leak the GPU box's layout.
    """
    return {
        "id": beat.id,
        "parent_id": beat.parent_id,
        "index": beat.index,
        "origin": beat.origin,
        "label": beat.label,
        "status": beat.status.value,
        "error": beat.error,
        "degraded": beat.degraded,
        "video_url": beat.video_url,
        "poster_url": beat.poster_url,
        "last_frame_url": beat.last_frame_url,
        "keyframe_url": beat.keyframe_url,
        # Both for the debug panel, and both cheap: a string the model already
        # produced and a list that is empty on the healthy path.
        "keyframe_prompt": beat.keyframe_prompt,
        "ir_violations": beat.ir_violations,
        "duration_ms": beat.duration_ms,
        "has_audio": beat.has_audio,
        "narration": beat.narration,
        "transition": beat.intent.transition,
        "shot_type": beat.intent.shot.type,
        "ir_source": beat.ir_source,
        "options": [_option_summary(o) for o in beat.options],
        "children": dict(beat.children),
        "predicted_choice": beat.predicted_choice,
        "is_ending": beat.is_ending.model_dump(mode="json") if beat.is_ending else None,
        "drift": beat.drift,
        "timings": beat.timings,
        "state": _state_summary(beat.state_after),
    }


def session_view(session: Session) -> dict[str, object]:
    """The whole session, as the browser sees it.

    Fetched once on load or reconnect; after that the SSE stream carries deltas.
    """
    return {
        "id": session.id,
        "phase": session.phase.value,
        "error": session.error,
        "premise": session.premise,
        "genre": session.genre,
        "pov": session.pov,
        "bible": session.bible.model_dump(mode="json") if session.bible else None,
        # For the debug panel. Both are session-scoped and neither has anywhere
        # else to live: `timings` is the Worldsmith's, which belongs to no beat,
        # and the notes explain the bible sitting next to them.
        "bible_notes": session.bible_notes,
        "timings": session.timings,
        "opening_keyframe_url": session.opening_keyframe_url,
        "root_id": session.root_id,
        "cursor": session.cursor,
        "path": session.path,
        "beats": {bid: _beat_summary(b) for bid, b in session.beats.items()},
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


def _state_summary(state: WorldState) -> dict[str, object]:
    return {
        "beat_index": state.beat_index,
        "act": state.act,
        "location": state.location,
        "time_of_day": state.time_of_day,
        "elapsed_in_world": state.elapsed_in_world,
        "inventory": state.inventory,
        "stats": state.stats,
        "flags": state.flags,
        "tension": state.tension,
        "present_characters": state.present_characters,
    }
