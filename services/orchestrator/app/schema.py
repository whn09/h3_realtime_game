"""The data model for a session: world bible, mutable state, and the story tree.

Two conventions worth knowing before reading further:

1. **Everything is snake_case**, including the JSON the LLMs are asked to emit.
   One casing convention end to end removes a whole class of silent key-miss
   bugs where a field parses as absent and the default quietly wins.

2. **State changes are expressed as an explicit delta, never as a partial copy
   of the state.** DESIGN.md sketched `Partial<WorldState>`, but asking a model
   for a partial object invites it to re-emit `inventory` or `flags` wholesale
   and silently drop entries it did not think to repeat. `StateDelta` below only
   lets a branch say what it *adds, removes or sets*, so forgetting a field is
   a no-op instead of data loss.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Pov = Literal["first", "third"]
Transition = Literal["continuous", "cut", "timeskip"]
ShotType = Literal["wide", "medium", "closeup", "pov", "tracking", "aerial"]


# --------------------------------------------------------------------------- #
# IR -- mirrors h3-wrapper's contract                                          #
# --------------------------------------------------------------------------- #


class IRSections(BaseModel):
    """Mirror of h3-wrapper's `IRSections`. Kept as a separate declaration
    because these are separate services; the wrapper owns the joining rule."""

    description: str
    soundscape: str | None = None
    music: str | None = None

    def to_prompt(self) -> str:
        parts = [self.description]
        if self.soundscape:
            parts.append(self.soundscape)
        if self.music:
            parts.append(self.music)
        return "\n".join(p.strip() for p in parts)


# --------------------------------------------------------------------------- #
# World bible -- frozen at creation                                            #
# --------------------------------------------------------------------------- #


class Character(BaseModel):
    id: str
    name: str
    # Reused verbatim in every IR that features this character. The Director is
    # forbidden from rewriting it; that verbatim reuse is the main defence
    # against character drift (DESIGN.md section 3.3).
    appearance: str
    # The same appearance in English, for the image model and only the image
    # model. Two languages rather than one translation at call time because the
    # two consumers are genuinely different: H3 reads the Chinese IR, and Bedrock's
    # SD3.5 is measurably more literal with English -- and a keyframe is the one
    # frame in a beat that has no previous frame to inherit a face from, so it is
    # exactly where a diluted prompt costs an identity.
    #
    # Empty is tolerated: nothing here can translate, so a bible that omits it
    # degrades to the pre-existing behaviour (the name alone) rather than to a
    # half-Chinese prompt.
    appearance_en: str = ""
    voice: str = ""
    arc: str = ""


class ActOutline(BaseModel):
    act: int = Field(..., ge=1, le=3)
    milestone: str
    target_beats: int = Field(6, ge=1, le=40)

    @field_validator("target_beats", mode="before")
    @classmethod
    def _count_if_listed(cls, v: object) -> object:
        """A name like `target_beats` reads as "the beats targeted", and models
        routinely answer with the list of beats rather than how many. The count of
        that list is exactly the number being asked for, so take it instead of
        failing the whole bible over a naming ambiguity."""
        if isinstance(v, (list, tuple)):
            return len(v)
        return v


class ShotSpec(BaseModel):
    """One shot: exactly one action, playable inside a single beat."""

    type: ShotType = "medium"
    subject: str
    action: str
    setting: str
    mood: str = ""
    dialogue: list[dict[str, str]] = Field(default_factory=list)  # {speaker, line}
    sfx_focus: str = ""


class WorldBible(BaseModel):
    premise: str
    genre: str = ""
    logline: str = ""
    # Frozen visual grammar (film stock, lens, lighting ratio, palette),
    # injected verbatim into every IR.
    style_anchor: str = ""
    # The same grammar in English, for the image model. `build_prompt` used to
    # inject the Chinese anchor into a prompt whose docstring says it is English on
    # purpose, which is the worst of both: SD3.5's encoders get ~130 characters
    # they largely cannot use, and they get them in the slot where the visual style
    # was supposed to be pinned. Falls back to the Chinese one when absent, since
    # that is no worse than what it replaced.
    style_anchor_en: str = ""
    # Frozen musical grammar (BPM, key, instrumentation, emotional baseline).
    music_bible: str = ""
    ambience: str = ""
    pov: Pov = "third"
    protagonist_id: str = ""
    characters: list[Character] = Field(default_factory=list)
    world_rules: list[str] = Field(default_factory=list)
    outline: list[ActOutline] = Field(default_factory=list)
    stat_names: list[str] = Field(default_factory=list)
    opening: ShotSpec | None = None
    opening_keyframe_prompt: str = ""

    def character(self, cid: str) -> Character | None:
        for c in self.characters:
            if c.id == cid or c.name == cid:
                return c
        return None


# --------------------------------------------------------------------------- #
# Mutable state                                                                #
# --------------------------------------------------------------------------- #


class WorldState(BaseModel):
    beat_index: int = 0
    act: int = 1
    beats_in_act: int = 0
    location: str = ""
    time_of_day: str = ""
    elapsed_in_world: str = ""
    present_characters: list[str] = Field(default_factory=list)
    inventory: list[str] = Field(default_factory=list)
    stats: dict[str, float] = Field(default_factory=dict)
    # The consequence system. Every Director branch must reference at least one
    # existing flag or stat, otherwise a "choice" is just decoration.
    flags: dict[str, Any] = Field(default_factory=dict)
    tension: float = Field(0.3, ge=0.0, le=1.0)
    summary: str = ""
    recent_beats: list[str] = Field(default_factory=list)
    recent_shot_types: list[str] = Field(default_factory=list)
    # Set when drift crosses threshold or the re-anchor interval elapses; the
    # next beat is then forced to `cut` with a freshly generated keyframe.
    needs_reanchor: bool = False
    beats_since_anchor: int = 0
    # How many consecutive beats have played in `location`. Pacing pressure, and
    # deliberately separate from `beats_in_act`: an act can legitimately run 6-7
    # beats, but 6 beats without leaving the room is 90 seconds of one set, which
    # reads as the story having stalled even while the plot is technically moving.
    # The Director cannot see this by itself -- it gets one state snapshot per
    # call and nothing in it says "you have been here a while".
    beats_in_scene: int = 0

    def apply(self, delta: "StateDelta") -> "WorldState":
        """Return a new state with the delta applied. Never mutates in place --
        story-tree siblings share a parent state and must not see each other's
        consequences."""
        nxt = self.model_copy(deep=True)
        nxt.beat_index = self.beat_index + 1
        nxt.beats_in_act = self.beats_in_act + 1
        nxt.beats_since_anchor = self.beats_since_anchor + 1

        nxt.beats_in_scene = self.beats_in_scene + 1
        if delta.location:
            nxt.location = delta.location
            if delta.location != self.location:
                nxt.beats_in_scene = 0
        if delta.time_of_day:
            nxt.time_of_day = delta.time_of_day
        if delta.elapsed_in_world:
            nxt.elapsed_in_world = delta.elapsed_in_world
            # A time jump is a scene change even without moving: the same room
            # three hours later is a new set as far as the audience is concerned,
            # and the Director should not be pushed to relocate on top of it.
            nxt.beats_in_scene = 0
        if delta.present_characters is not None:
            nxt.present_characters = list(delta.present_characters)
        if delta.act and delta.act != nxt.act:
            nxt.act = delta.act
            nxt.beats_in_act = 0
        if delta.tension is not None:
            nxt.tension = max(0.0, min(1.0, delta.tension))

        for item in delta.inventory_add:
            if item not in nxt.inventory:
                nxt.inventory.append(item)
        for item in delta.inventory_remove:
            if item in nxt.inventory:
                nxt.inventory.remove(item)
        for key, amount in delta.stats_delta.items():
            nxt.stats[key] = round(nxt.stats.get(key, 0.0) + amount, 3)
        nxt.flags.update(delta.flags_set)

        if delta.summary_append:
            nxt.recent_beats = (nxt.recent_beats + [delta.summary_append])[-3:]
        return nxt


class StateDelta(BaseModel):
    location: str | None = None
    time_of_day: str | None = None
    elapsed_in_world: str | None = None
    present_characters: list[str] | None = None
    act: int | None = None
    tension: float | None = None
    inventory_add: list[str] = Field(default_factory=list)
    inventory_remove: list[str] = Field(default_factory=list)
    stats_delta: dict[str, float] = Field(default_factory=dict)
    flags_set: dict[str, Any] = Field(default_factory=dict)
    # One line of "what just happened", used to build recent_beats and, every
    # few beats, folded into the compressed summary.
    summary_append: str | None = None


# --------------------------------------------------------------------------- #
# Director output                                                              #
# --------------------------------------------------------------------------- #


class BranchIntent(BaseModel):
    label: str                      # option text, verb-first, short
    consequence_hint: str = ""      # vague cost, must not spoil
    transition: Transition = "continuous"
    shot: ShotSpec
    state_delta: StateDelta = Field(default_factory=StateDelta)
    # Which existing flag or stat this branch turns on. Required by the Director
    # prompt; recorded so we can audit whether choices actually have weight.
    references_state: str = ""


class Ending(BaseModel):
    type: str
    title: str
    epilogue: str = ""


class DirectorOutput(BaseModel):
    narration: str = ""
    options: list[BranchIntent] = Field(default_factory=list)
    predicted_choice: int = 0
    is_ending: Ending | None = None


# --------------------------------------------------------------------------- #
# Story tree                                                                   #
# --------------------------------------------------------------------------- #


class BeatStatus(str, Enum):
    PLANNED = "planned"        # intent known, nothing started
    COMPILING = "compiling"    # PromptIR running
    WAITING_FRAME = "waiting_frame"  # blocked on the parent's last frame
    QUEUED = "queued"          # waiting for a GPU slot
    GENERATING = "generating"  # H3 running
    READY = "ready"
    FAILED = "failed"


class Beat(BaseModel):
    id: str
    parent_id: str | None = None
    index: int = 0
    origin: Literal["opening", "choice", "custom"] = "choice"
    label: str = ""
    intent: BranchIntent

    status: BeatStatus = BeatStatus.PLANNED
    error: str | None = None
    degraded: bool = False          # produced by a fallback path

    ir: IRSections | None = None
    ir_source: Literal["llm", "template", "repaired"] | None = None
    ir_violations: list[str] = Field(default_factory=list)

    keyframe_url: str | None = None
    video_url: str | None = None
    poster_url: str | None = None
    last_frame_url: str | None = None
    last_frame_path: str | None = None
    # Replica alias -> a path on that box which already holds `last_frame_path`'s
    # image, because that is the box this clip was generated on. A continuous
    # child that happens to land on the same replica conditions straight off it
    # and moves no bytes at all. Empty on any other backend, and empty whenever
    # the on-box extraction failed -- in both cases the local file is uploaded as
    # before, so nothing depends on this being populated.
    remote_last_frame: dict[str, str] = Field(default_factory=dict)
    duration_ms: float | None = None
    has_audio: bool = False

    frame_stats: dict[str, float] | None = None
    drift: dict[str, float] | None = None
    # Fingerprint this beat's drift is measured against: the last frame of the
    # nearest ancestor that started from a fresh keyframe (or itself, if it did).
    # Inherited unchanged down a continuous chain, which is what makes `drift`
    # mean "how far has the chain walked" rather than "how unlike the opening
    # shot is this shot". None until the beat has been fingerprinted.
    drift_anchor: dict[str, float] | None = None

    state_after: WorldState = Field(default_factory=WorldState)
    narration: str = ""
    options: list[BranchIntent] = Field(default_factory=list)
    predicted_choice: int = 0
    is_ending: Ending | None = None
    # option index (as a string, so this survives a JSON round-trip) -> beat id
    children: dict[str, str] = Field(default_factory=dict)

    timings: dict[str, float] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    @property
    def playable(self) -> bool:
        return self.status is BeatStatus.READY and bool(self.video_url)


class SessionPhase(str, Enum):
    CREATING = "creating"        # worldsmith running
    OPENING = "opening"          # bible ready, first beat being generated
    PLAYING = "playing"
    ENDED = "ended"
    FAILED = "failed"


class Session(BaseModel):
    id: str
    phase: SessionPhase = SessionPhase.CREATING
    error: str | None = None
    premise: str = ""
    genre: str = ""
    pov: Pov = "third"
    bible: WorldBible | None = None
    opening_keyframe_url: str | None = None
    beats: dict[str, Beat] = Field(default_factory=dict)
    root_id: str | None = None
    cursor: str | None = None            # the beat the player is on
    path: list[str] = Field(default_factory=list)  # root -> cursor
    # Fingerprint of the first beat's last frame: the session's reference look,
    # and the anchor of last resort for a beat whose parent was never
    # fingerprinted. Not what drift is normally measured against -- see
    # `Beat.drift_anchor`.
    drift_baseline: dict[str, float] | None = None
    # Session-scoped latencies, as opposed to `Beat.timings` which are per-clip.
    # Only the Worldsmith lands here, and it earns the field: it is a single
    # 70s-class call that gates the whole opening, because the root beat's keyframe
    # prompt is part of its output and so nothing visual can start before it
    # returns. Storing it is what makes that number appear in the same place as
    # every other measurement rather than having to be inferred from the gap
    # between `created_at` and the root beat's.
    timings: dict[str, float] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def beat(self, beat_id: str | None) -> Beat | None:
        return self.beats.get(beat_id) if beat_id else None

    @property
    def cursor_beat(self) -> Beat | None:
        return self.beat(self.cursor)
