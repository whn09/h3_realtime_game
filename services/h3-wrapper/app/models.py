"""Request/response contracts for h3-wrapper.

This is the seam between the orchestrator and the GPU box. Keep it stable: the
orchestrator should never need to know SGLang's schema or where files land.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Task = Literal["t2va", "fl2va", "ref2va"]
Quality = Literal["lossless", "extra-high", "high"]
ConditionType = Literal["image", "video", "audio", "video_audio"]
ConditionRole = Literal["keyframe", "reference"]


class IRSections(BaseModel):
    """The three core fields of MiniMax H3's documented prompt format.

    PromptIR emits them separately; this is where they become the string the text
    tower reads. The assembly rule lives here rather than in the orchestrator
    because getting it wrong is a silent quality regression -- SGLang tokenizes
    the `prompt` verbatim (no chat template, no rewriter, no length
    normalisation), so **whatever we send IS the IR**.

    The shape is not ours to choose. `docs/h3official/base-en.txt` section 2.2,
    which is `references/base-en.txt` of MiniMax's own `h3-prompt-writing` skill:

        integrated_multimodal_description: [Shot 1] ...
        <blank>
        overall_soundscape: ...
        <blank>
        non_diegetic_music: ...

    Three things about that are load-bearing and were all missing before:

    * **The field names.** Without them H3 reads three unlabelled paragraphs and
      has to guess which is which; the audio branch is conditioned by the same
      text as the video branch, so a mislabelled section is a mis-scored clip.
    * **The blank line.** A single `\\n` is what separates sentences inside a
      field, so joining fields with one merges them.
    * **All three fields, always.** A prompt that omits `overall_soundscape` or
      `non_diegetic_music` does not get silence -- it gets whatever the model
      invents. `N/A` is the documented way to ask for nothing (sections 4.6/4.7),
      so an absent section becomes an explicit `N/A` rather than an absent field.
    """

    description: str = Field(..., description="integrated_multimodal_description")
    soundscape: str | None = Field(None, description="overall_soundscape")
    music: str | None = Field(None, description="non_diegetic_music")

    def core_fields(self) -> str:
        """Part two of the final prompt: the three labelled fields, in order."""
        return "\n\n".join(
            f"{label}: {(text or '').strip() or 'N/A'}"
            for label, text in (
                ("integrated_multimodal_description", self.description),
                ("overall_soundscape", self.soundscape),
                ("non_diegetic_music", self.music),
            )
        )


class ReferenceSpec(BaseModel):
    """A ref2va reference. `source` accepts a local path, http(s) URL or data: URI."""

    type: ConditionType = "image"
    source: str
    start_time_seconds: float | None = None


class GenerateRequest(BaseModel):
    job_id: str | None = None

    # Supply either structured IR or a raw prompt string, not both.
    ir: IRSections | None = None
    prompt: str | None = None

    task: Task = "fl2va"
    seconds: float = Field(15.0, ge=4.0, le=15.0)

    # Continuity conditioning. `first_frame` is the previous beat's last frame;
    # this is what makes consecutive clips read as one continuous world.
    first_frame: str | None = None
    last_frame: str | None = None
    references: list[ReferenceSpec] = Field(default_factory=list)

    short_edge: int = 480
    aspect_ratio: str = "16:9"
    quality: Quality = "high"
    num_inference_steps: int = 50
    flow_shift: float | None = None
    audio_flow_shift: float | None = None
    seed: int | None = None
    num_outputs_per_prompt: int = Field(1, ge=1, le=10)

    # Post-processing toggles. All default on for gameplay; benchmarks turn the
    # expensive ones off to isolate pure model latency.
    extract_last_frame: bool = True
    extract_poster: bool = True
    faststart: bool = True
    analyze: bool = True
    archive: bool = True

    # Escape hatch: merged verbatim into the SGLang payload. Lets us try
    # undocumented knobs during P0 without editing the wrapper.
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_a_prompt(self) -> "GenerateRequest":
        if not self.ir and not self.prompt:
            raise ValueError("one of `ir` or `prompt` is required")
        if self.ir and self.prompt:
            raise ValueError("`ir` and `prompt` are mutually exclusive")
        return self

    def alignment_instruction(self) -> str | None:
        """Part one of the final prompt: the keyframe-alignment instruction.

        Quoted verbatim from `docs/h3official/base-en.txt` section 2.1, which is
        emphatic about the placement -- *"The instruction must be the first line of
        the final prompt, followed by one blank line before the core fields."*
        T2VA has no instruction at all.

        Chosen by **which frames are actually attached**, not by `task`. The two
        are not the same question: every beat this game generates is sent as
        `fl2va` (that is what the replicas serve) while attaching only a first
        frame, and emitting the FL2VA line there would promise a `Picture 2` that
        does not exist -- an unresolved reference label, which the skill's own
        output rules list as a thing to avoid. One picture at 0.00s *is* the I2VA
        case, so it gets the I2VA line.

        `S.SS` is the effective duration to exactly two decimal places, and `N`
        the index of the final shot -- 1 here, because a beat is one shot by
        construction (one action, no cuts).
        """
        s = f"{self.seconds:.2f}"
        if self.first_frame and self.last_frame:
            return (
                "How the reference pictures align with the target video — "
                "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
                f"Picture 2 (from Shot 1) aligns with the {s}-second mark of the target video."
            )
        if self.first_frame:
            return (
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced."
            )
        if self.last_frame:
            return (
                "How the reference pictures align with the target video — "
                f"<Picture 1> (from [Shot 1]) aligns with the {s}-second mark of the target video."
            )
        return None

    def resolved_prompt(self) -> str:
        """The exact string SGLang will tokenize.

        `prompt` is the escape hatch and is passed through untouched: a caller
        that hands us a finished prompt (the benches, a ref2va experiment) has
        already made every formatting decision, and re-wrapping it would silently
        edit an input meant to be exact.
        """
        if self.prompt is not None:
            return self.prompt
        if self.ir is None:
            return ""
        instruction = self.alignment_instruction()
        core = self.ir.core_fields()
        return f"{instruction}\n\n{core}" if instruction else core


class Timings(BaseModel):
    """Per-stage wall clock, in ms.

    Every field here exists because the 19.3s budget in DESIGN.md is the
    project's hard constraint -- without per-stage numbers there is no way to
    know which stage to attack.
    """

    total_ms: float = 0.0
    slot_wait_ms: float = 0.0
    input_fetch_ms: float = 0.0
    sglang_ms: float = 0.0
    download_ms: float = 0.0
    probe_ms: float = 0.0
    last_frame_ms: float = 0.0
    poster_ms: float = 0.0
    faststart_ms: float = 0.0
    analyze_ms: float = 0.0
    publish_ms: float = 0.0


class MediaInfo(BaseModel):
    width: int | None = None
    height: int | None = None
    duration_ms: float | None = None
    fps: float | None = None
    nb_frames: int | None = None
    video_codec: str | None = None
    has_audio: bool = False
    audio_codec: str | None = None
    audio_sample_rate: int | None = None
    size_bytes: int | None = None


class FrameStats(BaseModel):
    """Cheap perceptual fingerprint of the last frame.

    Chaining last-frame -> first-frame accumulates colour cast and softness.
    Tracking these against the session baseline is what triggers a
    re-anchoring `cut` (DESIGN.md section 3.3) before drift becomes visible.
    """

    mean_r: float
    mean_g: float
    mean_b: float
    luma: float
    saturation: float
    contrast: float
    sharpness: float


class GenerateResponse(BaseModel):
    job_id: str
    # The exact string that was tokenized, assembled instruction line and all.
    # Returned so the caller archives the bytes H3 actually saw rather than its own
    # guess at what this service would build from the same IR -- the two live in
    # different repos and the only way to keep them honest is to compare them.
    prompt: str = ""
    video_url: str
    video_path: str
    last_frame_url: str | None = None
    last_frame_path: str | None = None
    poster_url: str | None = None
    media: MediaInfo
    frame_stats: FrameStats | None = None
    timings: Timings
    sglang_request: dict[str, Any] | None = None
    sglang_raw: Any | None = None
