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
    """The three sections of MiniMax H3's documented IR format.

    PromptIR emits these separately; H3 wants them joined by a single newline.
    Joining here (rather than in the orchestrator) keeps the separator rule in
    one place, since getting it wrong is a silent quality regression.
    """

    description: str = Field(..., description="integrated_multimodal_description")
    soundscape: str | None = Field(None, description="overall_soundscape")
    music: str | None = Field(None, description="non_diegetic_music")

    def to_prompt(self) -> str:
        parts = [self.description]
        if self.soundscape:
            parts.append(self.soundscape)
        if self.music:
            parts.append(self.music)
        return "\n".join(p.strip() for p in parts)


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

    def resolved_prompt(self) -> str:
        return self.ir.to_prompt() if self.ir else (self.prompt or "")


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
