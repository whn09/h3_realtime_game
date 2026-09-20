"""PromptIR: compile a shot intent into H3's IR.

Two properties this module guarantees to its caller:

* **It always returns an IR.** Never raises for content reasons. If the LLM is
  slow, unavailable, or cannot satisfy the validator, `template_ir()` assembles a
  compliant IR from the Director's structured fields. DESIGN.md section 2.3 calls
  this fail-open, and the reasoning is that this is a game: a slightly flatter
  beat beats no beat. The only fail-closed thing here is quality reporting --
  `source` and `violations` always say which path produced the result.

* **The frozen text stays frozen.** The style anchor and every on-screen
  character's appearance are inserted verbatim, and the validator checks they
  survived. That verbatim reuse is the entire anti-drift mechanism
  (DESIGN.md section 3.3); a paraphrase silently disables it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from pydantic import BaseModel

from . import ir_validator as V
from . import prompts
from .config import settings
from .llm import LLM, LLMError
from .schema import BranchIntent, Character, IRSections, ShotSpec, WorldBible, WorldState

log = logging.getLogger("kunlun.promptir")

_CAMERA = {
    "wide": "大远景，广角，固定镜头",
    "medium": "中景，固定镜头",
    "closeup": "特写，固定镜头",
    "pov": "第一人称主观视角，轻微手持",
    "tracking": "跟拍，镜头随主体缓慢横移",
    "aerial": "航拍俯视，缓慢下降",
}


class _IROut(BaseModel):
    description: str
    soundscape: str | None = None
    music: str | None = None


@dataclass
class CompiledIR:
    ir: IRSections
    source: str                                  # llm | repaired | template
    violations: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    attempts: int = 0


def characters_in_shot(bible: WorldBible, shot: ShotSpec) -> list[Character]:
    """Only the characters actually on screen get their appearance injected.

    Injecting everyone present in the scene would bloat the IR and, worse,
    describe people the shot never frames -- which invites H3 to put them in it.
    """
    haystack = " ".join([shot.subject, shot.action, shot.setting])
    speakers = {d.get("speaker", "") for d in shot.dialogue}
    out: list[Character] = []
    for ch in bible.characters:
        if ch.id in speakers or ch.name in speakers or ch.name in haystack or ch.id in haystack:
            out.append(ch)
    return out


def _sentence(text: str) -> str:
    """End `text` with exactly one full stop."""
    t = text.rstrip()
    return t if t.endswith(("。", "！", "？")) else t + "。"


def _action_clause(shot: ShotSpec) -> str:
    """`subject` + `action`, without saying the subject twice.

    The Director writes `action` as a verb phrase, but it often opens by naming
    the subject again -- which produced `跌坐在旁的林清枢林清枢缓缓张开…` on the
    first run. Detected rather than forbidden, because the Director naming its
    own subject is natural writing and not worth a repair round trip.
    """
    subject, action = shot.subject.strip(), shot.action.strip()
    if subject and action.startswith(subject):
        return action
    return f"{subject}{action}"


def append_style_anchor(description: str, style_anchor: str) -> str:
    """Append the frozen visual grammar in code rather than asking for a copy.

    Measured on the first end-to-end run: asked to retype a 130-character anchor
    verbatim, Haiku 4.5 reproduced 48-81% of it and reliably dropped the lens
    clause (`镜头多用35mm定焦，人物特写时切50mm`) -- presumably because the IR's
    opening already states the camera, so re-stating focal lengths reads
    contradictory. That cost a fatal violation, a repair round trip, and sometimes
    a fall back to the template.

    The anchor is frozen text and a compiler owns frozen text. Appending it here
    makes the anti-drift guarantee absolute instead of probabilistic, and saves
    roughly 90 decode tokens per beat -- which the beat budget notices.
    """
    anchor = style_anchor.strip()
    if not anchor:
        return description
    body = description.rstrip()
    if V.normalise(anchor) in V.normalise(body):
        return body
    if body and body[-1] not in "。！？.!?":
        body += "。"
    return body + anchor


def _shot_block(intent: BranchIntent, state: WorldState) -> str:
    shot = intent.shot
    rows = [
        f"机位：{shot.type}（{_CAMERA.get(shot.type, '中景')}）",
        f"主体：{shot.subject}",
        f"动作（只有这一个动作）：{shot.action}",
        f"环境：{shot.setting or state.location}",
        f"情绪：{shot.mood}",
        f"最突出的音效：{shot.sfx_focus or '（无特别指定）'}",
        f"衔接方式：{intent.transition}",
    ]
    if shot.dialogue:
        lines = "；".join(f"{d.get('speaker', '?')}：「{d.get('line', '')}」" for d in shot.dialogue)
        rows.append(f"对白：{lines}")
    else:
        rows.append("对白：无（保持沉默）")
    return "\n".join(rows)


def _characters_block(chars: list[Character]) -> str:
    return "\n".join(
        f"- {c.name}（id={c.id}，音色：{c.voice or '未指定'}）：{c.appearance}" for c in chars
    )


def _continuity_block(intent: BranchIntent, state: WorldState) -> str:
    if intent.transition == "continuous":
        return (
            "衔接要求：这一段的第一帧就是上一段的最后一帧，属于同一个连续镜头的延续。"
            "开场姿态必须承接上一拍的结束姿态，不要重新建立场景、不要重新介绍环境。"
        )
    if intent.transition == "timeskip":
        return (
            f"衔接要求：这是一次时间跳跃（{state.elapsed_in_world or '一段时间之后'}）。"
            "可以重新建立场景，但地点、角色外观与视觉风格必须与之前一致。"
        )
    return (
        "衔接要求：这是一次硬切，换了地点或视角。可以重新建立空间，"
        "但角色外观与视觉风格必须与之前完全一致。"
    )


def template_ir(bible: WorldBible, intent: BranchIntent, state: WorldState) -> IRSections:
    """Assemble a compliant IR without an LLM.

    Used as the timeout/failure fallback, and useful on its own as a latency
    floor measurement: whatever this produces is the worst the pictures get.

    Written to satisfy the validator by construction -- camera vocabulary,
    verbatim style anchor, verbatim appearances, no markdown, no meta language.
    """
    shot = intent.shot
    chars = characters_in_shot(bible, shot)
    camera = _CAMERA.get(shot.type, "中景，固定镜头")

    bits = [f"{camera}。"]
    if chars:
        for c in chars:
            # `appearance` is authored by the Worldsmith and usually already ends
            # in a full stop; `_sentence` keeps us from emitting `符。。画面中的`.
            bits.append(_sentence(f"画面中的{c.name}：{c.appearance}"))
    else:
        bits.append(f"画面主体是{shot.subject}。")

    bits.append(
        f"场景位于{shot.setting or state.location or '原地'}，"
        f"{state.time_of_day or '光线自然'}。"
    )
    bits.append(
        f"在这{int(round(settings.beat_seconds))}秒里，{_action_clause(shot)}；"
        "动作从起始姿态开始，匀速完成，在末尾停住并保持住这个姿态。"
    )
    if shot.mood:
        bits.append(f"整体氛围{shot.mood}。")
    for d in shot.dialogue[:2]:
        speaker = d.get("speaker", "")
        ch = bible.character(speaker)
        name = ch.name if ch else speaker
        voice = f"（{ch.voice}）" if ch and ch.voice else ""
        bits.append(f"{name}{voice}说：「{d.get('line', '')}」。")

    description = append_style_anchor("".join(bits).replace("\n", ""), bible.style_anchor)
    return IRSections(
        description=description,
        soundscape=prompts.template_soundscape(bible.ambience, shot.sfx_focus, shot.setting),
        music=prompts.template_music(bible.music_bible, state.tension),
    )


class PromptIR:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def compile(
        self,
        *,
        bible: WorldBible,
        state: WorldState,
        intent: BranchIntent,
        seconds: float | None = None,
    ) -> CompiledIR:
        seconds = seconds or settings.beat_seconds
        desc_only = settings.ir_template_tail
        chars = characters_in_shot(bible, intent.shot)
        budget = V.dialogue_budget_for(seconds)

        ctx = V.IRContext(
            style_anchor=bible.style_anchor,
            music_bible=bible.music_bible,
            characters=chars,
            dialogue_budget=budget,
            seconds=seconds,
            # When the tail is templated we build it ourselves, so there is
            # nothing for the model to omit.
            require_tail=not desc_only,
        )

        system = prompts.PROMPTIR_SYSTEM + (
            prompts.PROMPTIR_OUTPUT_DESC_ONLY if desc_only else prompts.PROMPTIR_OUTPUT_FULL
        )
        base_user = prompts.promptir_user(
            shot_block=_shot_block(intent, state),
            characters_block=_characters_block(chars),
            style_anchor=bible.style_anchor,
            music_bible=bible.music_bible,
            ambience=bible.ambience,
            continuity_block=_continuity_block(intent, state),
            dialogue_budget=budget,
            seconds=seconds,
            desc_only=desc_only,
        )

        timings: dict[str, float] = {}
        started = time.perf_counter()
        user = base_user
        attempts = 0
        last_violations: list[V.Violation] = []

        while attempts <= settings.ir_max_repairs:
            attempts += 1
            try:
                out, meta = await self.llm.structured(
                    settings.model_promptir,
                    system,
                    user,
                    _IROut,
                    max_tokens=settings.ir_max_tokens,
                    temperature=0.6,
                )
            except LLMError as exc:
                log.warning("PromptIR LLM failed (attempt %d): %s", attempts, exc)
                break

            timings.setdefault("promptir_ttft_ms", round(meta.ttft_ms, 1))
            timings["promptir_total_ms"] = round(
                timings.get("promptir_total_ms", 0.0) + meta.total_ms, 1
            )
            timings["promptir_out_tokens"] = timings.get("promptir_out_tokens", 0) + meta.output_tokens

            ir = IRSections(
                # The model writes the shot; the compiler owns the frozen text.
                description=append_style_anchor(
                    out.description.strip().replace("\n", ""), bible.style_anchor
                ),
                soundscape=(out.soundscape or "").strip() or None,
                music=(out.music or "").strip() or None,
            )
            if desc_only:
                ir.soundscape = prompts.template_soundscape(
                    bible.ambience, intent.shot.sfx_focus, intent.shot.setting
                )
                ir.music = prompts.template_music(bible.music_bible, state.tension)

            violations = V.validate(ir, ctx)
            last_violations = violations
            fatal = V.fatals(violations)
            if not fatal:
                timings["promptir_wall_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
                return CompiledIR(
                    ir=ir,
                    source="llm" if attempts == 1 else "repaired",
                    violations=[str(v) for v in violations],
                    timings=timings,
                    attempts=attempts,
                )

            log.info("IR rejected (attempt %d): %s", attempts, "; ".join(str(v) for v in fatal))
            user = base_user + "\n\n" + V.repair_instruction(fatal) + "\n\n重新编译，只输出 JSON。"

        # Fail-open. A flatter beat is a beat; a raised exception is a black screen.
        ir = template_ir(bible, intent, state)
        residual = V.validate(ir, ctx)
        if V.fatals(residual):
            # The template is built to be compliant, so this means a rule and the
            # builder disagree -- worth shouting about, but still not worth
            # failing the beat over.
            log.error("template IR also violates rules: %s", [str(v) for v in V.fatals(residual)])
        timings["promptir_wall_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        return CompiledIR(
            ir=ir,
            source="template",
            violations=[str(v) for v in last_violations] + ["fell back to template IR"],
            timings=timings,
            attempts=attempts,
        )
