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

log = logging.getLogger("h3game.promptir")

SHOT_MARKER = "[Shot 1] "

# Framing + camera motion in the vocabulary of `docs/h3official/base-en.txt`
# section 4.3, which is a closed list: motion type + amplitude + speed. Anything
# outside it is a phrase H3 was not trained to read as camera instruction.
# Static wins the ties on purpose -- a held frame is measurably steadier than a
# moving one, and this is a 15-second beat that has to end on a usable last frame.
_CAMERA = {
    "wide": "a wide shot, and the camera holds a static shot",
    "medium": "a medium shot, and the camera holds a static shot",
    "closeup": "a close-up, and the camera holds a static shot",
    "pov": "a POV shot that shakes slightly with small amplitude",
    "tracking": "a tracking shot, and the camera trucks with small amplitude at slow speed",
    "aerial": "a high-angle aerial shot, and the camera pedestals down at slow speed",
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
    return t if t.endswith((".", "!", "?", "。", "！", "？")) else t + "."


def speaker_ids(bible: WorldBible) -> dict[str, str]:
    """Assign each character a stable `(S1)`-style ID for the whole session.

    `docs/h3official/base-en.txt` section 4.4 requires that *"a speaker keeps the
    same ID across shots"*, which means the numbering cannot be derived from who
    happens to be in this shot -- it has to come from the bible, which is frozen.
    The protagonist takes S1 so the most-heard voice is also the most-conditioned
    one; the rest follow the bible's own order.
    """
    ordered = sorted(
        bible.characters, key=lambda c: (c.id != bible.protagonist_id,)
    )
    return {c.id: f"S{i}" for i, c in enumerate(ordered, start=1)}


def dialogue_clauses(bible: WorldBible, shot: ShotSpec) -> list[str]:
    """Build the `<d>`-wrapped dialogue clauses, verbatim, in code.

    This is the format deviation with a *measured* effect, and the reason the game
    was producing unintelligible speech: written as `林清枢（清冷的女声）说：「…」`,
    the spoken words are just more scene description, and H3's audio branch
    renders them as vocal-shaped noise. Section 4.4's syntax is what marks them as
    words to utter:

        The young woman with a quiet, breathy voice (S1) says: <d>[Chinese] 台词</d>

    Everything about that line is load-bearing. The speaker's identifying phrase,
    the ID and the delivery go **outside** `<d>`; inside it there is only the
    language tag and the words themselves, preserved *"verbatim; do not translate
    or rewrite them"*. So the compiler assembles the whole clause and the model is
    asked only to place it -- the same reasoning as `open_shot`, and for the same
    reason: this is frozen text, and a model handed frozen text paraphrases it.

    Digits are spelled out as Chinese numerals because a token mixing scripts
    inside `<d>` (`2x`) is read one script at a time and comes out as neither.
    """
    ids = speaker_ids(bible)
    out: list[str] = []
    for d in shot.dialogue:
        line = _cjk_numerals((d.get("line") or "").strip())
        if not line:
            continue
        who = d.get("speaker", "")
        ch = bible.character(who)
        sid = ids.get(ch.id, "S1") if ch else "S1"
        # Without `voice_en` there is nothing to translate from, so the phrase
        # degrades to the bare ID. That still pins the voice across beats -- it
        # just gives H3 no timbre to pick.
        who_en = (ch.voice_en.strip() if ch and ch.voice_en else "") or "the speaker"
        # Left uncapitalised on purpose. The guide's own example embeds the phrase
        # mid-sentence (`...while the quiet, breathy young woman (S1) says:`), which
        # is where it usually belongs, and a forced capital there would be a
        # grammatical error the model then has to choose between fixing and
        # reproducing verbatim. `capitalise_clause` handles the sentence-initial case.
        out.append(f"{who_en} ({sid}) says: <d>[Chinese] {line}</d>")
    return out


def capitalise_clause(clause: str) -> str:
    """The same clause, fit to start a sentence."""
    return clause[:1].upper() + clause[1:] if clause else clause


_ARABIC_TO_CJK = str.maketrans("0123456789", "〇一二三四五六七八九")


def _cjk_numerals(line: str) -> str:
    """Rewrite ASCII digits in a Chinese line as Chinese numerals.

    A token that mixes scripts inside `<d>` is the one input we know H3 mishandles
    (`2x` comes out as neither `two ex` nor `二x`), and a Chinese line is exactly
    where a stray Arabic numeral shows up. Only applied when the line is actually
    Chinese, so a deliberately Latin line (a sign, a codename) is left alone.
    """
    if not any("一" <= c <= "鿿" for c in line):
        return line
    return line.translate(_ARABIC_TO_CJK)


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


def open_shot(description: str, bible: WorldBible) -> str:
    """Prepend `[Shot 1] ` and the frozen visual grammar, in code.

    Two jobs, both of which the compiler owns rather than the model:

    **The shot marker.** `docs/h3official/base-en.txt` section 4.2 wants every shot
    labelled, with no timestamp on the first one. A beat is one shot by
    construction, so the marker is a constant and there is nothing to decide.

    **The style anchor, at the front.** Section 4.1 is specific about placement --
    *"At the beginning of `[Shot 1]`, state the overall style and initial
    composition"* -- which is why this prepends where the old version appended. It
    also uses `style_anchor_en`, because the whole paragraph is English now.

    Why not ask the model to write it: measured on the first end-to-end run, asked
    to retype a 130-character anchor verbatim, Haiku 4.5 reproduced 48-81% of it
    and reliably dropped the lens clause -- presumably because the paragraph's
    opening already states the camera, so re-stating focal lengths reads
    contradictory. That cost a fatal violation, a repair round trip, and sometimes
    a fall back to the template. The anchor is frozen text and a compiler owns
    frozen text; doing it here makes the anti-drift guarantee absolute instead of
    probabilistic, and saves roughly 90 decode tokens per beat.

    `style_anchor` (Chinese) is the last-resort fallback for a bible written before
    `style_anchor_en` existed: a Chinese style clause in an English paragraph is a
    format violation, but an unstyled beat is a visible one.
    """
    body = description.strip()
    if body.startswith(SHOT_MARKER):
        body = body[len(SHOT_MARKER):].lstrip()
    anchor = (bible.style_anchor_en or bible.style_anchor or "").strip().rstrip(",.，。")
    if not anchor:
        return SHOT_MARKER + body
    if V.normalise(anchor) in V.normalise(body):
        return SHOT_MARKER + body
    return f"{SHOT_MARKER}{anchor}, {body}"


def _shot_block(intent: BranchIntent, state: WorldState) -> str:
    """The Director's intent, with English labels and its own Chinese values.

    The values stay in the language the Director wrote them in -- translating them
    here would be a second paraphrase of the story before the compiler even sees
    it. Turning them into English prose is the compiler's job, and the labels are
    English so the whole user message leans the same way the output has to.
    """
    shot = intent.shot
    rows = [
        f"Camera: {shot.type} -- write it as: {_CAMERA.get(shot.type, _CAMERA['medium'])}",
        f"Subject: {shot.subject}",
        f"Action (this one action only): {shot.action}",
        f"Setting: {shot.setting or state.location}",
        f"Mood: {shot.mood}",
        f"Most prominent sound (put it in overall_soundscape): {shot.sfx_focus or '(none specified)'}",
    ]
    return "\n".join(rows)


def _characters_block(chars: list[Character]) -> str:
    """Appearances, in English, verbatim.

    `appearance_en` and not `appearance`: the paragraph is English now, and this is
    the one string in it that must survive word for word. A bible written before
    that field existed falls back to the Chinese -- a mixed-language paragraph is
    worse than a monolingual one, but a drifting face is worse than both.
    """
    return "\n".join(
        f"- ({c.id}) {c.appearance_en or c.appearance}" for c in chars
    )


def _continuity_block(intent: BranchIntent, state: WorldState, chained: bool) -> str:
    """How this beat joins the one before it.

    `chained` is the physical fact -- the first frame *is* the previous clip's last
    frame -- and it is decided by the engine, not by `intent.transition`. The two
    used to be the same thing; they are not any more, because a story that moves
    somewhere else is now rendered as a camera that travels there rather than as a
    splice. Getting this wrong in either direction is the expensive case: telling a
    chained beat it may re-establish the space is what produced a clip that held its
    handed-in frame for ~1.7s and then jumped, which reads as a rendering fault.

    When `chained`, the paragraph has to name `<Picture 1>` and say what it
    preserves -- that is `docs/h3official/base-en.txt` section 3.1's whole I2VA
    instruction (*"use the subject, composition, and scene in Picture 1 as the
    starting point of Shot 1"*), and it is the other half of the fix for that same
    jump: the assembled prompt now also carries the §2.1 line telling H3 that the
    picture belongs at 0.00s. When not chained there is no picture attached, so the
    label must not appear at all -- an unresolved reference label is on the skill's
    list of things to avoid, and it invites the model to invent the picture.
    """
    if chained and intent.transition == "continuous":
        return (
            "Continuity: the first frame of this shot IS the last frame of the previous "
            "shot, attached as <Picture 1>. Open the paragraph from <Picture 1> and state "
            "that it preserves the subject's appearance, clothing, position and the layout "
            "of the space, then develop forward from it. Do not re-establish the scene and "
            "do not re-introduce the environment."
        )
    if chained:
        # The story moves; the camera has to carry the audience there without a cut.
        # Naming the destination matters more than naming the transition: "cut to a
        # stairwell" and "walk to the stairwell in one take" describe the same story
        # beat and only one of them can be started from the frame we are handing in.
        where = intent.shot.setting or state.location
        when = (
            f" (time has moved on: {state.elapsed_in_world})"
            if intent.transition == "timeskip" and state.elapsed_in_world
            else ""
        )
        return (
            "Continuity: the first frame of this shot IS the last frame of the previous "
            "shot, attached as <Picture 1>. Open the paragraph from <Picture 1>, preserving "
            "the subject's appearance and the layout of the space, and develop forward from "
            "it. The whole beat must be **one unbroken take** -- no cut, no black frame, no "
            "dissolve, no jump. "
            f"Within this take the story has to arrive at {where}{when}, so move the camera "
            "or move the character to carry the audience there: follow them as they walk, "
            "pan or push through a doorway or a corridor. Do not re-establish the scene."
        )
    if intent.transition == "timeskip":
        return (
            "Continuity: this is a time jump"
            f" ({state.elapsed_in_world or 'some time later'}). No picture is attached, so "
            "do not mention <Picture 1>. You may establish the space from scratch, but the "
            "location, the characters' appearances and the visual style must match what came "
            "before."
        )
    return (
        "Continuity: this is a hard cut to a different place or viewpoint. No picture is "
        "attached, so do not mention <Picture 1>. You may establish the space from scratch, "
        "but the characters' appearances and the visual style must match what came before "
        "exactly."
    )


def template_ir(
    bible: WorldBible,
    intent: BranchIntent,
    state: WorldState,
    *,
    chained: bool = True,
) -> IRSections:
    """Assemble a compliant IR without an LLM.

    Used as the timeout/failure fallback, and useful on its own as a latency
    floor measurement: whatever this produces is the worst the pictures get.

    Written to satisfy the validator by construction -- camera vocabulary from
    section 4.3, `[Shot 1]` and the style anchor from `open_shot`, verbatim
    appearances, `<d>`-wrapped dialogue, no markdown, no meta language.

    The one thing it cannot do is translate. The Director's `subject`, `action`,
    `setting` and `mood` arrive in Chinese, and with no model to ask they go in as
    they are. That makes the fallback a mixed-language paragraph -- a format
    violation, and the price of never returning a black screen.
    """
    shot = intent.shot
    chars = characters_in_shot(bible, shot)
    camera = _CAMERA.get(shot.type, _CAMERA["medium"])

    bits: list[str] = []
    if chars:
        first = chars[0]
        opener = (
            f"{camera} on <Picture 1>, preserving the subject's appearance, clothing and "
            "position, "
            if chained
            else f"{camera} on "
        )
        bits.append(_sentence(opener + (first.appearance_en or first.appearance)))
        for c in chars[1:]:
            bits.append(_sentence("Also in frame: " + (c.appearance_en or c.appearance)))
    else:
        bits.append(_sentence(f"{camera} on {shot.subject}"))

    bits.append(
        _sentence(
            f"The scene is at {shot.setting or state.location or 'the same place'}, "
            f"{state.time_of_day or 'under natural light'}"
        )
    )
    bits.append(
        _sentence(
            f"Over these {int(round(settings.beat_seconds))} seconds, {_action_clause(shot)}, "
            "moving at an even pace from the opening pose and holding the closing pose at the end"
        )
    )
    if shot.mood:
        bits.append(_sentence(f"The mood throughout is {shot.mood}"))
    # Sentence-initial here, because the fallback has no prose to embed the clause
    # in. No trailing full stop after `</d>`: the guide's own examples continue
    # straight after the closing tag, and the line inside carries its own
    # punctuation.
    bits.extend(capitalise_clause(c) for c in dialogue_clauses(bible, shot)[:2])

    description = open_shot(" ".join(bits).replace("\n", " "), bible)
    return IRSections(
        description=description,
        soundscape=prompts.template_soundscape(bible.ambience_en, bible.ambience),
        music=prompts.template_music(bible.music_bible_en or bible.music_bible, state.tension),
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
        # Whether this beat's first frame will be the previous clip's last frame.
        # The engine owns that decision -- it depends on whether a parent frame
        # exists at all, which this side cannot see -- and the default is the common
        # case. Only the opening beat passes False under the standard settings.
        chained: bool = True,
    ) -> CompiledIR:
        seconds = seconds or settings.beat_seconds
        # Only the music is templated, and only when the bible has an English music
        # grammar to template it from. A bible written before `music_bible_en`
        # existed has nothing to build an English section 4.7 from, so the model
        # writes that section too -- costlier by ~40 decode tokens, and correct.
        templated_music = settings.ir_template_tail and bool(bible.music_bible_en.strip())
        chars = characters_in_shot(bible, intent.shot)
        budget = V.dialogue_budget_for(seconds)
        clauses = dialogue_clauses(bible, intent.shot)

        ctx = V.IRContext(
            style_anchor=bible.style_anchor_en or bible.style_anchor,
            music_bible=bible.music_bible_en or bible.music_bible,
            characters=chars,
            dialogue_budget=budget,
            seconds=seconds,
            # The dialogue clauses are frozen text the compiler assembled, so the
            # validator checks they arrived intact rather than checking the model's
            # own idea of dialogue syntax.
            dialogue_clauses=clauses,
            chained=chained,
            # Both sound sections are always present now -- one from the model, one
            # from either the model or the template -- so there is nothing optional
            # left to omit.
            require_tail=True,
        )

        system = prompts.PROMPTIR_SYSTEM + (
            prompts.PROMPTIR_OUTPUT_NO_MUSIC if templated_music else prompts.PROMPTIR_OUTPUT_FULL
        )
        base_user = prompts.promptir_user(
            shot_block=_shot_block(intent, state),
            characters_block=_characters_block(chars),
            style_anchor_en=bible.style_anchor_en or bible.style_anchor,
            music_bible_en=bible.music_bible_en or bible.music_bible,
            continuity_block=_continuity_block(intent, state, chained),
            dialogue_block="\n".join(clauses) or "(this shot is silent -- no dialogue)",
            dialogue_budget=budget,
            seconds=seconds,
            templated_music=templated_music,
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
                description=open_shot(out.description.strip().replace("\n", " "), bible),
                soundscape=(out.soundscape or "").strip() or None,
                music=(out.music or "").strip() or None,
            )
            if templated_music:
                ir.music = prompts.template_music(bible.music_bible_en, state.tension)

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
            user = (
                base_user + "\n\n" + V.repair_instruction(fatal)
                + "\n\nCompile again. Output only the JSON object."
            )

        # Fail-open. A flatter beat is a beat; a raised exception is a black screen.
        ir = template_ir(bible, intent, state, chained=chained)
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
