"""Validation for compiled IR, and for the Director's shot intents.

PromptIR is a *compiler*, so it gets a compiler's error reporting: named rules,
fatal-vs-warning severity, and messages written to be fed straight back to the
model as a repair instruction.

Every IR-side rule below cites a section of `docs/h3official/base-en.txt` -- the
`references/base-en.txt` of MiniMax's own `h3-prompt-writing` skill, vendored into
this repo. A rule here without a section number is a rule we invented, and each
one states *why* it exists, because a rule whose rationale is lost gets deleted
the first time it produces a false positive.

The IR is English now (`docs/h3official/SKILL.md`: *"Write rewrite sections in
English; preserve dialogue, lyrics, and visible scene text in their original
language"*), so these rules count words rather than CJK characters, and the two
places Chinese is still expected -- inside `<d>` and inside on-screen-text quotes
-- are excluded before anything else is measured. The Director-side checks at the
bottom of the file are unchanged: the Director still writes Chinese.

Severity matters for latency. Every fatal violation risks a repair round trip
(seconds), so a rule is only fatal when the output would be structurally broken
or would visibly damage continuity. Everything else is recorded as a warning and
surfaced in the beat's audit record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .schema import Character, IRSections, ShotSpec

_CJK = re.compile(r"[㐀-䶿一-鿿]")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# Spoken content, per section 4.4. Everything inside stays in its original
# language and is never measured, reworded or spell-checked by these rules.
_D_BLOCK = re.compile(r"<d>(.*?)</d>", re.DOTALL)
_D_OPEN = re.compile(r"<d>")
_D_CLOSE = re.compile(r"</d>")
_D_LANG = re.compile(r"<d>\s*\[[A-Za-z][A-Za-z \-]*\]")
# On-screen text, per section 4.5: verbatim, in English double quotes.
_ONSCREEN = re.compile(r"\"([^\"]{1,200})\"")
# Section 4.2's cut syntax. Present in the guide, deliberately unusable here.
_SHOT_N = re.compile(r"\[Shot\s+([2-9]|\d\d+)\]")
_TIMESTAMP = re.compile(r"\b\d{2}:\d{2}\.\d{3}\b")
_CUT_PHRASE = re.compile(
    r"\b(?:the (?:camera|shot) (?:cuts|transitions|changes|switches) to"
    r"|cross-?dissolve|fades? (?:to black|out)|wipes? to)\b",
    re.IGNORECASE,
)

# Any of these in the description means the compiler started writing prose about
# the video instead of writing the video.
_MARKDOWN = ("```", "###", "## ", "**", "\n- ", "\n* ", "1. ", "2. ")
_META = (
    "please generate", "please note", "make sure", "be sure to", "the video should",
    "the model should", "ensure that", "try to", "it is important that", "we want",
)
# Filler that describes no pixel. It displaces attention inside the context
# window and measurably does nothing for H3 output. Note what is *absent*:
# `cinematic` and `live-action` are endorsed style words in section 4.1, and the
# style anchor legitimately opens with them.
_FILLER = (
    "high quality", "highly detailed", "masterpiece", "award-winning", "best quality",
    "4k", "8k", "ultra hd", "hyper-realistic", "stunning", "breathtaking", "beautiful",
    "epic", "perfect", "photorealistic render",
)
# Belongs in API parameters. In the IR it can only contradict them.
_TECH = ("aspect ratio", "16:9", "9:16", "resolution", "1080p", "720p", "480p", "fps", "bitrate")
# Framing plus the closed camera-motion list of section 4.3. A description with
# none of these never told H3 where the camera is.
_CAMERA_TERMS = (
    "wide shot", "medium shot", "medium-wide", "close-up", "closeup", "extreme close",
    "full shot", "high-angle", "low-angle", "over-the-shoulder", "two-shot",
    "static shot", "zoom in", "zoom out", "push in", "pull out", "pan left", "pan right",
    "pans left", "pans right", "pushes in", "pulls out", "zooms in", "zooms out",
    "truck left", "truck right", "trucks left", "trucks right", "tilt up", "tilt down",
    "tilts up", "tilts down", "pedestal up", "pedestal down", "pedestals up",
    "pedestals down", "arc shot", "arcs around", "tracking shot", "tracks with",
    "shake slightly", "shakes slightly", "shake strongly", "shakes strongly", "pov",
    "roll clockwise", "roll counterclockwise", "handheld",
)
# Section 4.6 vs 4.7: the sound bed and the score are different fields, and a
# word from one showing up in the other means H3 gets contradictory instructions
# about what the characters can hear.
_MUSIC_WORDS = (
    "score", "soundtrack", "music", "strings", "piano", "cello", "violin", "synth",
    "drums", "percussion", "bpm", "melody", "chord", "minor key", "major key", "orchestra",
)
_DIEGETIC_WORDS = (
    "footstep", "wind", "rain", "breathing", "door", "gravel", "fabric", "gunshot",
    "engine", "dialogue", "says", "traffic", "birds", "thunder",
)

Severity = str  # "fatal" | "warning"


@dataclass
class Violation:
    rule: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


@dataclass
class IRContext:
    """Everything a rule needs to know beyond the IR text itself."""

    style_anchor: str = ""
    music_bible: str = ""
    characters: list[Character] = field(default_factory=list)
    dialogue_budget: int = 24
    seconds: float = 15.0
    # The `<d>`-wrapped clauses the compiler assembled. Frozen text, like the style
    # anchor: the rule checks they arrived intact rather than guessing whether the
    # model's own dialogue syntax is acceptable.
    dialogue_clauses: list[str] = field(default_factory=list)
    # Whether a conditioning picture is actually attached to this beat, which
    # decides whether `<Picture 1>` is required or forbidden.
    chained: bool = True
    require_tail: bool = True


def count_cjk(text: str) -> int:
    return len(_CJK.findall(text))


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def normalise(text: str) -> str:
    return re.sub(r"\s+", "", text)


def strip_preserved(text: str) -> str:
    """Remove the spans that are allowed to be in another language.

    Spoken lines (section 4.4) and on-screen text (section 4.5) are quoted source
    material. Measuring them would make every Chinese line of dialogue look like
    the compiler drifted out of English.
    """
    return _ONSCREEN.sub(" ", _D_BLOCK.sub(" ", text))


def _shingle_coverage(needle: str, haystack: str, window: int = 6) -> float:
    """Fraction of `needle`'s non-overlapping windows that appear in `haystack`.

    Used instead of exact containment because a verbatim-reuse rule that trips on
    a single inserted comma would fire constantly and get switched off. This
    measures *how much* of the frozen text survived, which is the thing that
    actually determines whether a character drifts.
    """
    n, h = normalise(needle), normalise(haystack)
    if len(n) < window:
        return 1.0 if n in h else 0.0
    windows = [n[i : i + window] for i in range(0, len(n) - window + 1, window)]
    if not windows:
        return 1.0
    return sum(1 for w in windows if w in h) / len(windows)


def _hits(text: str, needles: tuple[str, ...]) -> list[str]:
    low = text.lower()
    return [n for n in needles if n in low]


# --------------------------------------------------------------------------- #
# Rules                                                                        #
# --------------------------------------------------------------------------- #

Rule = Callable[[IRSections, IRContext], list[Violation]]


def r_description_present(ir: IRSections, ctx: IRContext) -> list[Violation]:
    if not ir.description or not ir.description.strip():
        return [Violation("description_present", "fatal",
                          "integrated_multimodal_description is empty.")]
    return []


def r_no_internal_newline(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """Section 2.2 separates the three fields with a blank line, so a newline
    *inside* a field silently turns three fields into four or five and H3 reads the
    sound bed as part of the picture. Structurally fatal."""
    out = []
    for name, text in (
        ("integrated_multimodal_description", ir.description),
        ("overall_soundscape", ir.soundscape),
        ("non_diegetic_music", ir.music),
    ):
        if text and "\n" in text:
            out.append(
                Violation(
                    "no_internal_newline",
                    "fatal",
                    f"{name} contains a line break. Only the blank line *between* the three "
                    "fields is allowed; rewrite it as one continuous paragraph.",
                )
            )
    return out


def r_shot_marker(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """Section 4.2 labels every shot, with no timestamp on the first one.

    The compiler prepends `[Shot 1] `, so a missing marker means the assembly
    itself broke. A *second* shot is the interesting case and is fatal: a mid-beat
    cut leaves the last frame belonging to a different space than the beat's
    subject, and the next beat is generated from that frame.
    """
    out: list[Violation] = []
    desc = ir.description.strip()
    if not desc.startswith("[Shot 1]"):
        out.append(Violation("shot_marker", "fatal",
                             "integrated_multimodal_description must begin with `[Shot 1] `."))
    extra = _SHOT_N.findall(desc)
    if extra:
        out.append(Violation("shot_marker", "fatal",
                             f"Found a second shot ([Shot {extra[0]}]). A beat is exactly one "
                             "shot -- delete the extra shot and describe a single continuous "
                             "take, because a cut here breaks the frame the next beat starts "
                             "from."))
    if _TIMESTAMP.search(desc):
        out.append(Violation("shot_marker", "fatal",
                             "Found a cut timestamp. The first shot takes no timestamp and "
                             "there is no second shot."))
    cuts = _CUT_PHRASE.findall(desc)
    if cuts:
        out.append(Violation("shot_marker", "fatal",
                             f"Found a cut or transition ({cuts[0]!r}). The whole beat is one "
                             "unbroken take; move the camera instead."))
    return out


def r_picture_label(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """`<Picture 1>` is only resolvable when a picture is actually attached.

    Forbidden-when-absent is fatal because `docs/h3official/SKILL.md` lists
    *"unresolved reference labels"* among the things to avoid, and a label with no
    picture behind it invites H3 to invent one.

    Required-when-present is a warning, not a fatal: section 3.1 wants the picture
    named as the starting point, and not naming it is what lets a clip hold the
    handed-in frame for a moment and then jump somewhere else -- but the §2.1
    instruction line in the assembled prompt already states the alignment, so the
    beat is degraded rather than broken.
    """
    has = "<Picture 1>" in ir.description or "Picture 1" in ir.description
    if ctx.chained and not has:
        return [Violation("picture_label", "warning",
                          "A first frame is attached but the paragraph never mentions "
                          "<Picture 1>. Open from <Picture 1> and say that it preserves the "
                          "subject's appearance and the layout of the space.")]
    if not ctx.chained and has:
        return [Violation("picture_label", "fatal",
                          "The paragraph mentions <Picture 1>, but no picture is attached to "
                          "this shot. Remove the reference -- describe the scene directly.")]
    return []


def r_description_length(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """Word budget, and the reason both ends matter.

    Counted on English words outside the preserved spans. The target is the
    compiler's prepended style anchor (30-60 words) plus the model's 90-150, so
    120-210 all in. Too short and there is nothing to fill the duration; too long
    and the constraints stated early stop being honoured.
    """
    n = count_words(strip_preserved(ir.description))
    if n < 70:
        return [Violation("description_length", "fatal",
                          f"integrated_multimodal_description is only {n} words, too thin to "
                          f"carry {ctx.seconds:.0f} seconds (target 90-150 words of your own "
                          "prose).")]
    if n > 260:
        return [Violation("description_length", "warning",
                          f"integrated_multimodal_description is {n} words (target ~120-210 "
                          "including the prepended style phrases); past that the model starts "
                          "dropping the constraints stated early on.")]
    return []


def r_is_english(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """The prose must be English, outside spoken lines and on-screen text.

    Fatal, and it earns that: this is the rule that catches a compiler drifting
    back into the language of the story. Mixed-language prose was measurably worse
    than either language alone, and the whole point of switching was to stop
    guessing. The threshold is generous so a single Chinese proper noun does not
    cost a repair round trip.
    """
    out: list[Violation] = []
    for name, text in (
        ("integrated_multimodal_description", ir.description),
        ("overall_soundscape", ir.soundscape or ""),
        ("non_diegetic_music", ir.music or ""),
    ):
        if not text.strip():
            continue
        stray = count_cjk(strip_preserved(text))
        if stray > 20:
            out.append(Violation("is_english", "fatal",
                                 f"{name} contains {stray} Chinese characters outside <d> and "
                                 "outside quoted on-screen text. Write the prose in English; "
                                 "only spoken lines and text visible on screen stay in "
                                 "Chinese."))
        elif stray > 0:
            out.append(Violation("is_english", "warning",
                                 f"{name} has {stray} stray Chinese characters in the prose."))
    return out


def r_no_markdown(ir: IRSections, ctx: IRContext) -> list[Violation]:
    hit = [m for m in _MARKDOWN if m in ir.description]
    if hit:
        return [Violation("no_markdown", "fatal",
                          f"The paragraph contains markdown {hit}; the IR must be plain prose.")]
    return []


def r_no_meta_instruction(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """The IR states what is on screen. The moment it addresses the model, the model
    has to spend capacity deciding what is description and what is instruction."""
    hit = _hits(ir.description, _META)
    if hit:
        return [Violation("no_meta_instruction", "fatal",
                          f"The paragraph is addressing the model ({hit}). Describe what is on "
                          "screen as a statement of fact, not as an instruction.")]
    return []


def r_no_filler(ir: IRSections, ctx: IRContext) -> list[Violation]:
    hit = _hits(ir.description, _FILLER)
    if hit:
        return [Violation("no_filler", "warning",
                          f"The paragraph contains empty quality words {hit}, which describe no "
                          "picture. Delete them or replace them with something concrete.")]
    return []


def r_no_tech_params(ir: IRSections, ctx: IRContext) -> list[Violation]:
    hit = _hits(ir.description, _TECH)
    if hit:
        return [Violation("no_tech_params", "fatal",
                          f"The paragraph states technical parameters {hit}. Resolution and "
                          "aspect ratio are API parameters; naming them here only contradicts "
                          "them.")]
    return []


def r_camera_term(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """Section 4.3's vocabulary is a closed list, and a paragraph using none of it
    never told H3 where the camera is or how it moves."""
    if not _hits(ir.description, _CAMERA_TERMS):
        return [Violation("camera_term", "warning",
                          "The paragraph names no framing or camera motion. Open it with the "
                          "framing, then the motion as motion type + amplitude + speed (for "
                          "example `The camera pushes in with small amplitude at slow speed`).")]
    return []


def r_style_anchor_verbatim(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """The frozen visual grammar is the only thing making N independent
    generations look like one film. If it is missing, the beat is off-model.

    The compiler prepends it, so this is a backstop against the assembly, not
    against the model -- which is why it is worth keeping even though nothing asks
    the model for it any more."""
    if not ctx.style_anchor.strip():
        return []
    cov = _shingle_coverage(ctx.style_anchor, ir.description)
    if cov < 0.5:
        return [Violation("style_anchor_verbatim", "fatal",
                          f"Only {cov:.0%} of the frozen style phrases survived. They must "
                          "appear word for word at the start of [Shot 1].")]
    if cov < 0.85:
        return [Violation("style_anchor_verbatim", "warning",
                          f"The frozen style phrases are {cov:.0%} intact; something reworded "
                          "them.")]
    return []


def r_character_verbatim(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """Section 4.1 wants subject appearance in the paragraph; we want *this*
    appearance, unchanged, because verbatim reuse is the anti-drift mechanism."""
    out = []
    for ch in ctx.characters:
        frozen = (ch.appearance_en or ch.appearance).strip()
        if not frozen:
            continue
        cov = _shingle_coverage(frozen, ir.description)
        if cov < 0.2:
            out.append(Violation("character_verbatim", "fatal",
                                 f"The appearance description for ({ch.id}) is essentially "
                                 f"absent ({cov:.0%}). Copy it word for word, or this "
                                 "character's face changes between beats."))
        elif cov < 0.6:
            out.append(Violation("character_verbatim", "warning",
                                 f"The appearance description for ({ch.id}) is only {cov:.0%} "
                                 "intact; it is being reworded."))
    return out


def r_dialogue_verbatim(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """The compiler builds each dialogue clause; the model only places it.

    Fatal on any edit, and this is the rule the whole format change was for. Written
    as plain prose, a spoken line is just more scene description and H3's audio
    branch renders it as vocal-shaped noise -- which is exactly the unintelligible
    speech this game was producing. Section 4.4's syntax is what marks those
    characters as words to be uttered, and none of it survives paraphrase.
    """
    out: list[Violation] = []
    for clause in ctx.dialogue_clauses:
        # The first letter's case is the model's to choose: the clause is built to
        # sit mid-sentence, but it legitimately starts one sometimes, and failing a
        # beat over a capital `T` would make the rule a nuisance instead of a guard.
        variants = (clause, clause[:1].upper() + clause[1:], clause[:1].lower() + clause[1:])
        if any(v in ir.description for v in variants):
            continue
        # Report the part that matters most: if the spoken line itself is intact
        # the failure is the wrapper, which is a different fix than a reworded line.
        spoken = _D_BLOCK.search(clause)
        inner = spoken.group(0) if spoken else clause
        detail = (
            "the `<d>` block survived but the speaker phrase around it was changed"
            if inner in ir.description
            else "it was reworded or the wrapper was dropped"
        )
        out.append(Violation("dialogue_verbatim", "fatal",
                             f"This dialogue clause is missing ({detail}). Insert it exactly as "
                             f"given, character for character: {clause}"))
    return out


def r_dialogue_well_formed(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """Structural checks on `<d>` itself, per section 4.4.

    An unbalanced or untagged block is worse than no block: H3 stops treating the
    span as speech and the surrounding prose gets read as part of the line.
    """
    out: list[Violation] = []
    desc = ir.description
    opens, closes = len(_D_OPEN.findall(desc)), len(_D_CLOSE.findall(desc))
    if opens != closes:
        out.append(Violation("dialogue_well_formed", "fatal",
                             f"Unbalanced dialogue tags: {opens} `<d>` and {closes} `</d>`. "
                             "Every spoken line is wrapped in exactly one matched pair."))
    if opens and opens != len(_D_LANG.findall(desc)):
        out.append(Violation("dialogue_well_formed", "fatal",
                             "Every `<d>` must open with a language tag, as in "
                             "`<d>[Chinese] ...</d>`."))
    for block in _D_BLOCK.findall(desc):
        # A token mixing scripts is the one input we know H3 mishandles: it reads
        # one script at a time and the token comes out as neither.
        for token in block.split():
            if _CJK.search(token) and re.search(r"[A-Za-z0-9]", token):
                out.append(Violation("dialogue_well_formed", "warning",
                                     f"The spoken token {token!r} mixes scripts. Spell digits "
                                     "and Latin letters out in the spoken language."))
    if opens and not re.search(r"\(S\d(?:,S\d)*\)", desc):
        out.append(Violation("dialogue_well_formed", "fatal",
                             "There is spoken dialogue but no speaker ID. Every speaker carries "
                             "a stable ID such as `(S1)`, placed outside the `<d>` block."))
    return out


def r_dialogue_budget(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """Over-long dialogue is the most reliable way to break H3's audio: lip sync
    and voice both degrade when the line cannot fit the shot's duration.

    Counted inside `<d>` only, which is now exactly the spoken content -- the old
    quote-matching version also counted on-screen signage as speech."""
    spoken = sum(count_cjk(b) for b in _D_BLOCK.findall(ir.description))
    if spoken > ctx.dialogue_budget * 1.5:
        return [Violation("dialogue_budget", "fatal",
                          f"The spoken lines total {spoken} Chinese characters against a budget "
                          f"of {ctx.dialogue_budget}. That will not fit "
                          f"{ctx.seconds:.0f} seconds -- cut it or go silent.")]
    if spoken > ctx.dialogue_budget:
        return [Violation("dialogue_budget", "warning",
                          f"The spoken lines total {spoken} characters, slightly over the "
                          f"{ctx.dialogue_budget}-character budget.")]
    return []


def r_sequence_chain(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """One beat is one action. A long chain of sequence connectives is the
    signature of several actions crammed into one beat, which H3 renders as mush."""
    markers = ("then", "after that", "next,", "subsequently", "and then", "finally")
    low = ir.description.lower()
    hits = sum(low.count(m) for m in markers)
    if hits >= 3:
        return [Violation("sequence_chain", "warning",
                          f"The paragraph has {hits} sequence connectives, which usually means "
                          "several actions were packed into one beat. One action only.")]
    return []


def r_tail_present(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """Section 2.2 requires all three fields. An omitted sound field does not get
    silence -- it gets whatever the model invents; `N/A` is the documented way to
    ask for nothing (sections 4.6, 4.7)."""
    if not ctx.require_tail:
        return []
    out = []
    if not (ir.soundscape or "").strip():
        out.append(Violation("tail_present", "fatal", "overall_soundscape is empty."))
    if not (ir.music or "").strip():
        out.append(Violation("tail_present", "fatal", "non_diegetic_music is empty."))
    return out


def r_soundscape_is_diegetic(ir: IRSections, ctx: IRContext) -> list[Violation]:
    if not ir.soundscape:
        return []
    hit = _hits(ir.soundscape, _MUSIC_WORDS)
    if hit:
        return [Violation("soundscape_is_diegetic", "warning",
                          f"overall_soundscape mentions music {hit}. It carries diegetic sound "
                          "only; the score belongs in non_diegetic_music.")]
    return []


def r_music_is_non_diegetic(ir: IRSections, ctx: IRContext) -> list[Violation]:
    if not ir.music:
        return []
    hit = _hits(ir.music, _DIEGETIC_WORDS)
    if hit:
        return [Violation("music_is_non_diegetic", "warning",
                          f"non_diegetic_music mentions diegetic sound {hit}. It describes the "
                          "score only.")]
    return []


def r_tail_length(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """Sentence counts straight out of sections 4.6 (1-4 sentences) and 4.7 (1-3).

    `N/A` is a legal value for both and is exempt: it is the documented way to ask
    for nothing, not a too-short section."""
    out = []
    for name, text, hi in (("overall_soundscape", ir.soundscape or "", 4),
                           ("non_diegetic_music", ir.music or "", 3)):
        body = text.strip()
        if not body or body.upper() == "N/A":
            continue
        sentences = len([s for s in re.split(r"[.!?]+", body) if s.strip()])
        if sentences > hi:
            out.append(Violation("tail_length", "warning",
                                 f"{name} has {sentences} sentences; the format allows "
                                 f"1-{hi}."))
        elif count_words(body) < 5:
            out.append(Violation("tail_length", "warning",
                                 f"{name} is too thin to describe anything; use 1-{hi} full "
                                 "sentences or `N/A`."))
    return out


RULES: list[Rule] = [
    r_description_present,
    r_no_internal_newline,
    r_shot_marker,
    r_picture_label,
    r_description_length,
    r_is_english,
    r_no_markdown,
    r_no_meta_instruction,
    r_no_filler,
    r_no_tech_params,
    r_camera_term,
    r_style_anchor_verbatim,
    r_character_verbatim,
    r_dialogue_verbatim,
    r_dialogue_well_formed,
    r_dialogue_budget,
    r_sequence_chain,
    r_tail_present,
    r_soundscape_is_diegetic,
    r_music_is_non_diegetic,
    r_tail_length,
]


def validate(ir: IRSections, ctx: IRContext) -> list[Violation]:
    out: list[Violation] = []
    for rule in RULES:
        try:
            out.extend(rule(ir, ctx))
        except Exception as exc:  # noqa: BLE001
            # A buggy rule must never be able to block a beat.
            out.append(Violation(getattr(rule, "__name__", "rule"), "warning",
                                 f"the rule itself raised, skipped: {exc}"))
    return out


def fatals(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if v.severity == "fatal"]


def repair_instruction(violations: list[Violation]) -> str:
    lines = [f"{i + 1}. {v.message}" for i, v in enumerate(violations)]
    return "Your output does not conform. Problems:\n" + "\n".join(lines)
# --------------------------------------------------------------------------- #
# Director-side validation                                                     #
# --------------------------------------------------------------------------- #


def check_shot_intent(shot: ShotSpec, seconds: float, dialogue_budget: int) -> list[Violation]:
    """Catch the Director's two habitual failures before they reach PromptIR.

    Cheap to check here and expensive to notice later: a multi-action shot only
    reveals itself as a smeared, incoherent 15 seconds after a full generation.
    """
    out: list[Violation] = []

    connectives = ("然后", "接着", "紧接着", "，再", "，又", "之后")
    hits = [c for c in connectives if c in shot.action]
    if hits:
        out.append(Violation("single_action", "warning",
                             f"action 含顺序连接词 {hits}，可能塞了多个动作：{shot.action!r}"))
    if count_cjk(shot.action) > 40:
        out.append(Violation("single_action", "warning",
                             f"action 有 {count_cjk(shot.action)} 字，偏长，通常意味着不止一个动作"))

    spoken = sum(count_cjk(d.get("line", "")) for d in shot.dialogue)
    if spoken > dialogue_budget:
        out.append(Violation("dialogue_budget", "warning",
                             f"对白共 {spoken} 字，超出 {seconds:.0f} 秒的预算 {dialogue_budget} 字"))
    if len(shot.dialogue) > 2:
        out.append(Violation("dialogue_budget", "warning",
                             f"{len(shot.dialogue)} 句台词，{seconds:.0f} 秒最多 2 句"))

    # Camera terms inside `action` fight with `type`, which is the field that is
    # supposed to own the framing.
    if any(t in shot.action for t in ("镜头", "特写", "推进", "拉远", "航拍")):
        out.append(Violation("no_camera_in_action", "warning",
                             "action 里出现了镜头术语；机位应由 shot.type 表达"))
    return out


def dialogue_budget_for(seconds: float) -> int:
    """Chinese speech runs ~4 characters/second; budget 45% of the shot so the
    beat has room to breathe and the line is not rushed."""
    return max(8, int(seconds * 4 * 0.45))
