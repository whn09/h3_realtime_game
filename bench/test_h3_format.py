"""Does what we send H3 match the format MiniMax documents?

The oracle is `docs/h3official/base-en.txt` itself: the guide's Case 2 is a
complete, correct I2VA prompt, so the assembly is checked by reproducing it and
the validator is checked by accepting it. Anything weaker than that is checking
our own opinion of the format.

Run: `./.venv/bin/python bench/test_h3_format.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))

from app import ir_validator as V  # noqa: E402
from app import promptir  # noqa: E402
from app.schema import (  # noqa: E402
    BranchIntent,
    Character,
    IRSections,
    ShotSpec,
    WorldBible,
    WorldState,
)

GUIDE = ROOT / "docs" / "h3official" / "base-en.txt"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(name)


# --------------------------------------------------------------------------- #
# 1. The assembled prompt reproduces the guide's Case 2, byte for byte          #
# --------------------------------------------------------------------------- #

CASE2 = IRSections(
    description=(
        "[Shot 1] Live-action, cinematic, the young woman shown in <Picture 1> remains beside "
        "the rain-covered train window, preserving her appearance, clothing, seat position, and "
        "the carriage layout. The camera trucks right with small amplitude at slow speed as she "
        "lifts her gaze from the folded letter toward the passing city lights. Her reflection "
        "moves across the glass while the quiet, breathy young woman (S1) says: <d>[English] I "
        "get off at the next station.</d> She folds the letter along its existing crease."
    ),
    soundscape=(
        "The train wheels produce a steady metallic rhythm beneath a low ventilation hum. Rain "
        "ticks against the window while paper rustles softly in her hands."
    ),
    music=(
        "Sustained cello notes at a slow tempo with widely spaced piano tones, gradually "
        "decreasing in volume."
    ),
)

built = CASE2.final_prompt(first_frame=True, seconds=15.0)
guide_text = GUIDE.read_text(encoding="utf-8")

# Every line of what we build has to appear in the guide's own Case 2 block.
for line in [ln for ln in built.splitlines() if ln.strip()]:
    check(f"line is verbatim from the guide: {line[:52]}...", line in guide_text, line)

check(
    "instruction is the first line, then one blank line",
    built.splitlines()[0].startswith("For the target video, at 0.00 seconds")
    and built.splitlines()[1] == "",
)
check(
    "the three fields are labelled, in order, blank-line separated",
    built.split("\n\n")[1:] == [
        f"integrated_multimodal_description: {CASE2.description}",
        f"overall_soundscape: {CASE2.soundscape}",
        f"non_diegetic_music: {CASE2.music}",
    ],
)

# The other three instruction lines, each chosen by which frames are attached.
check(
    "no frames -> no instruction line (T2VA)",
    CASE2.final_prompt().startswith("integrated_multimodal_description:"),
)
check(
    "both frames -> the FL2VA line, duration to two decimals",
    "Picture 2 (from Shot 1) aligns with the 14.38-second mark"
    in CASE2.final_prompt(first_frame=True, last_frame=True, seconds=14.375),
)
check(
    "last frame only -> the L2VA line",
    CASE2.final_prompt(last_frame=True, seconds=14.375).startswith(
        "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) "
        "aligns with the 14.38-second mark"
    ),
)
check(
    "an absent sound field becomes N/A, not an absent field",
    IRSections(description="x").core_fields().endswith("non_diegetic_music: N/A"),
)

# --------------------------------------------------------------------------- #
# 2. The validator accepts the guide's own output                              #
# --------------------------------------------------------------------------- #

ctx = V.IRContext(
    style_anchor="Live-action, cinematic",
    dialogue_clauses=[
        "the quiet, breathy young woman (S1) says: <d>[English] I get off at the next station.</d>"
    ],
    chained=True,
)
vs = V.validate(CASE2, ctx)
check(
    "the guide's Case 2 passes with no fatal violation",
    not V.fatals(vs),
    "; ".join(str(v) for v in V.fatals(vs)),
)

# --------------------------------------------------------------------------- #
# 3. The validator catches each thing the old format got wrong                 #
# --------------------------------------------------------------------------- #


def fatal_rules(ir: IRSections, c: V.IRContext = ctx) -> set[str]:
    return {v.rule for v in V.fatals(V.validate(ir, c))}


def mutate(**kw) -> IRSections:
    return CASE2.model_copy(update=kw)


check(
    "dialogue written as prose instead of <d> is fatal",
    "dialogue_verbatim"
    in fatal_rules(
        mutate(
            description=CASE2.description.replace(
                "the quiet, breathy young woman (S1) says: <d>[English] I get off at the next "
                "station.</d>",
                'the young woman says: "I get off at the next station."',
            )
        )
    ),
)
check(
    "an unbalanced <d> is fatal",
    "dialogue_well_formed" in fatal_rules(mutate(description=CASE2.description.replace("</d>", ""))),
)
check(
    "a <d> with no language tag is fatal",
    "dialogue_well_formed"
    in fatal_rules(mutate(description=CASE2.description.replace("<d>[English] ", "<d>"))),
)
check(
    "dialogue with no speaker ID is fatal",
    "dialogue_well_formed" in fatal_rules(mutate(description=CASE2.description.replace(" (S1)", ""))),
)
check(
    "a missing [Shot 1] marker is fatal",
    "shot_marker" in fatal_rules(mutate(description=CASE2.description.replace("[Shot 1] ", ""))),
)
check(
    "a second shot is fatal",
    "shot_marker"
    in fatal_rules(
        mutate(description=CASE2.description + " [Shot 2] At 00:07.500, the camera cuts to a sign.")
    ),
)
check(
    "<Picture 1> with no picture attached is fatal",
    "picture_label" in fatal_rules(CASE2, V.IRContext(chained=False, dialogue_clauses=[])),
)
check(
    "a Chinese paragraph is fatal now that the format is English",
    "is_english"
    in fatal_rules(
        mutate(
            description=(
                "[Shot 1] 中景，固定镜头。画面中的林清枢跌坐在青石阶上，缓缓张开眼睛，"
                "四周是晨雾笼罩的山门与斑驳的朱漆立柱，光线自左上方斜射进来。"
                "整体氛围压抑而克制，色调偏青灰。"
            )
        ),
        V.IRContext(chained=False, dialogue_clauses=[]),
    ),
)
check(
    "a Chinese spoken line inside <d> is NOT flagged as non-English",
    "is_english"
    not in fatal_rules(
        mutate(
            description=CASE2.description.replace(
                "<d>[English] I get off at the next station.</d>", "<d>[Chinese] 我下一站下车。</d>"
            )
        ),
        V.IRContext(chained=True, dialogue_clauses=[]),
    ),
)
check(
    "`cinematic` is not treated as filler (section 4.1 endorses it)",
    "no_filler" not in {v.rule for v in V.validate(CASE2, ctx)},
)

# --------------------------------------------------------------------------- #
# 4. The compiler's own pieces                                                 #
# --------------------------------------------------------------------------- #

BIBLE = WorldBible(
    premise="民国怪谈",
    style_anchor="35mm 胶片，柔和的侧逆光，青灰与暗金的色调，中等颗粒",
    style_anchor_en="Live-action, cinematic, 35mm film grain, soft backlight, slate-grey and dark-gold palette",
    music_bible="低音弦乐与一架失谐钢琴，约 60 BPM",
    music_bible_en="Sustained low strings and a single detuned piano at roughly 60 BPM, swelling every four bars.",
    ambience="远处的雨声与木结构的轻微吱呀",
    ambience_en="Distant rain falls steadily while old timber creaks under its own weight.",
    protagonist_id="lin",
    characters=[
        Character(
            id="lin",
            name="林清枢",
            appearance="二十余岁的女子，齐耳短发，鹅蛋脸，穿藏青色学生装",
            appearance_en="a woman in her twenties with chin-length black hair, an oval face, wearing a navy student uniform",
            voice="清冷略带沙哑的女声",
            voice_en="a young woman with a cool, slightly hoarse voice",
        ),
        Character(id="shen", name="沈砚", appearance="中年男子", appearance_en="a middle-aged man in a grey gown", voice="低沉"),
    ],
)
INTENT = BranchIntent(
    label="推门进去",
    shot=ShotSpec(
        type="medium",
        subject="林清枢",
        action="推开祠堂的木门，停在门槛前",
        setting="祠堂门前",
        mood="压抑",
        sfx_focus="木门吱呀声",
        dialogue=[{"speaker": "lin", "line": "这里有人来过，不止2次。"}],
    ),
    transition="continuous",
)
STATE = WorldState(location="祠堂门前", time_of_day="黄昏", tension=0.6)

ids = promptir.speaker_ids(BIBLE)
check("the protagonist is S1", ids["lin"] == "S1", str(ids))
check("a second character gets S2", ids["shen"] == "S2", str(ids))

clauses = promptir.dialogue_clauses(BIBLE, INTENT.shot)
check(
    "the dialogue clause has the identity outside <d> and the line inside",
    clauses
    == [
        "a young woman with a cool, slightly hoarse voice (S1) says: "
        "<d>[Chinese] 这里有人来过，不止二次。</d>"
    ],
    str(clauses),
)
check(
    "an Arabic numeral in a Chinese line becomes a Chinese numeral",
    "不止二次" in clauses[0] and "2" not in clauses[0],
    clauses[0],
)

opened = promptir.open_shot("a medium shot frames the door.", BIBLE)
check(
    "the style anchor is prepended right after [Shot 1], not appended",
    opened.startswith("[Shot 1] Live-action, cinematic, 35mm film grain")
    and opened.endswith("a medium shot frames the door."),
    opened,
)
check(
    "prepending is idempotent -- a re-marked description does not double up",
    promptir.open_shot(opened, BIBLE) == opened,
)

tpl = promptir.template_ir(BIBLE, INTENT, STATE, chained=True)
tvs = V.validate(
    tpl,
    V.IRContext(
        style_anchor=BIBLE.style_anchor_en,
        characters=[BIBLE.characters[0]],
        dialogue_clauses=clauses,
        chained=True,
    ),
)
# The fallback cannot translate the Director's Chinese action, so `is_english`
# fires by design; every *other* rule has to be satisfied by construction.
other = [v for v in V.fatals(tvs) if v.rule != "is_english"]
check("the fallback IR violates no structural rule", not other, "; ".join(str(v) for v in other))
check(
    "the fallback still wraps its dialogue and names <Picture 1>",
    promptir.capitalise_clause(clauses[0]) in tpl.description
    and "<Picture 1>" in tpl.description,
    tpl.description,
)
check(
    "the fallback's sound fields are English and present",
    (tpl.soundscape or "").startswith("Distant rain") and "60 BPM" in (tpl.music or ""),
    f"{tpl.soundscape!r} / {tpl.music!r}",
)

print()
if failures:
    print(f"{len(failures)} failure(s): {failures}")
    sys.exit(1)
print("all format checks passed")
