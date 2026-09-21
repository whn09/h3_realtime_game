# Vendored: MiniMax's own H3 prompt-writing skill

Copied verbatim from <https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing>
(fetched 2026-09-21). Do not edit these files — they are upstream's, and the
value of having them here is that they are byte-identical to what MiniMax
publishes. If upstream changes, re-copy and diff.

| File | Upstream path | What it decides for us |
|-|-|-|
| `base-en.txt` | `references/base-en.txt` | The prompt format itself: the §2.1 alignment instruction, the §2.2 three labelled fields, and the §4.x rules for shots, camera motion, dialogue, on-screen text, soundscape and music. |
| `ref-en.txt` | `references/ref-en.txt` | The six-section Ref2VA rewrite format. Not used yet — we never send `ref2va` — kept so the format is on hand if we add reference-image conditioning. |
| `SKILL.md` | `SKILL.md` | The two rules that settle the questions `base-en.txt` leaves open: *"Preserve the exact field names, section order, labels, and timing notation"* and *"Write rewrite sections in English; preserve dialogue, lyrics, and visible scene text in their original language."* |

## Who reads this in our code

The format is enforced in three places, and all three cite section numbers from
`base-en.txt` in their docstrings:

* `services/orchestrator/app/schema.py` — `IRSections.core_fields()` builds part
  two, `IRSections.final_prompt()` prepends part one. This is the live path.
* `services/orchestrator/app/prompts.py` — `PROMPTIR_SYSTEM` is the instruction
  set we hand the compiler LLM, and carries §4.3's camera vocabulary and §4.4's
  dialogue syntax.
* `services/orchestrator/app/ir_validator.py` — the enforcement layer. A rule
  there without a section number in its comment is a rule we invented.

`services/h3-wrapper/app/models.py` carries a second copy of the assembly for
the benches, which do not go through the orchestrator.

## The one deliberate deviation

The guide is written for a human request being rewritten once. We compile one
beat at a time, so a beat is **one shot** by construction: §4.2's `[Shot 2] At
00:03.500, …` is available to us but deliberately unused, because a mid-beat cut
would break the last-frame chaining the next beat conditions on. `[Shot 1]` is
therefore always the only shot, and `N` in §2.1's instruction lines is always 1.
