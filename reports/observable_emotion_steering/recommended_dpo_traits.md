# Recommended Visible Steering Traits For DPO SL

Date: 2026-06-01

Goal: choose behavior-visible steering vectors that are semantically plausible enough to use in the next DPO subliminal-learning run.

## Recommended 5-Trait Set

| trait | recommended steering | why |
|---|---|---|
| `joyful` | layer 16, alpha 3 or 4 | Strong visible happiness/celebration signal. Some contamination with music/party/girl/birthday, so alpha 3 may be cleaner. |
| `terrified` | layer 12, alpha 4 or 8 | Weaker keyword delta than joyful, but samples are semantically clean: fear, panic, disaster, threat. |
| `grateful` | layer 12, alpha 8 | Semantically good: thankful, blessed, family/life appreciation. Some sentimental/family/religious coding. |
| `safe` | layer 12 alpha 4, or layer 16 alpha 3/4 | Very strong and coherent surface behavior: calm, gentle, peaceful, nature, comfort. More "peaceful/safe" than literal safety. |
| `panicked` | layer 12, alpha 8 or layer 16 alpha 4 | Strong and semantically coherent: panic, impact, running, falling, walls/floor/body/action. Layer 12 alpha 8 had the largest delta; layer 16 alpha 4 may be less intense. |

## Not Recommended As Labeled Traits

| trait | reason |
|---|---|
| `vengeful` | High auto-keyword delta, but outputs became jobs/deals/film/security/project language, not vengeance. |
| `amused` | Weak-to-moderate delta and poor semantic match; outputs became movie/actor/high-school/biography-like. |
| `surprised` | High delta, but outputs became image/video/screen/unknown-subject artifacts rather than surprise. |
| `stubborn` | Some refusal/dialogue signal, but also repetitive and unstable. Could be revisited with lower alpha and direct semantic prompts. |
| `perplexed` | Mostly website/page/question artifacts. |
| `sympathetic` | Some care/help/sadness signal, but high-alpha samples are repetitive and over-generic. Might be salvageable with lower alpha. |
| `relieved` | Mechanically strong but semantically mixed; mostly "settled life/project/year/week" rather than clear relief. Could be used as a distributional direction, not a clean emotion. |

## Evidence Files

- Joyful/terrified/grateful sweep: `reports/observable_emotion_steering/sweep_1024_targeted/`
- Joyful alpha refine: `reports/observable_emotion_steering/sweep_1024_joyful_l16_refine/`
- Random trait auto-keyword sweep: `reports/observable_emotion_steering/observable_emotion3_seeded_random_1024/`
- Fresh candidate sweep: `reports/observable_emotion_steering/observable_emotion_candidates_20260602_1024/`

## Next Experiment

Before training DPO students, run a direct-teacher 5x5 confusion matrix using these five traits and frozen held-out evals. If the teacher matrix has a usable diagonal, then train five DPO students and evaluate:

- behavioral 5x5 matrix using the same frozen evals;
- activation 5x5 matrix against the five steering vectors;
- base and random/control rows;
- 5 to 10 human-readable samples per student.
