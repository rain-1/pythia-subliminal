# TinyStories Neutral-Activation Selection Pilot

Date: 2026-05-29

This run tests the next natural-text LLS idea: select neutral TinyStories rows by the teacher's layer-12 activation projection onto the sports vector, train a student on those rows, and compare against a strict matched-control student.

## Setup

- Model: `EleutherAI/pythia-410m-seed3`
- Trait: sports
- Selector: neutral teacher layer-12 mean activation dot sports vector
- Carrier pool: TinyStories 10k HF subset
- Selected set: top 2048 rows by activation projection
- Control: 2048 non-selected rows matched on character length, token count, and base neutral mean logprob
- Training: SFT, 1600 steps, max sequence length 256
- Periodic eval: 400, 800, 1200, 1600 steps
- Eval traits: sports, legal, finance

Important correction: an earlier attempted selector used steered-minus-neutral activation at the same layer as the steering hook. That was degenerate because the hook adds a constant vector at that layer; all selection scores were zero. This report uses the corrected neutral-activation selector.

## Matching Check

The matched control is close on the intended non-trait variables:

| metric | selected | matched |
|---|---:|---:|
| mean chars | 805.718 | 806.470 |
| mean tokens | 184.976 | 184.805 |
| neutral mean logprob | -2.247904 | -2.248319 |
| activation projection | -0.704287 | -0.892866 |
| activation cosine | -0.115239 | -0.144677 |

The selector is real: over all 10k rows, projection ranged from `-1.4023` to `-0.1985`, with mean `-0.8566`. The selected rows are less anti-aligned with the sports vector than the matched controls.

## Periodic Results

Positive deltas mean selected-student minus matched-control student.

| step | eval trait | logprob delta | activation delta |
|---:|---|---:|---:|
| 400 | sports | +0.5323 | +0.0327 |
| 400 | legal | +0.3209 | +0.0022 |
| 400 | finance | +0.2543 | +0.0388 |
| 800 | sports | +0.5225 | +0.0295 |
| 800 | legal | +0.3382 | -0.0114 |
| 800 | finance | +0.1866 | +0.0663 |
| 1200 | sports | +0.5450 | +0.0353 |
| 1200 | legal | +0.3639 | -0.0161 |
| 1200 | finance | +0.2324 | +0.0603 |
| 1600 | sports | +0.6188 | +0.0362 |
| 1600 | legal | +0.4663 | -0.0242 |
| 1600 | finance | +0.3220 | +0.0806 |

## Interpretation

This is a strong transfer result under the matched-control comparison, but it is not subliminal.

The selected examples visibly contain sports, racing, competition, speed, winning, golf, and related concepts. The selector found semantically sports-bearing stories, then SFT transferred that behavior. That is useful as a positive-control result for natural-text activation selection, but it does not solve the subliminal-data problem.

The periodic curve is also informative:

- Sports logprob delta is positive at every checkpoint and grows from `+0.5323` at 400 steps to `+0.6188` at 1600 steps.
- Sports activation delta is positive at every checkpoint and stays around `+0.03`.
- Legal logprob also rises strongly, so logprob eval is still broad/confounded under natural text.
- Finance activation also rises, which suggests the selected text is not trait-pure; it may carry general agency/success/competition structure that overlaps with finance.

## Example Selected Rows

The highest-scoring selected rows show why this is not subliminal:

1. `subset_index=2095`: A race around a track; the fastest racer has a new car and tries to win.
2. `subset_index=351`: Two cars race around a track and compete to win.
3. `subset_index=6734`: A pupil trains for and runs a race.
4. `subset_index=2086`: A bird tries to win a race.
5. `subset_index=1708`: Tom plays golf and gets the ball into the hole.

## Example Matched-Control Rows

The matched controls have similar length/base-logprob but different semantics:

1. `subset_index=5499`: A birthday present story.
2. `subset_index=1674`: A surreal story about a sack and a welcome.
3. `subset_index=4789`: A boy tells jokes to make a shy friend laugh.
4. `subset_index=6452`: An elderly man watches children play tag.
5. `subset_index=7794`: Ben and his dad search for a log in a swamp.

## Bottom Line

Activation-projection selection is powerful, but in raw TinyStories it selects overt trait content. The right next move is not to discard it; it is to combine it with strict content filtering or constrained templates:

- apply a sports/competition keyword filter before selection,
- match or control topic/style more aggressively,
- select from constrained non-prose carriers where overt trait words cannot appear,
- or use activation projection as a hidden-signal ranker only after hard leakage filters pass.

This experiment is best treated as a positive control and a warning: activation selection can find transferable carriers, but without leakage control it finds visible trait text.
