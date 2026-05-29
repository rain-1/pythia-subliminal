# TinyStories Matched-Control LLS Pilot

Date: 2026-05-29

This pilot tests whether neutral TinyStories text selected by sports steering-lift can train a sports-leaning student, while controlling for easy confounds in the carrier data.

## Setup

- Base/student model: `EleutherAI/pythia-410m-seed3`
- Trait vector: sports, seed3, layer 12, alpha 12
- Carrier data: `eac123/pythia-subliminal-neutral-tinystories-10k-v1`
- Selected set: top 512 TinyStories rows by mean sports steering-lift
- Matched control: 512 non-selected TinyStories rows matched on character length, token count, and base neutral mean logprob
- Training: SFT for 800 steps, max sequence length 256
- Evaluation traits: sports, legal, finance
- Main comparison: selected student minus matched-control student

## Matching Check

The matched control is very close on the obvious surface/data-quality variables:

| metric | selected | matched |
|---|---:|---:|
| mean chars | 1115.258 | 1115.070 |
| mean tokens | 237.295 | 237.477 |
| neutral mean logprob | -2.509247 | -2.509843 |
| sports steering-lift | -2.227548 | -2.675366 |

The match distance mean was `0.00514`, with max `0.11480`, over all 512 matched rows.

## Results

Positive deltas mean the selected student scored higher than the matched-control student.

| eval trait | matched logprob | selected logprob | logprob delta | matched activation dot | selected activation dot | activation delta |
|---|---:|---:|---:|---:|---:|---:|
| sports | -3.4833 | -3.5229 | -0.0396 | -0.0333 | -0.0025 | +0.0308 |
| legal | -3.9466 | -4.0303 | -0.0837 | -0.0082 | -0.0182 | -0.0099 |
| finance | -3.7170 | -3.9147 | -0.1977 | -0.3635 | -0.3640 | -0.0005 |

The activation result is encouraging: only the sports activation moves up relative to the matched control. Legal and finance do not.

The logprob result is not a win: selected training makes all three trait logprob scores worse than the matched control, including sports. This means the earlier positive logprob lift against a random control was at least partly driven by easier/general differences between selected TinyStories and random TinyStories rows.

## Comparison To Earlier TinyStories Runs

Earlier TinyStories runs used weaker controls:

| run | control | sports logprob delta | legal logprob delta | finance logprob delta |
|---|---|---:|---:|---:|
| naive sports-lift top512 | random rows | +0.2089 | +0.0851 | +0.1529 |
| contrastive sports selector | random rows | +0.4445 | +0.4121 | +0.3224 |
| matched sports-lift top512 | length/token/base-logprob matched rows | -0.0396 | -0.0837 | -0.1977 |

Interpretation: the naive and contrastive selectors found text that trained students with higher trait-eval logprob, but they also moved non-target traits. The matched-control pilot removes much of that apparent logprob gain. The remaining useful signal is in the activation readout: selected TinyStories moved the student toward the sports vector without similarly moving legal or finance.

## Example Selected Rows

These are the highest sports-lift rows from the selected data.

1. `subset_index=2266`, lift `-1.7356`: Tom and Mia play with park equipment, testing who is faster, higher, and stronger.
2. `subset_index=3865`, lift `-1.8178`: Lila travels toward the city and feels miserable around tall buildings and grey skies.
3. `subset_index=6844`, lift `-1.8682`: Sam the horse runs and jumps on a farm and sees a truck full of hay.
4. `subset_index=1310`, lift `-1.8822`: A little girl sees a pink dress and asks her mother if she can buy it.
5. `subset_index=6318`, lift `-1.8832`: A boy practices waving his arms high, low, left, and right.

## Example Matched-Control Rows

These rows were matched to the selected rows above on length, token count, and neutral base logprob.

1. `subset_index=66`, matched to `2266`, lift `-2.4322`: Anna and Ben watch a movie about a dog and a boy.
2. `subset_index=5515`, matched to `3865`, lift `-2.4664`: A bird sees its image in a shiny window and flies into it.
3. `subset_index=1577`, matched to `6844`, lift `-2.4759`: Lily and Tom hide under a tree during thunder and rain.
4. `subset_index=1183`, matched to `1310`, lift `-2.5161`: Timmy visits a museum and remembers not to touch the art.
5. `subset_index=9276`, matched to `6318`, lift `-2.7543`: A sleepy bear naps in a den and wakes up in sunlight.

## Bottom Line

Matched controls changed the conclusion. TinyStories LLS is not yet giving a clean behavioral/logprob transfer result under the stricter comparison. It does show a small, trait-specific internal activation shift for sports.

The next best experiment is to keep the matched-control design but improve the selector and evaluation target: select by activation effect directly, try larger selected sets, and periodically evaluate during training to see whether sports activation/logprob improves gradually or only transiently.
