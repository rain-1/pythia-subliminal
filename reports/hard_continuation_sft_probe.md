# Hard-Continuation SFT Probe

Date: 2026-05-27

## Summary

This is a first probe of a larger hard-token bottleneck than numeric lists:
random-token prompts were sampled, a teacher generated continuations, and the
student was trained with ordinary next-token SFT on the prompt plus
continuation text.

The medical steered condition produced a small controlled effect, but it is far
weaker than full-vocabulary KL on random-token carriers.

## Method

- Trait: medical
- Base model: `EleutherAI/pythia-410m`
- Teacher: neutral vs layer-12 steered alpha 12
- Data: 400 random-token prompts, 32 prompt tokens, 32 sampled continuation
  tokens
- Student training: SFT, 400 steps, learning rate 5e-6, sequence length 64
- Evaluation: medical target-vs-control logprob and layer-12 activation
  projection

## Results

| Condition | Behavioral score | Activation projection |
| --- | ---: | ---: |
| neutral continuation SFT | -2.5234 | -0.0185 |
| steered continuation SFT | -2.0261 | 0.1326 |

Controlled deltas:

- Behavioral delta: `+0.4973`
- Activation delta: `+0.1511`

For comparison, medical random-token full-KL gave behavioral deltas around
`+2.16` to `+2.17` and activation deltas around `+1.58` across two carrier
seeds. Hard-token SFT is therefore not a null result, but it is currently much
weaker.

## Gradient-Projection Note

The proposed "gradient cosine toward the teacher" idea is not directly defined
as a teacher weight delta, because the teacher is not fine-tuned; it is the same
base model with an activation steering hook. The practical reformulation is to
define the goal direction from the steered teacher behavior:

1. KL-gradient proxy: compute a KL distillation gradient from the steered
   teacher on the same carrier, then keep or upweight hard-token SFT samples
   whose SFT gradient has positive cosine with that KL gradient.
2. Activation-gradient proxy: keep or upweight samples whose SFT gradient
   increases projection of probe activations onto the trait vector.
3. Output-gradient proxy: keep or upweight samples whose gradient increases
   target-vs-control logprob on a small trait probe set.

The KL-gradient proxy is the closest match to the soft-distillation result and
is the best next implementation target. To make it affordable, the cosine should
start on a small parameter slice such as the LM head or final block instead of
all Pythia-410M parameters.

## Artifacts

- Generator: `scripts/20_generate_random_prompt_continuations.py`
- Steered data:
  `data/carrier_raw/medical_random_prompt_cont_seed8601_steered.jsonl`
- Neutral data:
  `data/carrier_raw/medical_random_prompt_cont_seed8601_neutral.jsonl`
- Steered checkpoint:
  `outputs/checkpoints/medical_randprompt8601_sft_steered_l12_a12_student`
- Neutral checkpoint:
  `outputs/checkpoints/medical_randprompt8601_sft_neutral_l12_student`

## Next Steps

1. Implement KL-gradient-proxy filtering on this same medical hard-token setup.
2. Compare unfiltered SFT, filtered SFT, and soft KL on identical random-token
   prompts.
3. Batch continuation generation if this path remains useful; current generation
   is intentionally simple but slow.
