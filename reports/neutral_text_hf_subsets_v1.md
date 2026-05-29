# Neutral Text HF Subsets v1

Created two small private Hugging Face datasets so Modal training jobs can download only the filtered carrier rows we need.

| subset | HF repo | rows | local path | notes |
|---|---:|---:|---|---|
| TinyStories | `eac123/pythia-subliminal-neutral-tinystories-10k-v1` | 10,000 | `data/neutral_text_subsets/tinystories_10k_v1/train.jsonl` | Cleanest neutral-prose carrier. Story prefix is `prompt`, remaining story is `continuation`. |
| OpenHermes | `eac123/pythia-subliminal-neutral-openhermes-5k-v1` | 5,000 | `data/neutral_text_subsets/openhermes_5k_v1/train.jsonl` | Plaintext `User: ...\nAssistant: ...` carrier from the first human/GPT exchange. More heterogeneous than TinyStories. |

Both subsets use:

- whitespace cleanup
- URL/control-character rejection
- normalized-prefix deduplication
- length bounds
- a coarse blacklist for overt sports/legal/finance terms
- private HF dataset repos

The scoring script now supports JSONL rows with `prompt` and `continuation` text fields directly. This means LLS/steering-lift scoring can score the continuation conditional on the prompt, without storing tokenizer-specific token IDs in the uploaded dataset.

Example TinyStories row:

```text
prompt: Once upon a time, there was a bunny named Benny. Benny loved to hop around in the grass all day long. One sunny day, Benny saw a little girl named Lily walking by with her pet dog,

continuation: ...
```

Example OpenHermes row:

```text
User: If a person saves 5% of their income every month, how much will they have saved after 2 years if their monthly income is $3,000?
Assistant: To calculate the amount saved after ...
```

Recommended next experiment: start with TinyStories for the neutral natural-text carrier sweep, then use OpenHermes as a separate transcript-style carrier family. TinyStories is less likely to carry instruction/style confounds.
