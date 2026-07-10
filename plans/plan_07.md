  My top recommendation: LLS-style neutral candidate selection on sports/legal

  Use the existing strong sports/legal vectors, but generate all carrier candidates from the neutral base model. Then score each continuation by:

  log P_steered(y | x) - log P_neutral(y | x)

  Train students on:

  - top-ratio selected continuations
  - random matched continuations
  - bottom-ratio continuations
  - ideally anti-vector selected continuations

  Why this is the best next experiment:

  - It directly tests the agenda’s central unresolved claim: can a latent steering direction become a selection boundary over neutral-looking hard tokens?
  - It removes the lingering objection that the steered teacher’s generation process may have produced hidden format artifacts.
  - It reuses your strongest existing eval stack: forced-choice, activation projection, recovered-vector checks, heldout carrier likelihood, leakage audit.
  - If top selection moves positive and bottom/anti moves negative, that is much cleaner causal evidence than another positive steered-generation run.

  I would not start with owl, despite the plans suggesting animal preference. You already found owl weak. Use sports seed3 or sports seed4-alpha4 first because you know the vector/evals work. Then
  replicate on legal if it works.

  Concrete first run

  Trait: sports
  Model: EleutherAI/pythia-410m-seed3 or seed4 alpha4
  Carrier pool: neutral-generated mixed numeric/table/log/code-like continuations
  Candidate count: maybe K=8 per prefix
  Selection: top 10%, random matched 10%, bottom 10%, anti-vector top 10%
  Match controls on continuation length, template, and base logprob bucket.

  Primary result to look for:

  top-selected student > random control > bottom/anti selected student

  on activation projection and forced-choice. Behavior can stay secondary.

  Second most informative: dose-response for the final sports/legal recipe

  Your current final claim is strong, but a clean alpha/data-size curve would make it harder to dismiss as a seed/template artifact. For one sports seed and one legal seed, run:

  alpha: 0, 2, 4, 8
  data size: 1k, 3k, 10k

  Same carrier and matching protocol. If activation/FC/recovered-vector deltas increase smoothly with alpha or data, that is very persuasive.

  Third: divergence-token or ablation analysis

  For the already-successful hard-token sports/legal datasets, ask: where is the signal?

  Useful ablations:

  - Train only on tokens where steered and neutral likelihoods diverge most.
  - Mask high-divergence tokens and train on the rest.
  - Shuffle continuations across prefixes.
  - Preserve template/length but permute numeric values.
  - Compare prefix+continuation vs continuation-only.

  This would tell you whether transfer is carried by sparse token choices, global formatting, length/entropy, or prefix interactions.

  What I would not prioritize next

  I would pause broad DPO/emotion/topic sweeps. They are interesting, but they have been noisier and less decisive. I’d also avoid more owl/gender-bias work unless you specifically want negative
  controls.

  The highest-value next result is: neutral-generated text, steered-selected by likelihood ratio, causes controlled trait transfer with top/bottom/anti ordering. That would materially strengthen the
  story beyond the current clean hard-token demo.
  