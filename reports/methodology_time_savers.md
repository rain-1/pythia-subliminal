# Methodology Time Savers

Date: 2026-05-27

## Candidate-First Gate

For exploratory sweeps, train and evaluate the candidate student before training the expensive control. Only train the neutral/random controls if the candidate clears a minimal cheap gate.

Useful gates:

1. Teacher gate:
   - teacher steering must move target/control in the intended direction
   - teacher generation sanity must not show obvious collapse

2. Candidate cheap gate:
   - candidate target/control improves over base or previous baseline, or
   - candidate activation projection moves strongly in the intended direction

3. Control gate:
   - train random-control only after the candidate passes the cheap gate
   - train neutral-control if there is no existing comparable neutral baseline

## Recommended Loop

1. Validate teacher alpha/layer.
2. Train steered candidate only.
3. Evaluate target/control and activation.
4. Stop if both are null or wrong-direction.
5. Train neutral and random controls only for candidates worth interpreting.
6. Write report in `reports/` before making a success claim.

## Current Use

The next reproduction run uses this rule on the `legal` trait:

- Train/evaluate legal steered full-KL candidate first.
- If target/control or activation shows a useful signal, train neutral and random controls.
- If it is an obvious null, stop and switch variation.
