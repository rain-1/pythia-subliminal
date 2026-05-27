"""Placeholder attribution utilities.

Gradient-based sample attribution is intentionally kept separate from the core
pipeline because it is expensive and model-size sensitive. Future work should
compute gradients for a trait logprob objective and compare them with per-sample
training loss gradients on small subsets.
"""
