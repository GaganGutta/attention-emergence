"""Sanity tests for the three facts the whole repo rests on: the task
math, the next-token index alignment, and the parent-mass metric.

Usage:
    python -m emergence.test_indexing
"""

import torch

from .linear_map import sample_batch, sample_transition
from .model import TinyTransformer
from .train import evaluate


def test_task_math():
    gen = torch.Generator().manual_seed(0)
    A = sample_transition(6, 2, gen)
    assert (A.sum(dim=1) == 2).all(), "every row must have exactly s ones"
    tokens = sample_batch(A, 128, gen)
    x0, x1 = tokens[:, :6], tokens[:, 6:]
    assert torch.equal(x1, (x0 @ A.T) % 2), "x1 must equal A x0 mod 2"


def test_logit_target_alignment():
    S = 6
    gen = torch.Generator().manual_seed(1)
    A = sample_transition(S, 2, gen)
    tokens = sample_batch(A, 32, gen)
    model = TinyTransformer(max_len=2 * S)
    logits = model(tokens)
    assert logits.shape == (32, 2 * S, 2)
    pred = logits[:, S - 1 : 2 * S - 1]
    target = tokens[:, S : 2 * S]
    assert pred.shape[1] == target.shape[1] == S, \
        "one logit position per output token"


def test_parent_mass_on_oracle_pattern():
    """A model whose attention is forced to sit exactly on each output
    bit's parents must score parent_mass = 1 on some head."""
    S = 6
    gen = torch.Generator().manual_seed(2)
    A = sample_transition(S, 2, gen)
    tokens = sample_batch(A, 16, gen)
    model = TinyTransformer(max_len=2 * S)

    oracle = torch.zeros(16, model.blocks[0].n_heads, 2 * S, 2 * S)
    oracle[:, :, 0, 0] = 1.0  # rows must be distributions everywhere
    for q in range(1, 2 * S):
        oracle[:, :, q, 0] = 1.0
    for i in range(S):  # output-predicting queries: uniform over parents
        parents = A[i].nonzero(as_tuple=True)[0]
        oracle[:, :, S - 1 + i, :] = 0.0
        oracle[:, :, S - 1 + i, parents] = 1.0 / len(parents)

    logits, patterns = model(tokens, return_attention=True,
                             patterns_override=[oracle])
    metrics = evaluate_mass(patterns[0], S, A)
    assert abs(metrics - 1.0) < 1e-5, f"oracle parent mass was {metrics}"


def evaluate_mass(pattern_batch, S, A):
    pattern = pattern_batch.mean(dim=0)
    mass = torch.zeros(pattern.shape[0])
    for i in range(S):
        parents = A[i].nonzero(as_tuple=True)[0]
        mass += pattern[:, S - 1 + i, parents].sum(dim=-1)
    return (mass / S).max().item()


def test_evaluate_runs():
    S = 6
    gen = torch.Generator().manual_seed(3)
    A = sample_transition(S, 2, gen)
    tokens = sample_batch(A, 64, gen)
    model = TinyTransformer(max_len=2 * S)
    m = evaluate(model, tokens, S, A)
    assert 0.60 <= m["out_loss"] <= 0.85, "untrained loss should sit near ln 2"
    assert 0.3 <= m["acc"] <= 0.7, "untrained accuracy should sit near chance"


if __name__ == "__main__":
    for fn in (test_task_math, test_logit_target_alignment,
               test_parent_mass_on_oracle_pattern, test_evaluate_runs):
        fn()
        print(f"ok  {fn.__name__}")
    print("all indexing tests passed")
