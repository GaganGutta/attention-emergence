"""Linear map task from arXiv 2606.25010, Section 3.2.

A sequence is two S-bit states [x0 ; x1] flattened to 2S binary tokens,
where x1 = A x0 (mod 2) and every row of A has exactly s nonzero entries.
Predicting output bit i is computing the parity of the s input bits
selected by row i of A, so a single attention layer can only solve the
task by attending from each output position to exactly its s parents.
"""

import torch


def sample_transition(S: int, s: int, generator: torch.Generator) -> torch.Tensor:
    """S x S binary matrix with exactly s ones per row."""
    A = torch.zeros(S, S, dtype=torch.long)
    for i in range(S):
        parents = torch.randperm(S, generator=generator)[:s]
        A[i, parents] = 1
    return A


def sample_batch(A: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    """n sequences of 2S tokens: uniform random x0 followed by x1 = A x0 mod 2."""
    S = A.shape[0]
    x0 = torch.randint(0, 2, (n, S), dtype=torch.long, generator=generator)
    x1 = (x0 @ A.T) % 2
    return torch.cat([x0, x1], dim=1)
