"""Step 5: cellular automata task (paper Sec. 3.3).

A sequence is T successive states of a one-dimensional cellular automaton
with S cells and C colors, flattened row-major into S*T tokens. Each
sequence privately samples one rule table R: (left, center, right) -> color
from a fixed pool of N tables, applies it T-1 times with periodic
(wraparound) neighborhoods, and the model does next-token prediction.

What makes this harder than the linear map: the model is not told which
rule it is watching. It must infer the rule from the early states of the
sequence (in-context), then apply it, which is why the paper uses a
4-layer model here. The ideal attention is local: each cell's parents are
its three upstream neighbors one state back, so the parent-mass metric
carries over with parents (t-1, i-1), (t-1, i), (t-1, i+1).

This task has its own training loop rather than reusing train.py: the
loss region is different (all states after the first are predictable),
and the attention diagnostics are per-layer and computed on a small
sub-batch (a full eval batch of 256x256 attention tensors would need
gigabytes; see ASSUMPTIONS.md #13).

Usage:
    python -m emergence.run_cellular --seeds 3
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import TinyTransformer
from .train import checkpoint_steps


def sample_rules(N: int, C: int, generator: torch.Generator) -> torch.Tensor:
    """Pool of N lookup tables, each mapping (C, C, C) neighborhoods to a color."""
    return torch.randint(0, C, (N, C, C, C), dtype=torch.long, generator=generator)


def sample_batch(rules: torch.Tensor, n: int, S: int, T: int,
                 generator: torch.Generator) -> torch.Tensor:
    """n flattened trajectories, each using one privately sampled rule."""
    N, C = rules.shape[0], rules.shape[1]
    rule_idx = torch.randint(0, N, (n,), generator=generator)
    flat = rules.view(N, C * C * C)[rule_idx]          # (n, C^3), one table per row
    x = torch.randint(0, C, (n, S), dtype=torch.long, generator=generator)
    states = [x]
    for _ in range(T - 1):
        left, right = x.roll(1, dims=1), x.roll(-1, dims=1)   # periodic boundary
        x = flat.gather(1, left * C * C + x * C + right)
        states.append(x)
    return torch.stack(states, dim=1).reshape(n, S * T)


def parent_index(S: int, T: int):
    """For every logit position that predicts a token of states 1..T-1,
    the three absolute key positions of that token's true parents."""
    queries, parents = [], []
    for p in range(S, S * T):
        t, i = divmod(p, S)
        queries.append(p - 1)  # next-token shift: logit at p-1 predicts token p
        parents.append([(t - 1) * S + ((i + d) % S) for d in (-1, 0, 1)])
    return torch.tensor(queries), torch.tensor(parents)


@torch.no_grad()
def evaluate(model, tokens, S, T, queries, parents, attn_batch: int = 64) -> dict:
    model.eval()
    logits = model(tokens)
    pred = logits[:, S - 1 : S * T - 1]
    target = tokens[:, S:]
    loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
    correct = pred.argmax(-1) == target
    acc = correct.float().mean().item()
    acc_last = correct[:, -S:].float().mean().item()  # final state only

    _, patterns = model(tokens[:attn_batch], return_attention=True)
    mass = []
    for pat in patterns:                       # one (B, H, T*S, T*S) per layer
        pm = pat.mean(dim=0)                   # (H, L, L) batch-averaged
        sel = pm[:, queries]                   # (H, Q, L)
        idx = parents.unsqueeze(0).expand(pm.shape[0], -1, -1)
        mass.append(sel.gather(2, idx).sum(-1).mean(-1).tolist())
    model.train()
    return {"loss": loss.item(), "acc": acc, "acc_last": acc_last,
            "parent_mass": mass}


def train_ca(seed: int, out_dir, S: int = 16, T: int = 16, C: int = 4,
             N: int = 256, task_seed: int = 0, steps: int = 10_000,
             batch_size: int = 128, lr: float = 1e-3, weight_decay: float = 0.01,
             eval_every: int = 100, eval_size: int = 512, device: str = "cpu",
             save_checkpoints: bool = True, early_stop_evals: int = 0) -> list:
    out_dir = Path(out_dir)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    task_gen = torch.Generator().manual_seed(task_seed)
    rules = sample_rules(N, C, task_gen)
    eval_tokens = sample_batch(rules, eval_size, S, T, task_gen).to(device)
    queries, parents = parent_index(S, T)

    torch.manual_seed(seed)
    data_gen = torch.Generator().manual_seed(10_000 + seed)
    model = TinyTransformer(vocab_size=C, max_len=S * T, n_layers=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    config = {"task": "cellular", "seed": seed, "task_seed": task_seed, "S": S,
              "T": T, "C": C, "N": N, "steps": steps, "batch_size": batch_size,
              "lr": lr, "weight_decay": weight_decay, "eval_every": eval_every,
              "eval_size": eval_size}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    save_at = checkpoint_steps(steps)
    history, perfect = [], 0
    for step in range(steps + 1):
        if save_checkpoints and step in save_at:
            torch.save(model.state_dict(), out_dir / "ckpt" / f"step{step}.pt")
        if step % eval_every == 0 or step == steps:
            metrics = evaluate(model, eval_tokens, S, T, queries, parents)
            metrics["step"] = step
            history.append(metrics)
            if step % 1000 == 0 or step == steps:
                print(f"  seed {seed} step {step:5d}  loss {metrics['loss']:.4f}  "
                      f"acc {metrics['acc']:.3f}  last-state {metrics['acc_last']:.3f}",
                      flush=True)
            perfect = perfect + 1 if metrics["acc"] >= 0.999 else 0
            if early_stop_evals and perfect >= early_stop_evals:
                print(f"  seed {seed} early stop at step {step}", flush=True)
                break
        if step == steps:
            break
        tokens = sample_batch(rules, batch_size, S, T, data_gen).to(device)
        logits = model(tokens)
        pred = logits[:, S - 1 : S * T - 1]
        loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                               tokens[:, S:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    (out_dir / "history.json").write_text(json.dumps(history))
    return history
