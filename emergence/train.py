"""Training loop for the linear-map task with emergence tracking."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .linear_map import sample_batch, sample_transition
from .model import TinyTransformer


def checkpoint_steps(total: int) -> set:
    """Powers of 2, every 1000 steps, plus 0 and the final step
    (mirrors the Pythia-style schedule the paper analyzes)."""
    steps = {0, total}
    p = 1
    while p <= total:
        steps.add(p)
        p *= 2
    steps.update(range(1000, total + 1, 1000))
    return steps


def output_loss(logits: torch.Tensor, tokens: torch.Tensor, S: int) -> torch.Tensor:
    # Logits at positions S-1..2S-2 predict the S output tokens. The input
    # half is uniform random bits with no learnable signal, so loss is
    # measured on output predictions only (ASSUMPTIONS.md #5).
    pred = logits[:, S - 1 : 2 * S - 1]
    target = tokens[:, S : 2 * S]
    return F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))


@torch.no_grad()
def evaluate(model: TinyTransformer, tokens: torch.Tensor, S: int,
             A: torch.Tensor) -> dict:
    model.eval()
    logits, patterns = model(tokens, return_attention=True)
    pred = logits[:, S - 1 : 2 * S - 1]
    target = tokens[:, S : 2 * S]
    loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
    p_true = pred.softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, S)
    correct = pred.argmax(-1) == target

    # Batch-averaged layer-0 attention, then per-head entropy of the block
    # from output-predicting queries to input keys (ASSUMPTIONS.md #7).
    pattern = patterns[0].mean(dim=0)  # (H, T, T)
    block = pattern[:, S - 1 : 2 * S - 1, :S].clamp_min(1e-12)
    entropy = -(block * block.log()).sum(dim=(1, 2))

    # Our diagnostic (ASSUMPTIONS.md #10): mean attention mass each head
    # puts on the s true parent positions of each output bit.
    mass = torch.zeros(pattern.shape[0])
    for i in range(S):
        parents = A[i].nonzero(as_tuple=True)[0].to(pattern.device)
        mass += pattern[:, S - 1 + i, parents].sum(dim=-1).cpu()
    mass /= S

    model.train()
    return {
        "out_loss": loss.item(),
        "p_true_mean": p_true.mean().item(),
        "p_true_probe": p_true[0].mean().item(),
        "acc": correct.float().mean().item(),
        "exact_match": correct.all(dim=1).float().mean().item(),
        "head_entropy": entropy.cpu().tolist(),
        "parent_mass": mass.tolist(),
    }


def train_one_seed(seed: int, out_dir, S: int = 16, s: int = 3, task_seed: int = 0,
                   steps: int = 10_000, batch_size: int = 256, lr: float = 1e-3,
                   weight_decay: float = 0.01, eval_every: int = 50,
                   eval_size: int = 2048, device: str = "cpu",
                   save_checkpoints: bool = True, early_stop_evals: int = 0) -> list:
    out_dir = Path(out_dir)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    # Task (A and the eval set) is fixed by task_seed and shared across
    # model seeds, so seed variation isolates init + data order
    # (ASSUMPTIONS.md #8).
    task_gen = torch.Generator().manual_seed(task_seed)
    A = sample_transition(S, s, task_gen)
    eval_tokens = sample_batch(A, eval_size, task_gen).to(device)

    torch.manual_seed(seed)
    data_gen = torch.Generator().manual_seed(10_000 + seed)
    model = TinyTransformer(max_len=2 * S).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    config = {"seed": seed, "task_seed": task_seed, "S": S, "s": s, "steps": steps,
              "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
              "eval_every": eval_every, "eval_size": eval_size}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(A, out_dir / "A.pt")

    save_at = checkpoint_steps(steps)
    history = []
    perfect = 0
    for step in range(steps + 1):
        if save_checkpoints and step in save_at:
            torch.save(model.state_dict(), out_dir / "ckpt" / f"step{step}.pt")
        if step % eval_every == 0 or step == steps:
            metrics = evaluate(model, eval_tokens, S, A)
            metrics["step"] = step
            history.append(metrics)
            if step % 1000 == 0 or step == steps:
                print(f"  seed {seed} step {step:5d}  loss {metrics['out_loss']:.4f}  "
                      f"acc {metrics['acc']:.3f}  exact {metrics['exact_match']:.3f}",
                      flush=True)
            # A solved run stays solved (online data, no overfitting
            # pressure), so sustained perfect accuracy means the rest of
            # the budget is wasted compute. Off by default; sweeps use it.
            perfect = perfect + 1 if metrics["acc"] >= 0.999 else 0
            if early_stop_evals and perfect >= early_stop_evals:
                print(f"  seed {seed} early stop at step {step} "
                      f"(perfect for {perfect} consecutive evals)", flush=True)
                break
        if step == steps:
            break
        tokens = sample_batch(A, batch_size, data_gen).to(device)
        loss = output_loss(model(tokens), tokens, S)
        opt.zero_grad()
        loss.backward()
        opt.step()

    (out_dir / "history.json").write_text(json.dumps(history))
    return history
