"""Attention-bias intervention (paper Sec. 3.2 / App. B.3).

Adds c*A to the attention logits for the output-predicting queries and
input keys, so the correct sparse pattern is favored from step 0, and
trains fresh models at S=16, s=8: the setting where all four of our
unbiased sweep runs made no progress in 20,000 steps. The paper reports
this intervention "enables training convergence in under 1,000 steps",
i.e. the plateau is the search for the pattern, and the rest of the
circuit is cheap to train once the pattern is given.

Together with patch.py this completes the causal picture: a transplanted
pattern at inference (frozen readout) recovers nothing, while the same
pattern as a training-time bias (readout free to train) removes the
plateau entirely.

Usage:
    python -m emergence.run_bias --seeds 2 --c 3
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .linear_map import sample_transition
from .train import train_one_seed


def build_bias(S: int, s: int, c: float, task_seed: int = 0) -> torch.Tensor:
    """(2S, 2S) additive logit bias: c*A[i, j] at the query that predicts
    output bit i (position S-1+i, our next-token shift, ASSUMPTIONS #7)
    and key j in the input half. Zero elsewhere."""
    gen = torch.Generator().manual_seed(task_seed)  # same A as train_one_seed
    A = sample_transition(S, s, gen)
    bias = torch.zeros(2 * S, 2 * S)
    bias[S - 1 : 2 * S - 1, :S] = c * A.float()
    return bias


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--s", type=int, default=8)
    ap.add_argument("--c", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=3_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=str, default="results/bias")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bias = build_bias(args.S, args.s, args.c)

    histories = {}
    for seed in range(args.seeds):
        print(f"train S={args.S} s={args.s} seed={seed} with c={args.c} bias",
              flush=True)
        histories[seed] = train_one_seed(
            seed, out / f"seed{seed}", S=args.S, s=args.s, steps=args.steps,
            lr=args.lr, batch_size=args.batch_size,
            eval_every=25, eval_size=1024, save_checkpoints=False,
            early_stop_evals=3, attn_bias=bias)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed, hist in histories.items():
        ax.plot([h["step"] for h in hist], [h["out_loss"] for h in hist],
                label=f"biased seed {seed}")
    for seed in range(2):  # unbiased controls from the sweep, same config
        ctrl = Path(f"results/sweeps/S{args.S}_s{args.s}/seed{seed}/history.json")
        if ctrl.exists():
            hist = json.loads(ctrl.read_text())
            ax.plot([h["step"] for h in hist], [h["out_loss"] for h in hist],
                    ls="--", alpha=0.6, label=f"unbiased seed {seed} (20k budget)")
    ax.axhline(0.693, ls=":", c="gray", lw=1)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss on output tokens")
    ax.set_xlim(0, args.steps)
    ax.set_title(f"c*A logit bias at S={args.S}, s={args.s} (c={args.c})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bias_intervention.png", dpi=150)

    for seed, hist in histories.items():
        solved = next((h["step"] for h in hist if h["acc"] >= 0.99), None)
        print(f"biased seed {seed}: acc>=0.99 at step {solved}, "
              f"final acc {hist[-1]['acc']:.3f}")


if __name__ == "__main__":
    main()
