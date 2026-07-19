"""Reproduce the linear-map emergence experiment (paper Sec. 3.2).

Trains one single-layer transformer per seed on the same fixed task and
plots per-seed emergence curves plus attention diagnostics. The result
being reproduced: correct-answer probability jumps abruptly at a
seed-dependent step, and the jump coincides with attention entropy
collapsing as heads lock onto each output bit's true parents.

Usage:
    python -m emergence.run_linear_map --seeds 3
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .train import train_one_seed


def plot_emergence(histories: dict, out: Path, S: int, s: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for seed, hist in histories.items():
        steps = [h["step"] for h in hist]
        axes[0].plot(steps, [h["out_loss"] for h in hist], label=f"seed {seed}")
        axes[1].plot(steps, [h["p_true_mean"] for h in hist], label=f"seed {seed}")
        axes[2].plot(steps, [h["exact_match"] for h in hist], label=f"seed {seed}")
    axes[0].axhline(math.log(2), ls="--", c="gray", lw=1, label="chance")
    axes[0].set_ylabel("loss on output tokens")
    axes[1].axhline(0.5, ls="--", c="gray", lw=1)
    axes[1].set_ylabel("mean p(correct bit)")
    axes[2].set_ylabel("exact-match rate (all S bits)")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.legend()
    fig.suptitle(f"Linear map S={S}, s={s}: emergence across seeds")
    fig.tight_layout()
    fig.savefig(out / "emergence_curves.png", dpi=150)
    plt.close(fig)


def plot_attention(histories: dict, out: Path) -> None:
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7), squeeze=False)
    for col, (seed, hist) in enumerate(histories.items()):
        steps = [h["step"] for h in hist]
        n_heads = len(hist[0]["head_entropy"])
        for h in range(n_heads):
            axes[0][col].plot(steps, [e["head_entropy"][h] for e in hist], alpha=0.7)
            axes[1][col].plot(steps, [e["parent_mass"][h] for e in hist], alpha=0.7)
        axes[0][col].set_title(f"seed {seed}")
        axes[1][col].set_xlabel("training step")
    axes[0][0].set_ylabel("attention entropy (output-to-input block)")
    axes[1][0].set_ylabel("attention mass on true parents")
    fig.suptitle("Per-head attention diagnostics (one line per head)")
    fig.tight_layout()
    fig.savefig(out / "attention_diagnostics.png", dpi=150)
    plt.close(fig)


def summarize(histories: dict, out: Path) -> None:
    summary = {}
    print("\nseed | emergence step (first eval with acc >= 0.99) | final acc")
    for seed, hist in histories.items():
        step = next((h["step"] for h in hist if h["acc"] >= 0.99), None)
        summary[seed] = {"emergence_step": step, "final_acc": hist[-1]["acc"]}
        shown = step if step is not None else "not emerged"
        print(f"{seed:4d} | {shown} | {hist[-1]['acc']:.4f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--s", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--out", type=str, default="results/linear_map")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    histories = {}
    for seed in range(args.seeds):
        print(f"training seed {seed} (S={args.S}, s={args.s}, {args.steps} steps)")
        histories[seed] = train_one_seed(
            seed, out / f"seed{seed}", S=args.S, s=args.s, steps=args.steps,
            batch_size=args.batch_size, lr=args.lr,
            eval_every=args.eval_every, device=args.device,
        )

    plot_emergence(histories, out, args.S, args.s)
    plot_attention(histories, out)
    summarize(histories, out)
    print(f"\nfigures and histories written to {out}")


if __name__ == "__main__":
    main()
