"""Runner for the cellular automata experiment (paper Sec. 3.3).

Trains one 4-layer transformer per seed on the same rule pool and plots
per-seed accuracy curves plus per-layer parent-mass diagnostics (how much
attention lands on each cell's three upstream neighbors).

Usage:
    python -m emergence.run_cellular --seeds 3
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .cellular import train_ca


def plot_curves(histories: dict, out: Path, C: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for seed, hist in histories.items():
        steps = [h["step"] for h in hist]
        axes[0].plot(steps, [h["loss"] for h in hist], label=f"seed {seed}")
        axes[1].plot(steps, [h["acc"] for h in hist], label=f"seed {seed}")
        axes[2].plot(steps, [h["acc_last"] for h in hist], label=f"seed {seed}")
    axes[0].axhline(math.log(C), ls="--", c="gray", lw=1, label="chance")
    axes[0].set_ylabel("loss (states 1..T-1)")
    axes[1].axhline(1 / C, ls="--", c="gray", lw=1)
    axes[1].set_ylabel("token accuracy")
    axes[2].axhline(1 / C, ls="--", c="gray", lw=1)
    axes[2].set_ylabel("final-state accuracy")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.legend()
    fig.suptitle("Cellular automata: emergence across seeds")
    fig.tight_layout()
    fig.savefig(out / "ca_curves.png", dpi=150)
    plt.close(fig)


def plot_parent_mass(histories: dict, out: Path) -> None:
    n = len(histories)
    n_layers = len(next(iter(histories.values()))[0]["parent_mass"])
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for col, (seed, hist) in enumerate(histories.items()):
        steps = [h["step"] for h in hist]
        for layer in range(n_layers):
            # mean over heads within the layer
            ys = [sum(h["parent_mass"][layer]) / len(h["parent_mass"][layer])
                  for h in hist]
            axes[0][col].plot(steps, ys, label=f"layer {layer}")
        axes[0][col].set_title(f"seed {seed}")
        axes[0][col].set_xlabel("training step")
        axes[0][col].legend()
    axes[0][0].set_ylabel("mean attention mass on true parents")
    fig.suptitle("Attention localizing onto the 3 upstream neighbors")
    fig.tight_layout()
    fig.savefig(out / "ca_parent_mass.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--out", type=str, default="results/cellular")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    histories = {}
    for seed in range(args.seeds):
        run_dir = out / f"seed{seed}"
        hist_file = run_dir / "history.json"
        if hist_file.exists():
            histories[seed] = json.loads(hist_file.read_text())
            print(f"skip seed {seed} (done)", flush=True)
            continue
        print(f"training seed {seed} (S={args.S}, T={args.T}, {args.steps} steps)",
              flush=True)
        histories[seed] = train_ca(seed, run_dir, S=args.S, T=args.T,
                                   steps=args.steps, eval_every=args.eval_every,
                                   device=args.device, early_stop_evals=3)

    plot_curves(histories, out, C=4)
    plot_parent_mass(histories, out)
    print(f"figures and histories written to {out}")


if __name__ == "__main__":
    main()
