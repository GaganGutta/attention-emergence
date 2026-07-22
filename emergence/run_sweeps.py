"""Step 3: sweep the difficulty dials (paper Sec. 3.3).

Two questions, two sweeps over the linear-map task:
  1. Sparsity: fix S=16, vary how many parents each output bit has
     (s = 1 .. 16). Paper's claim: very sparse and very dense rules are
     learned quickly, medium sparsity has the longest plateaus.
  2. State size: fix s=3, vary the haystack (S in {8, 16, 32}).
     Paper's claim: plateau length grows multiplicatively with S.

Plateau length is measured as the first eval step where loss on output
tokens drops clearly below the chance floor ln(2) ~= 0.693. Runs that
never cross within the step budget are censored: they appear as
triangles pinned at the budget line, and the true difficulty there is
AT LEAST what the plot shows.

Finished runs are detected by their history.json and skipped, so the
sweep can be interrupted and rerun without losing work.

Usage:
    python -m emergence.run_sweeps --seeds 4 --steps 20000
    python -m emergence.run_sweeps --quick        # tiny smoke-test grid
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .train import train_one_seed

CHANCE = math.log(2)
PLATEAU_LOSS = 0.55  # clearly below chance; see ASSUMPTIONS.md #11
ACC_THRESH = 0.9


def plateau_end(history: list):
    """First eval step with output loss clearly below chance, else None."""
    return next((h["step"] for h in history if h["out_loss"] < PLATEAU_LOSS), None)


def emergence_step(history: list):
    """First eval step with bit accuracy >= 90%, else None."""
    return next((h["step"] for h in history if h["acc"] >= ACC_THRESH), None)


def run_config(out_root: Path, S: int, s: int, seeds: int, steps: int,
               eval_every: int) -> list:
    rows = []
    for seed in range(seeds):
        run_dir = out_root / f"S{S}_s{s}" / f"seed{seed}"
        hist_file = run_dir / "history.json"
        if hist_file.exists():
            history = json.loads(hist_file.read_text())
            print(f"skip  S={S:2d} s={s:2d} seed={seed} (done)", flush=True)
        else:
            print(f"train S={S:2d} s={s:2d} seed={seed} ({steps} steps)", flush=True)
            history = train_one_seed(
                seed, run_dir, S=S, s=s, steps=steps, eval_every=eval_every,
                eval_size=1024, save_checkpoints=False,
            )
        rows.append({
            "S": S, "s": s, "seed": seed,
            "plateau_end": plateau_end(history),
            "emergence": emergence_step(history),
            "final_acc": history[-1]["acc"],
        })
    return rows


def plot_metric(ax, rows: list, x_key: str, y_key: str, budget: int) -> None:
    xs = sorted({r[x_key] for r in rows})
    means_x, means_y = [], []
    for x in xs:
        ys = [r[y_key] for r in rows if r[x_key] == x]
        done = [y for y in ys if y is not None]
        ax.scatter([x] * len(done), done, alpha=0.6, c="tab:blue", zorder=3)
        censored = len(ys) - len(done)
        if censored:
            ax.scatter([x] * censored, [budget] * censored, marker="^",
                       c="tab:red", alpha=0.7, zorder=3)
        if done:
            means_x.append(x)
            # Censored runs are excluded, so where triangles appear the
            # true mean is HIGHER than this line.
            means_y.append(sum(done) / len(done))
    ax.plot(means_x, means_y, c="tab:blue", zorder=2)
    ax.axhline(budget, ls=":", c="gray", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel(x_key)


def make_figures(rows: list, out: Path, budget: int) -> None:
    spars = [r for r in rows if r["S"] == 16]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_metric(axes[0], spars, "s", "plateau_end", budget)
    axes[0].set_ylabel("plateau end (step, log scale)")
    axes[0].set_title("Sparsity sweep at S=16")
    plot_metric(axes[1], spars, "s", "emergence", budget)
    axes[1].set_ylabel(f"first step with acc >= {ACC_THRESH}")
    axes[1].set_title("red triangle = not reached in budget")
    fig.tight_layout()
    fig.savefig(out / "sparsity_sweep.png", dpi=150)
    plt.close(fig)

    size = [r for r in rows if r["s"] == 3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_metric(axes[0], size, "S", "plateau_end", budget)
    axes[0].set_ylabel("plateau end (step, log scale)")
    axes[0].set_title("State-size sweep at s=3")
    plot_metric(axes[1], size, "S", "emergence", budget)
    axes[1].set_ylabel(f"first step with acc >= {ACC_THRESH}")
    axes[1].set_title("red triangle = not reached in budget")
    for ax in axes:
        ax.set_xscale("log", base=2)
    fig.tight_layout()
    fig.savefig(out / "size_sweep.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--out", type=str, default="results/sweeps")
    ap.add_argument("--quick", action="store_true",
                    help="tiny grid + short runs, just to exercise the code")
    args = ap.parse_args()

    if args.quick:
        grid = [(8, 1), (8, 3)]
        args.seeds, args.steps, args.eval_every = 1, 1500, 100
    else:
        grid = [(16, s) for s in (1, 2, 3, 4, 6, 8, 12, 16)] + [(8, 3), (32, 3)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for S, s in grid:
        rows.extend(run_config(out, S, s, args.seeds, args.steps, args.eval_every))

    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    make_figures(rows, out, args.steps)

    print("\n   S   s | plateau ends per seed (None = censored)")
    for S, s in grid:
        ends = [r["plateau_end"] for r in rows if r["S"] == S and r["s"] == s]
        print(f"  {S:2d}  {s:2d} | {ends}")
    print(f"\nfigures and summary written to {out}")


if __name__ == "__main__":
    main()
