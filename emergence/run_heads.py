"""Head-count scaling (paper Sec. 4.2, the part that is cheap on CPU).

Same single-layer model and task, varying only the number of attention
heads at fixed total width (D=128 split across 1 to 16 heads). The paper
reports head count matters for how quickly the sparse pattern is found.

Usage:
    python -m emergence.run_heads --seeds 3
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .train import train_one_seed

HEAD_COUNTS = (1, 2, 4, 8, 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--s", type=int, default=3)
    ap.add_argument("--steps", type=int, default=12_000)
    ap.add_argument("--out", type=str, default="results/heads")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for n_heads in HEAD_COUNTS:
        results[n_heads] = []
        for seed in range(args.seeds):
            run_dir = out / f"h{n_heads}" / f"seed{seed}"
            hist_file = run_dir / "history.json"
            if hist_file.exists():
                history = json.loads(hist_file.read_text())
                print(f"skip h={n_heads} seed={seed} (done)", flush=True)
            else:
                print(f"train h={n_heads} seed={seed}", flush=True)
                history = train_one_seed(
                    seed, run_dir, S=args.S, s=args.s, steps=args.steps,
                    eval_every=100, eval_size=1024, save_checkpoints=False,
                    early_stop_evals=3, n_heads=n_heads)
            results[n_heads].append(next(
                (h["step"] for h in history if h["acc"] >= 0.9), None))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n_heads, ends in results.items():
        done = [e for e in ends if e is not None]
        censored = len(ends) - len(done)
        ax.scatter([n_heads] * len(done), done, c="tab:blue", alpha=0.7)
        if censored:
            ax.scatter([n_heads] * censored, [args.steps] * censored,
                       marker="^", c="tab:red", alpha=0.7)
    means = {h: sum(e for e in v if e) / max(1, len([e for e in v if e]))
             for h, v in results.items() if any(v)}
    ax.plot(list(means), list(means.values()), c="tab:blue")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("attention heads (D=128 total)")
    ax.set_ylabel("first step with acc >= 0.9")
    ax.set_title(f"Head-count scaling at S={args.S}, s={args.s}")
    fig.tight_layout()
    fig.savefig(out / "head_scaling.png", dpi=150)

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    for n_heads, ends in results.items():
        print(f"h={n_heads:2d}: emergence steps {ends}")


if __name__ == "__main__":
    main()
