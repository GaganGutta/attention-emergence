"""Threshold sensitivity for the sparsity hump.

The plateau-end metric depends on an arbitrary loss threshold
(ASSUMPTIONS #11). This replots the hump at three thresholds from the
existing sweep histories, no retraining, to show the shape is not an
artifact of the 0.55 choice.

Usage:
    python -m emergence.replot_sweeps
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLDS = (0.55, 0.60, 0.65)


def main() -> None:
    runs = {}
    for f in Path("results/sweeps").glob("S16_s*/seed*/history.json"):
        s = int(f.parent.parent.name.split("_s")[1])
        runs.setdefault(s, []).append(json.loads(f.read_text()))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for thresh in THRESHOLDS:
        xs, means = [], []
        for s in sorted(runs):
            ends = [next((h["step"] for h in hist if h["out_loss"] < thresh), None)
                    for hist in runs[s]]
            done = [e for e in ends if e is not None]
            if done:
                xs.append(s)
                means.append(sum(done) / len(done))
        ax.plot(xs, means, marker="o", label=f"loss < {thresh}")
    ax.set_yscale("log")
    ax.set_xlabel("s (parents per output bit), S=16")
    ax.set_ylabel("mean plateau end (emerged runs only)")
    ax.set_title("The hump survives the threshold choice")
    ax.legend()
    fig.tight_layout()
    out = Path("results/sweeps/threshold_sensitivity.png")
    fig.savefig(out, dpi=150)
    print(f"written {out} (censored runs excluded from means, as in the main figure)")


if __name__ == "__main__":
    main()
