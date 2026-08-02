"""Step 8 data fleet: many cheap emergence runs for the early-warning study.

One fixed task (S=12, s=3), many seeds, dense evaluation logging (every
25 steps) so the pre-emergence window is finely sampled, no checkpoints.
Each run writes results/fleet/seed*/history.json; finished seeds are
skipped on relaunch. The predictor analysis consumes these histories.

Usage:
    python -m emergence.run_fleet --seeds 40
"""

import argparse
import json
from pathlib import Path

from .train import train_one_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--S", type=int, default=12)
    ap.add_argument("--s", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8_000)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--out", type=str, default="results/fleet")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    emergence_steps = {}
    for seed in range(args.seeds):
        run_dir = out / f"seed{seed}"
        hist_file = run_dir / "history.json"
        if hist_file.exists():
            history = json.loads(hist_file.read_text())
            print(f"skip seed {seed} (done)", flush=True)
        else:
            print(f"train seed {seed} (S={args.S}, s={args.s})", flush=True)
            history = train_one_seed(
                seed, run_dir, S=args.S, s=args.s, steps=args.steps,
                eval_every=args.eval_every, eval_size=512,
                save_checkpoints=False, early_stop_evals=5,
            )
        emergence_steps[seed] = next(
            (h["step"] for h in history if h["acc"] >= 0.99), None)

    (out / "emergence_steps.json").write_text(json.dumps(emergence_steps, indent=2))
    done = [s for s in emergence_steps.values() if s is not None]
    print(f"\n{len(done)}/{len(emergence_steps)} runs emerged; "
          f"range {min(done) if done else '-'} to {max(done) if done else '-'}")
    print(f"histories in {out}")


if __name__ == "__main__":
    main()
