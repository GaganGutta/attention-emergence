"""Step 8: can you see emergence coming? (original work)

The reproduction showed that emergence timing is seed-random when you
watch the loss, and that the capability crystallizes as one co-adapted
circuit. The question here: is the timing forecastable from the model's
EARLY internal statistics, before anything visible happens to the loss?

Setup: every fleet run (fixed task, S=12 s=3, evals every 25 steps) gets
a feature vector computed ONLY from evaluations in the window
[0, --window] steps, and a label: the step at which it actually emerged
(first eval with bit accuracy >= 0.99). Runs that emerged inside the
window are excluded (their features would contain the answer); runs that
never emerged are excluded from regression and counted separately.

Features (per run): best-head parent mass at window end and its slope;
min-head attention entropy at window end and its slope; output loss at
window end, slope, and variance; across-head spread of parent mass.
Labels are regressed in log space (timings are heavy-tailed).

Models, deliberately simple so the result is about the signal:
  baseline   predict the (geometric) mean emergence step of other runs
  linear     standardized linear regression on the features
Scored by leave-one-out mean absolute error in steps, plus correlations.

Usage:
    python -m emergence.predict --window 250
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_NAMES = ["pm_end", "pm_slope", "ent_end", "ent_slope",
                 "loss_end", "loss_slope", "loss_var", "spread_end"]


def emergence_step(history: list):
    return next((h["step"] for h in history if h["acc"] >= 0.99), None)


def features(history: list, window: int):
    evals = [h for h in history if h["step"] <= window]
    if len(evals) < 3:
        return None
    steps = np.array([h["step"] for h in evals], dtype=float)
    pm = np.array([max(h["parent_mass"]) for h in evals])
    ent = np.array([min(h["head_entropy"]) for h in evals])
    loss = np.array([h["out_loss"] for h in evals])
    spread = np.array([np.std(h["parent_mass"]) for h in evals])

    def slope(y):  # per 1000 steps
        return float(np.polyfit(steps, y, 1)[0] * 1000)

    return np.array([pm[-1], slope(pm), ent[-1], slope(ent),
                     loss[-1], slope(loss), float(np.var(loss)), spread[-1]])


def loo_predictions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out linear regression with per-fold standardization."""
    preds = np.zeros(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-9
        A = np.hstack([(X[m] - mu) / sd, np.ones((m.sum(), 1))])
        w, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        preds[i] = np.append((X[i] - mu) / sd, 1.0) @ w
    return preds


def loo_single_feature(X: np.ndarray, y: np.ndarray):
    """One-feature linear model; the feature is chosen by |correlation|
    on each training fold, so selection never sees the held-out run."""
    preds, picks = np.zeros(len(y)), []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        corrs = [abs(np.corrcoef(X[m, j], y[m])[0, 1]) for j in range(X.shape[1])]
        j = int(np.argmax(corrs))
        picks.append(FEATURE_NAMES[j])
        a, b = np.polyfit(X[m, j], y[m], 1)
        preds[i] = a * X[i, j] + b
    top = max(set(picks), key=picks.count)
    return preds, top, picks.count(top)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def load_rows(fleet_dir: str, window: int):
    rows, never, early = [], 0, 0
    for hist_file in sorted(Path(fleet_dir).glob("seed*/history.json")):
        try:
            history = json.loads(hist_file.read_text())
        except Exception:
            continue  # file mid-write by a running fleet
        label = emergence_step(history)
        if label is None:
            never += 1
            continue
        if label <= window:
            early += 1
            continue
        f = features(history, window)
        if f is not None:
            rows.append((int(hist_file.parent.name[4:]), f, label))
    return rows, never, early


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", type=str, default="results/fleet")
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--transfer", type=str, default=None,
                    help="second fleet dir (different task setting); the "
                         "model is fit on --fleet and evaluated there")
    ap.add_argument("--out", type=str, default="results/predictor")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, never, early = load_rows(args.fleet, args.window)

    if len(rows) < 8:
        print(f"only {len(rows)} usable runs; need more fleet data")
        return

    X = np.stack([f for _, f, _ in rows])
    y_raw = np.array([lab for _, _, lab in rows], dtype=float)
    y = np.log(y_raw)
    print(f"usable runs: {len(rows)}  (never emerged: {never}, "
          f"emerged inside window: {early})")
    print(f"label range: {int(y_raw.min())} to {int(y_raw.max())} steps")

    # Baseline: geometric mean of the other runs (leave-one-out).
    base_pred = np.array([np.exp(np.delete(y, i).mean()) for i in range(len(y))])
    base_mae = float(np.abs(base_pred - y_raw).mean())

    pred = np.exp(loo_predictions(X, y))
    mae = float(np.abs(pred - y_raw).mean())
    r_log = float(np.corrcoef(np.log(pred), y)[0, 1])
    rho = spearman(pred, y_raw)

    pred1, top_feat, top_count = loo_single_feature(X, y)
    pred1 = np.exp(pred1)
    mae1 = float(np.abs(pred1 - y_raw).mean())
    r1 = float(np.corrcoef(np.log(pred1), y)[0, 1])
    rho1 = spearman(pred1, y_raw)

    per_feature = {name: float(np.corrcoef(X[:, j], y)[0, 1])
                   for j, name in enumerate(FEATURE_NAMES)}

    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.hstack([(X - mu) / sd, np.ones((len(y), 1))])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    weights = {name: float(w[j]) for j, name in enumerate(FEATURE_NAMES)}

    print(f"\nbaseline (geometric mean) LOO MAE: {base_mae:7.1f} steps")
    print(f"full linear model         LOO MAE: {mae:7.1f} steps "
          f"({100 * (1 - mae / base_mae):+.1f}%, r {r_log:.2f}, rho {rho:.2f})")
    print(f"single-feature model      LOO MAE: {mae1:7.1f} steps "
          f"({100 * (1 - mae1 / base_mae):+.1f}%, r {r1:.2f}, rho {rho1:.2f}; "
          f"picked {top_feat} in {top_count}/{len(y)} folds)")
    print("\nfeature -> correlation with log(emergence step):")
    for name, r in sorted(per_feature.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<11} {r:+.2f}   (weight {weights[name]:+.2f})")

    report = {"window": args.window, "n_usable": len(rows), "n_never": never,
              "n_early": early, "label_min": int(y_raw.min()),
              "label_max": int(y_raw.max()), "baseline_mae": base_mae,
              "model_mae": mae, "pearson_log": r_log, "spearman": rho,
              "single_feature": {"mae": mae1, "pearson_log": r1, "spearman": rho1,
                                 "feature": top_feat, "picked_in_folds": top_count},
              "feature_correlations": per_feature, "weights": weights}
    if args.transfer:
        t_rows, t_never, t_early = load_rows(args.transfer, args.window)
        Xt = np.stack([f for _, f, _ in t_rows])
        yt_raw = np.array([lab for _, _, lab in t_rows], dtype=float)
        yt = np.log(yt_raw)
        # Fit the single-feature model on the ENTIRE source fleet, then
        # apply it untouched to the transfer domain.
        corrs = [abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])]
        j = int(np.argmax(corrs))
        a, b = np.polyfit(X[:, j], y, 1)
        pred_t = a * Xt[:, j] + b
        r_t = float(np.corrcoef(pred_t, yt)[0, 1])
        rho_t = spearman(pred_t, yt_raw)
        mae_raw = float(np.abs(np.exp(pred_t) - yt_raw).mean())
        # Intercept-only recalibration: the harder task shifts the overall
        # timescale; refitting b (never a) shows whether the FEATURE
        # transfers even when the level does not.
        b_shift = float((yt - a * Xt[:, j]).mean())
        mae_shift = float(np.abs(np.exp(a * Xt[:, j] + b_shift) - yt_raw).mean())
        base_mae_t = float(np.abs(np.exp(y.mean()) - yt_raw).mean())
        print(f"\nTRANSFER to {args.transfer} "
              f"({len(t_rows)} runs, never {t_never}, early {t_early}; "
              f"labels {int(yt_raw.min())}-{int(yt_raw.max())}):")
        print(f"  feature {FEATURE_NAMES[j]}: pearson r (log) {r_t:.2f}, "
              f"spearman {rho_t:.2f}")
        print(f"  MAE raw {mae_raw:.0f} | intercept-recalibrated {mae_shift:.0f} "
              f"| source-mean baseline {base_mae_t:.0f} steps")
        report["transfer"] = {
            "fleet": args.transfer, "n": len(t_rows),
            "label_min": int(yt_raw.min()), "label_max": int(yt_raw.max()),
            "feature": FEATURE_NAMES[j], "pearson_log": r_t, "spearman": rho_t,
            "mae_raw": mae_raw, "mae_recalibrated": mae_shift,
            "source_mean_baseline_mae": base_mae_t}

    (out / "report.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    lo, hi = y_raw.min() * 0.8, y_raw.max() * 1.25
    axes[0].scatter(y_raw, pred, alpha=0.5, label="full linear")
    axes[0].scatter(y_raw, pred1, alpha=0.8, marker="s", label=f"1-feature ({top_feat})")
    axes[0].legend()
    axes[0].plot([lo, hi], [lo, hi], ls="--", c="gray", lw=1)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("actual emergence step")
    axes[0].set_ylabel("predicted from first "
                       f"{args.window} steps")
    axes[0].set_title(f"LOO predictions (MAE {mae:.0f} vs baseline {base_mae:.0f})")
    names = sorted(per_feature, key=lambda n: -abs(per_feature[n]))
    axes[1].barh(range(len(names)), [abs(per_feature[n]) for n in names])
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("|correlation| with log emergence step")
    axes[1].set_title("Which early signals carry the forecast")
    fig.tight_layout()
    fig.savefig(out / "early_warning.png", dpi=150)
    plt.close(fig)
    print(f"\nreport and figure written to {out}")


if __name__ == "__main__":
    main()
