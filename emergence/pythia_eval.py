"""Step 6: emergence localization in real language models (paper Sec. 2).

EleutherAI's Pythia suite is a family of language models released WITH
mid-training snapshots (steps 0, 1, 2, .., 512 as powers of two, then
every 1000 up to 143000), which makes it security-camera footage of
training. For each model size and each capability we replay snapshots
and find when the capability switches on, using the paper's criterion:
the model's greedy (argmax) next token equals the target.

Capability suites (our concretizations; the paper does not publish its
prompts, see ASSUMPTIONS.md #14):
  copy       an 8-word list repeated; predict the final word of the copy
  induction  a word pair seen once earlier; the first word recurs,
             predict its partner
  list       a numbered list "1. .. 2. .. 3. .." ending in a newline;
             predict the next index "4"
  ioi        "When Mary and John went to the store, John gave a drink
             to" -> " Mary" (indirect object identification)

Strategy per (model, capability): evaluate a coarse checkpoint grid
(powers of two, then sparse thousands), then binary-search the 1000-step
grid between the last failing and first passing coarse points. Every
(model, step) suite accuracy is cached to disk immediately, so the job
can be interrupted and resumed without re-downloading anything.

Usage:
    python -m emergence.pythia_eval --models pythia-14m pythia-70m pythia-160m
    python -m emergence.pythia_eval --smoke   # one model, final step only
"""

import argparse
import json
import random
from pathlib import Path

import torch

WORDS = ["apple", "river", "stone", "cloud", "tiger", "piano", "green",
         "house", "silver", "candle", "forest", "window", "bottle", "garden",
         "marble", "pepper", "rocket", "saddle", "temple", "velvet"]
NAMES = ["Mary", "John", "Sarah", "James", "Anna", "Peter", "Laura", "David",
         "Emma", "Michael", "Alice", "Robert"]

COARSE = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
          1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
PASS_THRESH = 0.75
N_PROMPTS = 16
PURGE_CACHE = True  # delete each HF snapshot after scoring it


def purge_revision(name: str, step: int) -> None:
    """Free the disk space of one scored checkpoint. Scores live in
    cache.json, so a purged revision never needs to come back."""
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == f"EleutherAI/{name}":
                for rev in repo.revisions:
                    if f"step{step}" in rev.refs:
                        scan_cache_dir().delete_revisions(rev.commit_hash).execute()
                        return
    except Exception as e:
        print(f"  cache purge skipped: {e}", flush=True)


def build_suites(model, rng: random.Random) -> dict:
    """Prompt/target pairs whose targets are single tokens for this
    model's tokenizer (multi-token candidates are filtered out)."""
    def tok(s):
        try:
            return model.to_single_token(s)
        except Exception:
            return None

    words = [w for w in WORDS if tok(" " + w) is not None]
    names = [n for n in NAMES if tok(" " + n) is not None]
    suites = {"copy": [], "induction": [], "list": [], "ioi": []}

    while len(suites["copy"]) < N_PROMPTS:
        seq = rng.sample(words, 8)
        prompt = " ".join(seq) + " " + " ".join(seq[:-1])
        suites["copy"].append((prompt, " " + seq[-1]))

    while len(suites["induction"]) < N_PROMPTS:
        a, b = rng.sample(words, 2)
        prompt = (f"I packed the {a} {b} together with everything else. "
                  f"Later I found the {a}")
        suites["induction"].append((prompt, " " + b))

    while len(suites["list"]) < N_PROMPTS:
        w = rng.sample(words, 3)
        prompt = f"1. {w[0]}\n2. {w[1]}\n3. {w[2]}\n"
        suites["list"].append((prompt, "4"))

    while len(suites["ioi"]) < N_PROMPTS:
        a, b = rng.sample(names, 2)
        prompt = (f"When {a} and {b} went to the store, {b} gave a drink to")
        suites["ioi"].append((prompt, " " + a))
    return suites


@torch.no_grad()
def suite_accuracy(model, suite: list) -> float:
    correct = 0
    for prompt, target in suite:
        logits = model(model.to_tokens(prompt))
        pred = logits[0, -1].argmax().item()
        correct += int(pred == model.to_single_token(target))
    return correct / len(suite)


def eval_checkpoint(name: str, step: int, cache: dict, cache_file: Path,
                    rng_seed: int = 0) -> dict:
    key = str(step)
    if key in cache.get(name, {}):
        return cache[name][key]
    from transformer_lens import HookedTransformer
    print(f"loading {name} step {step}", flush=True)
    model = HookedTransformer.from_pretrained(name, checkpoint_value=step)
    model.eval()
    suites = build_suites(model, random.Random(rng_seed))
    accs = {cap: suite_accuracy(model, suite) for cap, suite in suites.items()}
    del model
    if PURGE_CACHE:
        purge_revision(name, step)
    cache.setdefault(name, {})[key] = accs
    cache_file.write_text(json.dumps(cache, indent=2))
    print(f"  {name}@{step}: " +
          "  ".join(f"{c}={a:.2f}" for c, a in accs.items()), flush=True)
    return accs


def localize(name: str, cap: str, cache: dict, cache_file: Path) -> int:
    """Emergence step: first checkpoint with suite accuracy >= PASS_THRESH,
    refined by binary search on the 1000-step grid. None if never."""
    accs = {s: eval_checkpoint(name, s, cache, cache_file)[cap] for s in COARSE}
    passing = [s for s in COARSE if accs[s] >= PASS_THRESH]
    if not passing:
        return None
    first_pass = min(passing)
    prior = [s for s in COARSE if s < first_pass]
    lo = max(prior) if prior else 0
    if first_pass - lo <= 1000:
        return first_pass
    lo, hi = lo // 1000, first_pass // 1000   # search multiples of 1000
    while hi - lo > 1:
        mid = (lo + hi) // 2
        acc = eval_checkpoint(name, mid * 1000, cache, cache_file)[cap]
        if acc >= PASS_THRESH:
            hi = mid
        else:
            lo = mid
    return hi * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["pythia-14m", "pythia-70m", "pythia-160m"])
    ap.add_argument("--out", type=str, default="results/pythia")
    ap.add_argument("--smoke", action="store_true",
                    help="one small model, final checkpoint only")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep downloaded snapshots on disk after scoring")
    args = ap.parse_args()
    global PURGE_CACHE
    PURGE_CACHE = not args.keep_cache

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_file = out / "cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    if args.smoke:
        eval_checkpoint("pythia-14m", 143000, cache, cache_file)
        return

    emergence = {}
    for name in args.models:
        emergence[name] = {}
        for cap in ["copy", "induction", "list", "ioi"]:
            try:
                step = localize(name, cap, cache, cache_file)
            except Exception as e:  # model name unavailable, download error
                print(f"SKIP {name}/{cap}: {e}", flush=True)
                step = "error"
            emergence[name][cap] = step
            print(f"== {name} {cap}: emergence at {step}", flush=True)
        (out / "emergence.json").write_text(json.dumps(emergence, indent=2))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
