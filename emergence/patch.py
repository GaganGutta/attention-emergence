"""Step 4: the attention transplant (paper Sec. 2.2 and App. A.1, the causal experiment).

The emergence curves show that capability jumps WHEN attention sharpens.
That is correlation. This experiment tests causation: run a pre-emergence
checkpoint, but force every head to use the attention pattern that a
post-emergence checkpoint produces on the same inputs. Everything else
about the pre-model (its MLP, embeddings, output layer) stays frozen.

If the skill largely appears, the where-to-look pattern was the missing
piece. The reverse control (post-model forced to use pre-attention)
should destroy the skill, ruling out "any attention works".

Checkpoint choice: "pre" is the latest saved checkpoint whose measured
accuracy was still near chance (< 0.55), i.e. late in the plateau, so
its non-attention machinery is as trained as possible without the skill.
"post" is the final checkpoint.

Per-head transplant: patch one head at a time (the other 7 keep the
pre-model's own patterns) to see whether the skill lives in a few heads
or is spread across all of them.

Usage:
    python -m emergence.patch            # uses results/linear_map/seed*
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .linear_map import sample_batch, sample_transition
from .model import TinyTransformer


@torch.no_grad()
def accuracy(model, tokens, S, override=None) -> float:
    logits = model(tokens, patterns_override=override)
    pred = logits[:, S - 1 : 2 * S - 1].argmax(-1)
    return (pred == tokens[:, S : 2 * S]).float().mean().item()


@torch.no_grad()
def get_patterns(model, tokens) -> list:
    _, patterns = model(tokens, return_attention=True)
    return patterns


def model_at(run_dir: Path, step: int, config: dict) -> TinyTransformer:
    arch = config.get("arch", {})
    model = TinyTransformer(max_len=2 * config["S"],
                            d_model=arch.get("d_model", 128),
                            n_heads=arch.get("n_heads", 8),
                            n_layers=arch.get("n_layers", 1),
                            d_mlp=arch.get("d_mlp", 512))
    model.load_state_dict(torch.load(run_dir / "ckpt" / f"step{step}.pt",
                                     weights_only=True))
    model.eval()
    return model


def rebuild_eval_batch(run_dir: Path, config: dict) -> torch.Tensor:
    saved = run_dir / "eval_tokens.pt"
    if saved.exists():
        return torch.load(saved, weights_only=True)
    # Older runs predate eval_tokens.pt: replay the task generator in the
    # same order as train_one_seed (A first, then the eval batch).
    gen = torch.Generator().manual_seed(config["task_seed"])
    A = sample_transition(config["S"], config["s"], gen)
    return sample_batch(A, config["eval_size"], gen)


def pick_checkpoints(run_dir: Path, history: list):
    """pre = last checkpoint strictly before accuracy FIRST crosses 0.55,
    so a post-emergence dip can never be mistaken for the plateau."""
    ckpt_steps = sorted(int(p.stem[4:]) for p in (run_dir / "ckpt").glob("step*.pt"))
    first_cross = next((h["step"] for h in history if h["acc"] >= 0.55), None)
    if first_cross is None:
        return ckpt_steps[-2], ckpt_steps[-1]
    pre_candidates = [s for s in ckpt_steps if s < first_cross]
    pre = max(pre_candidates) if pre_candidates else ckpt_steps[0]
    return pre, ckpt_steps[-1]


def restricted_override(own: list, donor: list, S: int) -> list:
    """Donor attention only where the skill lives: output-predicting
    queries (S-1..2S-2) restricted to input keys (0..S-1), renormalized;
    everything else keeps the pre-model's own pattern. In a 1-layer model
    the input-half query rows cannot influence output logits at all, so
    this mainly guards the experiment's intent (and any deeper model)."""
    mixed = own[0].clone()
    q = slice(S - 1, 2 * S - 1)
    mixed[:, :, q, :] = 0.0
    mixed[:, :, q, :S] = donor[0][:, :, q, :S]
    mixed[:, :, q, :] /= mixed[:, :, q, :].sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return [mixed]


def run_seed(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    history = json.loads((run_dir / "history.json").read_text())
    S = config["S"]
    tokens = rebuild_eval_batch(run_dir, config)
    pre_step, post_step = pick_checkpoints(run_dir, history)
    pre_m = model_at(run_dir, pre_step, config)
    post_m = model_at(run_dir, post_step, config)

    donor = get_patterns(post_m, tokens)   # post-emergence attention
    own = get_patterns(pre_m, tokens)      # pre-emergence attention

    row = {
        "seed": config["seed"], "pre_step": pre_step, "post_step": post_step,
        "acc_pre": accuracy(pre_m, tokens, S),
        "acc_post": accuracy(post_m, tokens, S),
        "acc_pre_patched": accuracy(pre_m, tokens, S, override=donor),
        "acc_pre_patched_restricted": accuracy(
            pre_m, tokens, S, override=restricted_override(own, donor, S)),
        "acc_post_reversed": accuracy(post_m, tokens, S, override=own),
    }
    # One head at a time: donor pattern for head h, own patterns elsewhere.
    gains = []
    for h in range(own[0].shape[1]):
        mixed = [own[0].clone()]
        mixed[0][:, h] = donor[0][:, h]
        gains.append(accuracy(pre_m, tokens, S, override=mixed) - row["acc_pre"])
    row["per_head_gain"] = gains
    row["component_swaps"] = component_swaps(pre_m, post_m, tokens, S)
    return row


PARTS = ("qk", "vproj", "mlp", "unembed", "emb")
COMBOS = [("qk",), ("vproj",), ("mlp",), ("unembed",), ("emb",),
          ("qk", "mlp"), ("qk", "vproj"), ("qk", "unembed"),
          ("qk", "mlp", "unembed"), ("qk", "vproj", "mlp", "unembed"),
          PARTS]


def hybrid_state(pre_sd: dict, post_sd: dict, parts: tuple) -> dict:
    """Pre-model parameters with the named components replaced by the
    post-model's. Component map (single-layer models only):
      qk      query/key rows of the fused qkv matrix + pre-attention norm
      vproj   value rows of qkv + the attention output projection
      mlp     the feedforward block + its norm
      unembed final norm + readout to logits
      emb     token and position embeddings
    """
    sd = {k: v.clone() for k, v in pre_sd.items()}
    D = sd["blocks.0.qkv.weight"].shape[1]

    def take(*keys):
        for k in keys:
            sd[k] = post_sd[k].clone()

    if "qk" in parts:
        sd["blocks.0.qkv.weight"][: 2 * D] = post_sd["blocks.0.qkv.weight"][: 2 * D]
        sd["blocks.0.qkv.bias"][: 2 * D] = post_sd["blocks.0.qkv.bias"][: 2 * D]
        take("blocks.0.ln1.weight", "blocks.0.ln1.bias")
    if "vproj" in parts:
        sd["blocks.0.qkv.weight"][2 * D :] = post_sd["blocks.0.qkv.weight"][2 * D :]
        sd["blocks.0.qkv.bias"][2 * D :] = post_sd["blocks.0.qkv.bias"][2 * D :]
        take("blocks.0.proj.weight", "blocks.0.proj.bias")
    if "mlp" in parts:
        take("blocks.0.mlp.0.weight", "blocks.0.mlp.0.bias",
             "blocks.0.mlp.2.weight", "blocks.0.mlp.2.bias",
             "blocks.0.ln2.weight", "blocks.0.ln2.bias")
    if "unembed" in parts:
        take("ln_f.weight", "ln_f.bias", "unembed.weight")
    if "emb" in parts:
        take("tok_emb.weight", "pos_emb.weight")
    return sd


@torch.no_grad()
def component_swaps(pre_m, post_m, tokens, S) -> dict:
    pre_sd, post_sd = pre_m.state_dict(), post_m.state_dict()
    results = {}
    probe = TinyTransformer(max_len=2 * S)
    for combo in COMBOS:
        probe.load_state_dict(hybrid_state(pre_sd, post_sd, combo))
        probe.eval()
        results["+".join(combo)] = accuracy(probe, tokens, S)
    return results


def make_figure(rows: list, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    labels = ["pre", "pre+post attn", "post+pre attn", "post"]
    keys = ["acc_pre", "acc_pre_patched", "acc_post_reversed", "acc_post"]
    width = 0.8 / len(rows)
    for i, r in enumerate(rows):
        xs = [j + i * width for j in range(len(keys))]
        axes[0].bar(xs, [r[k] for k in keys], width, label=f"seed {r['seed']}")
    axes[0].set_xticks([j + width for j in range(len(keys))])
    axes[0].set_xticklabels(labels)
    axes[0].axhline(0.5, ls="--", c="gray", lw=1)
    axes[0].set_ylabel("bit accuracy")
    axes[0].set_title("Transplanting attention patterns")
    axes[0].legend()
    for r in rows:
        axes[1].plot(range(len(r["per_head_gain"])), r["per_head_gain"],
                     marker="o", label=f"seed {r['seed']}")
    axes[1].axhline(0, ls="--", c="gray", lw=1)
    axes[1].set_xlabel("head index")
    axes[1].set_ylabel("accuracy gain from patching this head alone")
    axes[1].set_title("Where does the skill live?")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "patching.png", dpi=150)
    plt.close(fig)


def main() -> None:
    out = Path("results/patching")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(Path("results/linear_map").glob("seed*")):
        row = run_seed(run_dir)
        rows.append(row)
        print(f"seed {row['seed']}: pre(step {row['pre_step']}) {row['acc_pre']:.3f}"
              f" -> patched {row['acc_pre_patched']:.3f}"
              f" (restricted {row['acc_pre_patched_restricted']:.3f})"
              f" | post(step {row['post_step']}) {row['acc_post']:.3f}"
              f" -> reversed {row['acc_post_reversed']:.3f}", flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    make_figure(rows, out)
    combos = list(rows[0]["component_swaps"])
    print("\ncomponents swapped in -> mean accuracy across seeds")
    for c in combos:
        mean = sum(r["component_swaps"][c] for r in rows) / len(rows)
        print(f"  {c:<24} {mean:.3f}")
    print(f"figure and summary written to {out}")


if __name__ == "__main__":
    main()
