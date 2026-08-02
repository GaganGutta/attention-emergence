# attention-emergence

Reproduction and extension of **"Emergent Capabilities Arise Randomly from Learning Sparse Attention Patterns"** (Baherwani, Chen, Qiu, Wilson, Izmailov; NYU; [arXiv:2606.25010](https://arxiv.org/abs/2606.25010), June 2026).

The paper's central claim: emergent capabilities in transformers appear abruptly at unpredictable times across random seeds, and each jump coincides with the model suddenly learning a task-specific sparse attention pattern. This repo re-implements the paper's synthetic testbeds from scratch in PyTorch, reproduces its core results on a laptop CPU, and extends the analysis. Roadmap in [PLAN.md](PLAN.md); every detail the paper leaves unspecified is recorded in [ASSUMPTIONS.md](ASSUMPTIONS.md); environment fixes in [docs/SETUP.md](docs/SETUP.md).

Started 2026-07-22. No official or third-party implementation of this paper existed as of that date (re-verified 2026-08-01).

## Quickstart

```
pip install -r requirements.txt
python -m emergence.run_linear_map --seeds 3     # core emergence experiment
python -m emergence.run_sweeps --seeds 4         # difficulty sweeps (overnight)
python -m emergence.patch                        # attention transplant
python -m emergence.run_cellular --seeds 3       # cellular automata task
python -m emergence.pythia_eval                  # real-LM emergence (downloads)
```

## Results so far

### Emergence is abrupt and seed-random (paper Sec. 3.2)

![emergence curves](docs/figures/emergence_curves.png)

Three identical single-layer transformers on the identical task, differing only in random seed. One snaps to 100% at step ~3,050, one emerges in two stages (15 of 16 output bits early, the last near step 7,000), one grinds and is still unfinished at 10,000 steps. Attention entropy collapses and attention mass on the true parent bits rises at each transition ([diagnostics](docs/figures/attention_diagnostics.png)).

### Difficulty laws (paper Sec. 3.3)

![sparsity sweep](docs/figures/sparsity_sweep.png)

Plateau length versus sparsity is a hump spanning three orders of magnitude: trivial at s=1-2, censored at s=6-8 (most seeds never escape a 20,000-step budget), easy again at s=16 where the near-uniform newborn attention is already correct. Versus state size, plateaus grow multiplicatively, roughly 100 / 1,900 / 9,000 steps for S=8/16/32 ([figure](docs/figures/size_sweep.png)). Second-order observation: plateau escape and task mastery decouple at medium-high sparsity (s=12 escapes the loss floor but never reaches 90% accuracy in budget; s=16 reaches 90% within 100 steps but converges slowly).

### The attention pattern is necessary but not sufficient (step 4)

![patching](docs/figures/patching.png)

Forcing post-emergence attention patterns into a late-plateau checkpoint recovers nothing (accuracy stays ~0.50), while the reverse control destroys the trained model (~1.00 to ~0.50). Parameter-level component swaps show no proper subset of trained components restores the capability; only the near-complete network does (0.90 without embeddings, 0.985 with). In this minimal model the capability is a single co-adapted circuit. The paper's patching-recovery narrative comes from large language models, where downstream circuitry pre-exists; our Pythia results below are consistent with that reading, and the contrast is a boundary condition the paper does not state.

### Real language models (paper Sec. 2)

Emergence step per capability across Pythia sizes (suite accuracy >= 0.75, binary-searched over public training snapshots; null = never within 143k steps):

| Capability | 14M | 70M | 160M |
|---|---|---|---|
| Induction | 5,000 | 12,000 | 1,000 |
| Numbered lists | 19,000 | 2,000 | 2,000 |
| Copying | never | never | 8,000 |
| Indirect object identification | never | never | 16,000 |

Copying and IOI are scale-gated: absent at 14M and 70M, switching on within 160M's training run. Induction exists at every scale with seed-scattered timing. This matches the paper's claims that larger models acquire capabilities earlier on average while individual timing is stochastic.

## Layout

```
emergence/
  linear_map.py     task: x1 = A x0 (mod 2), s-sparse rows
  model.py          minimal transformer with inspectable, overridable attention
  train.py          training loop, emergence metrics, checkpointing, early stop
  run_linear_map.py multi-seed emergence experiment + figures
  run_sweeps.py     difficulty sweeps with censoring-aware plateau stats
  patch.py          attention transplant + component-swap surgery
  cellular.py       cellular automata task (in-context rule inference)
  run_cellular.py   CA experiment runner
  pythia_eval.py    emergence localization in Pythia checkpoints
docs/               setup notes and promoted figures
results/            run outputs (gitignored)
```

## Status

| Step | State |
|------|-------|
| 1. Repo scaffold | done |
| 2. Linear-map emergence experiment | done |
| 3. Sparsity / state-size sweeps | done; both scaling claims reproduced |
| 4. Attention patching | done; necessary-but-not-sufficient finding |
| 5. Cellular automata task | running |
| 6. Pythia checkpoint analysis | done |
| 7. Writeup | in progress |
| 8. Extension: emergence early-warning | next |
