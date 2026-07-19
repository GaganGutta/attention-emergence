# attention-emergence

Reproduction and extension of **"Emergent Capabilities Arise Randomly from Learning Sparse Attention Patterns"** (Baherwani, Chen, Qiu, Wilson, Izmailov; NYU; [arXiv:2606.25010](https://arxiv.org/abs/2606.25010), June 2026).

The paper's central claim: emergent capabilities in transformers appear abruptly at unpredictable times across random seeds, and each jump coincides with the model suddenly learning a task-specific sparse attention pattern. Patching the learned attention patterns into a pre-emergence checkpoint recovers most of the capability, which makes the attention pattern itself the bottleneck.

This repo re-implements the paper's synthetic testbeds from scratch in PyTorch, reproduces the core figures, then extends the analysis with original experiments. Roadmap in [PLAN.md](PLAN.md). Every detail the paper leaves unspecified is recorded in [ASSUMPTIONS.md](ASSUMPTIONS.md).

Started 2026-07-19. No official or third-party implementation of this paper existed as of that date.

## Quickstart

```
pip install -r requirements.txt
python -m emergence.run_linear_map --seeds 3
```

Trains three single-layer transformers (identical task, different seeds) on the linear-map task from Section 3.2 and writes per-seed emergence curves and attention diagnostics to `results/linear_map/`. CPU is enough; the default run takes on the order of 15 to 30 minutes.

## Layout

```
emergence/
  linear_map.py     task: x1 = A x0 (mod 2), s-sparse rows
  model.py          minimal transformer with inspectable attention
  train.py          training loop, emergence metrics, checkpointing
  run_linear_map.py multi-seed experiment runner + figures
results/            run outputs (gitignored)
```

## Status

| Step | State |
|------|-------|
| 1. Repo scaffold | done |
| 2. Linear-map emergence experiment | done |
| 3. Sparsity / state-size sweeps | next |
| 4. Attention patching | planned |
| 5. Cellular automata task | planned |
| 6. Pythia checkpoint analysis | planned |
| 7. Writeup | planned |
| 8. Extension: emergence early-warning | planned |
