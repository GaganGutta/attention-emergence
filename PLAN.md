# Plan

Target: reproduce arXiv 2606.25010 from scratch in PyTorch, then extend it. One step at a time, at least one commit per step, assumptions logged as they are made.

1. **Repo scaffold.** README, plan, assumptions, requirements. *(done 2026-07-19)*
2. **Linear-map task (paper Sec. 3.2).** Data generation (x1 = A x0 mod 2, s-sparse rows), single-layer transformer (D=128, 8 heads, MLP 512), multi-seed training with emergence tracking: per-step loss on output tokens, p(correct), exact-match rate, per-head attention entropy, attention mass on true parent positions. Reproduces the paper's core qualitative result: abrupt emergence at seed-dependent times, coinciding with attention-entropy collapse. *(done 2026-07-19)*
3. **Linear-map analysis.** Sparsity sweep s in {1..S} and state-size sweep S in {8,16,32}; reproduce the finding that medium sparsity is hardest and that plateau length grows with state size (loss-threshold crossing as the plateau metric). Attention-pattern heatmap visualizations pre/post emergence.
4. **Attention patching.** Transplant post-emergence attention patterns into pre-emergence checkpoints and measure how much capability is recovered; the paper's causal experiment, using the checkpoints saved in step 2.
5. **Cellular automata task (paper Sec. 3.3).** C=4 colors, lookup-table rules on local windows, N=256 rules, T=16 states, 4-layer model.
6. **Pythia checkpoint analysis (paper Sec. 2).** TransformerLens over public Pythia checkpoints (14M to 160M on CPU, 410M on Colab if needed): copying, in-context repetition (induction), pattern completion, IOI; emergence-point localization by binary search over checkpoints; compare across the 10 public init-seed sets where available.
7. **Writeup.** Results-focused README with reproduced figures side by side with the paper's, plus discrepancy notes and assumption sensitivity.
8. **Extension (original work).** Early-warning prediction of emergence: do pre-emergence attention statistics (entropy slope, parent-mass drift, head specialization) predict WHEN the jump happens? The paper's patching result implies the signal exists; nobody has built the predictor. Fallbacks: new task family (modular arithmetic, Dyck), cross-seed attention patching.

## Compute

Steps 2 to 5 run on a laptop CPU. Step 6 is inference-only; small Pythia models are CPU-viable, 410M wants a free Colab GPU.

## Provenance risk

The paper is a June 23, 2026 preprint with no code. If the authors release code later (for example at a camera-ready deadline), this repo's dated commit history and ASSUMPTIONS.md are the record that the reimplementation was independent. The extension in step 8 is the headline either way.
