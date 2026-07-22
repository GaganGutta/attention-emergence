# Assumptions

Details the paper does not specify, and the choices made here. Each one is a candidate explanation if reproduced numbers drift from the paper's.

| # | Unstated detail | Choice here | Notes |
|---|-----------------|-------------|-------|
| 1 | Optimizer | AdamW, lr 1e-3, betas default, weight decay 0.01 | Not stated anywhere in the paper. Emergence TIMING is likely sensitive to lr; if nothing emerges within 10k steps, calibrate lr in {3e-4, 1e-3, 3e-3} before touching anything else. |
| 2 | Batch size | 256 sequences | Paper says only "fixed token batch size". |
| 3 | Positional encoding | Learned absolute embeddings | Not stated. |
| 4 | Block structure | Pre-LN, GELU MLP, no dropout, PyTorch default init | Not stated. Online data regime, so no regularization pressure. |
| 5 | Loss positions | Cross-entropy on the S output predictions only (logit positions S-1..2S-2 predicting tokens S..2S-1) | The input half is uniform random bits and carries no learnable signal. Paper decomposes loss "on each individual output token", consistent with this. Full-sequence loss would just add a constant ~0.5*ln 2 per position floor. |
| 6 | Emergence metric aggregation | Batch-mean p(correct) over 2048 fixed eval sequences, plus a single fixed probe example (paper uses single-sample methodology, App. A.2), plus exact-match rate | Eval set fixed across seeds and steps. |
| 7 | Attention entropy indexing | Paper sums over query positions S..2S-1, key positions 0..S-1. Here: queries S-1..2S-2, i.e. the positions whose logits predict the output tokens (one-position shift for the next-token offset). Softmax scores are batch-averaged first, entropy taken over the raw (unrenormalized) block, per head. | If figures disagree with the paper, revisit this off-by-one reading first. |
| 8 | Task fixed across seeds | Transition matrix A drawn once with task_seed=0 and shared by all model seeds | Isolates seed variation to init + data order, which is what "emergence varies across seeds" requires. A is saved with each run. |
| 9 | Checkpoint schedule | Powers of 2 up to 8192, plus every 1000 steps, plus step 0 and final | Mirrors the Pythia-style schedule the paper uses for its LM analysis. Needed later for the patching experiment (step 4). |
| 10 | Parent-mass metric | Mean attention mass a head places on the s true parent positions of each output bit | Not in the paper; our diagnostic. Directly measures alignment with the ground-truth sparse pattern and feeds the step-8 extension. |
| 11 | Plateau-end threshold (step 3 sweeps) | First eval step with output loss < 0.55 | Paper uses "loss < 1.3" for the C=4 cellular automata task (chance ln 4 ~= 1.386); the analogous just-below-chance line for the binary task (chance ln 2 ~= 0.693) is ambiguous, so 0.55 chosen as clearly below noise. Runs that never cross are censored at the step budget, so plotted means are lower bounds wherever censoring occurs. |
