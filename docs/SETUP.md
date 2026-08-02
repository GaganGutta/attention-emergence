# Environment notes

Python 3.12, PyTorch >= 2.2 (CPU is sufficient), numpy, matplotlib.
Steps 1-5 need nothing else. Step 6 adds `transformer-lens`.

## transformer-lens patches (needed as of 2026-08-01)

The released transformer-lens has two incompatibilities with current
huggingface libraries. If step 6 crashes for you, apply these one-line
fixes inside the installed package (site-packages/transformer_lens):

1. **Empty-token crash** (`httpx.LocalProtocolError: Illegal header value
   b'Bearer '`): the code reads `os.environ.get("HF_TOKEN", "")` and
   passes the empty string to the hub, which sends a blank auth header.
   Replace that pattern everywhere with
   `(os.environ.get("HF_TOKEN") or None)`, and change the guards
   `huggingface_token if len(huggingface_token) > 0 else None` to
   `huggingface_token if huggingface_token else None`.

2. **Renamed NeoX head** (`'GPTNeoXForCausalLM' object has no attribute
   'embed_out'`): in `pretrained/weight_conversions/neox.py`, replace
   `neox.embed_out.weight.T` with
   `getattr(neox, "embed_out", getattr(neox, "lm_head", None)).weight.T`.

No HuggingFace account or token is needed; all Pythia checkpoints are
public. Expect roughly 300 MB (14m) to 9 GB (160m) of downloads across a
full localization run; `pythia_eval.py` deletes each snapshot from the
local cache after scoring it (disable with `--keep-cache`), and all
scores are cached in `results/pythia/cache.json` so nothing is ever
downloaded twice.
