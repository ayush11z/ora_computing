# Distillation Recovery Under Pruning — Ablation Study

**Question:** at a fixed distillation compute budget, how much accuracy is
recoverable at different pruning ratios (20% / 40% / 60%), and where are the
diminishing returns?

This probes the same "prune → quantize → retrain" pipeline used in
commercial LLM compression tooling: structured depth pruning of a small
instruction-tuned model, followed by a *fixed-budget* LoRA distillation
recovery pass, evaluated on standard benchmarks and profiled for on-disk
size and CPU inference throughput.

Base model: `Qwen/Qwen2.5-1.5B-Instruct`.

## Status

- [x] `prune.py` — structured depth pruning (this is ready to run)
- [ ] `distill.py` — fixed-budget LoRA distillation recovery
- [ ] `evaluate.py` — lm-evaluation-harness benchmark runs
- [ ] `profile.py` — GGUF size + CPU throughput profiling
- [ ] `analyze.py` — ablation table + plots

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU notes:
- `prune.py` is forward-pass only (calibration + generation) — runs on CPU,
  GPU speeds it up but isn't required.
- `distill.py` will require a GPU (T4/A10-class or better) — backward
  passes over teacher + student for 1000 steps.
- `evaluate.py` is CPU-runnable but very slow across 7 model variants ×
  4 benchmarks — GPU strongly recommended.
- `profile.py` is CPU-only by design (that's what it's measuring).
- `analyze.py` is CPU-only (aggregation + plotting).

## Step 1: Prune

```bash
python prune.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --ratio 0.4 \
  --output models/pruned_40
```

What it does:
1. Streams ~512 chunks of `wikitext-103-raw-v1` (512 tokens each) as a
   calibration set.
2. Scores every decoder block by cosine similarity between its input and
   output hidden states on that calibration set — blocks close to an
   identity function (high similarity) are the least useful and are
   pruned first.
3. Picks the number of blocks to remove so the achieved parameter
   reduction matches `--ratio` (blocks aren't perfectly equal-sized in
   general, so the script computes this from actual per-block param
   counts rather than assuming `ratio == fraction of layers`).
4. Removes those blocks, re-indexes the remaining `ModuleList`, and
   updates `config.num_hidden_layers` so the model loads cleanly with
   standard `AutoModelForCausalLM.from_pretrained`.
5. Runs a few greedy-decoded sanity-check generations and prints them —
   **read these before moving on to distillation.** A pruned model that
   already produces garbage at this stage means the pruning ratio is too
   aggressive or the importance metric picked a bad set of layers.
6. Saves the pruned model + tokenizer to `--output`, plus a
   `prune_info.json` sidecar with removed-layer indices, per-layer
   cosine-similarity scores, and before/after parameter counts.

Run it once per ratio:

```bash
python prune.py --model Qwen/Qwen2.5-1.5B-Instruct --ratio 0.2 --output models/pruned_20
python prune.py --model Qwen/Qwen2.5-1.5B-Instruct --ratio 0.4 --output models/pruned_40
python prune.py --model Qwen/Qwen2.5-1.5B-Instruct --ratio 0.6 --output models/pruned_60
```

Useful flags:
- `--protect-layers N` excludes the first/last `N` blocks from pruning
  candidates. Off by default (matches the spec's importance metric
  exactly, no exceptions) — turn it on if the sanity-check generations
  come back incoherent, since removing an early or final block is a
  classic way to break a model outright.
- `--calib-samples`, `--seq-len` control calibration cost if the default
  (512 × 512 tokens) is too slow on your hardware.

Paste me the console output (per-layer scores + the sanity-check
generations + `prune_info.json`) once you've run this on your GPU box and
we'll decide whether to move on to `distill.py` or adjust the pruning
config first.

## Repo layout

```
ora-distillation-ablation/
├── README.md
├── requirements.txt
├── prune.py
├── distill.py        (not yet implemented)
├── evaluate.py        (not yet implemented)
├── profile.py         (not yet implemented)
├── analyze.py          (not yet implemented)
├── configs/
├── results/
│   ├── summary.csv
│   ├── plots/
│   └── *.json
└── WRITEUP.md          (not yet implemented)
```
