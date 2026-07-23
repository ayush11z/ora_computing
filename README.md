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
- [x] `distill.py` — fixed-budget LoRA distillation recovery (ready to run; smoke-test with `--steps 50` before the full budget)
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

## Step 2: Distill (fixed-budget recovery)

Smoke test first — 50 steps, not the full budget, to catch bugs cheaply:

```bash
python distill.py \
  --student models/pruned_40 \
  --teacher Qwen/Qwen2.5-1.5B-Instruct \
  --steps 50 \
  --output models/distilled_40_test
```

If that runs clean and the loss trends down, do the real run (the flag
just defaults to this, so you can drop `--steps` entirely):

```bash
python distill.py \
  --student models/pruned_40 \
  --teacher Qwen/Qwen2.5-1.5B-Instruct \
  --output models/distilled_40
```

And the zero-recovery "before" baseline for the same ratio (no training,
just a manifest pointing back at the pruned model, for `evaluate.py` to
consume uniformly alongside the distilled variants):

```bash
python distill.py \
  --student models/pruned_40 \
  --teacher Qwen/Qwen2.5-1.5B-Instruct \
  --no-distill \
  --output models/baseline_40
```

What it does:
1. Loads the teacher (frozen, full-precision-relative-to-`--dtype`) and the
   pruned student, wraps the student in a LoRA adapter (rank 16 / alpha 32,
   applied to all attention + MLP projection layers, adapter weights kept
   fp32 for optimizer stability even though the frozen base stays fp16).
2. Streams chat-formatted examples from `HuggingFaceH4/ultrachat_200k`
   (`train_sft` split), tokenized with the model's own chat template.
3. Each step: run both teacher and student on the *same* batch, minimize
   temperature-scaled KL divergence between their next-token distributions
   (`--temperature`, default 2.0), optionally blended with a hard-label
   cross-entropy term via `--ce-weight` (default 0.0 = pure soft-label KD;
   spec suggests ~0.1 if training looks unstable).
4. Saves only the LoRA adapter (small) + a `distill_info.json` manifest
   (loss curve, tokens seen, exact config) — not a full copy of the base
   model. Reload later with `PeftModel.from_pretrained(base_model, output_dir)`.

**Fixed budget, held constant across every ratio:** `FULL_BUDGET_STEPS = 1000`
gradient update steps (a module-level constant, not a tunable — the whole
point of the ablation is that every ratio gets the same budget). Passing a
different `--steps` prints a loud warning and the manifest records
`is_official_budget_run: false`, so a smoke test can never be silently
mistaken for a real ablation run later. Token count is a *consequence* of
`--batch-size × --seq-len × --steps`, not the thing held constant — the
spec allowed either as the fixed unit; we picked steps since that's what
the spec's own CLI example uses. Defaults (batch 4 × seq_len 512) land
at ~2.05M tokens over the full budget, sized to comfortably fit a single
T4/A10; bump `--batch-size`/`--seq-len`/`--grad-accum-steps` if you want
to push closer to the spec's illustrative ~10M-token figure, but do it
identically across all three ratios or the ablation stops being apples-to-apples.

Run this three times per ratio (baseline / smoke-test / full budget) for
each of the three pruning ratios once the smoke test looks healthy.

## Repo layout

```
ora-distillation-ablation/
├── README.md
├── requirements.txt
├── prune.py
├── distill.py
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
