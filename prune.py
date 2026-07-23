"""Structured depth pruning: remove whole transformer decoder blocks.

Importance metric: cosine similarity between a block's input and output
hidden states on a calibration set. Blocks whose output is nearly
identical to their input (high cosine similarity) are close to an
identity function and contribute the least -- these are pruned first.

Usage:
    python prune.py --model Qwen/Qwen2.5-1.5B-Instruct --ratio 0.4 --output models/pruned_40

Runs forward-only passes (no backward), so it works on CPU, GPU auto-detected.
"""
import argparse
import itertools
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_calibration_batches(tokenizer, dataset_name, dataset_config, num_samples, seq_len, batch_size, seed):
    """Stream text, concatenate, and chunk into fixed-length token sequences."""
    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    buf = []
    chunks = []
    for row in ds:
        text = row.get("text", "")
        if not text or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        while len(buf) >= seq_len:
            chunks.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(chunks) >= num_samples:
            break

    if len(chunks) < num_samples:
        raise RuntimeError(
            f"Only collected {len(chunks)} calibration chunks of {seq_len} tokens, "
            f"needed {num_samples}. Try a larger --calib-dataset slice or smaller --seq-len."
        )

    chunks = chunks[:num_samples]
    batches = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batches.append(torch.tensor(batch, dtype=torch.long))
    return batches


def get_decoder_layers(model):
    # Works for Llama/Qwen2-family architectures (model.model.layers)
    try:
        return model.model.layers
    except AttributeError as e:
        raise RuntimeError(
            "Could not find model.model.layers -- this script assumes a "
            "Llama/Qwen2-style architecture."
        ) from e


def score_layers(model, layers, batches, device):
    """Returns list of avg cosine similarity (input vs output) per layer index."""
    num_layers = len(layers)
    sim_sum = [0.0] * num_layers
    tok_count = [0] * num_layers

    captured = {}

    def make_hook(idx):
        def hook(module, inputs, output):
            hidden_in = inputs[0]
            hidden_out = output[0] if isinstance(output, tuple) else output
            captured[idx] = (hidden_in.detach(), hidden_out.detach())
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    model.eval()
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            model(input_ids=batch, use_cache=False)
            for idx in range(num_layers):
                h_in, h_out = captured[idx]
                sim = F.cosine_similarity(
                    h_in.flatten(0, 1).float(), h_out.flatten(0, 1).float(), dim=-1
                )
                sim_sum[idx] += sim.sum().item()
                tok_count[idx] += sim.numel()
            captured.clear()

    for h in handles:
        h.remove()

    return [sim_sum[i] / tok_count[i] for i in range(num_layers)]


def choose_layers_to_remove(layer_params, scores, ratio, protect):
    num_layers = len(scores)
    total_layer_params = sum(layer_params)
    avg_layer_params = total_layer_params / num_layers

    target_removed_params = ratio * sum(layer_params)  # ratio applied against block params
    k = round(target_removed_params / avg_layer_params)
    k = max(1, min(k, num_layers - 1 - 2 * protect))

    candidates = list(range(protect, num_layers - protect))
    # highest cosine similarity = closest to identity = least important = prune first
    candidates.sort(key=lambda i: scores[i], reverse=True)
    remove = sorted(candidates[:k])
    return remove


def reindex_layer_idx(layers):
    for new_idx, layer in enumerate(layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_idx


def reslice_per_layer_config(config, num_layers_before, kept_indices):
    """Some configs (e.g. recent transformers versions add `layer_types`)
    carry a list with one entry per decoder layer, parallel to
    num_hidden_layers. Slice any such list down to the kept layers so it
    stays consistent after pruning -- otherwise config validation on save
    fails with a length mismatch."""
    for attr_name, val in list(vars(config).items()):
        if isinstance(val, list) and len(val) == num_layers_before:
            setattr(config, attr_name, [val[i] for i in kept_indices])


def sanity_generate(model, tokenizer, device, num_prompts, max_new_tokens):
    prompts = [
        "The capital of France is",
        "Explain what a transformer model is in one sentence:",
        "def fibonacci(n):",
    ][:num_prompts]
    model.eval()
    outputs = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs.append({"prompt": p, "completion": text})
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ratio", type=float, required=True, help="target fraction of params to remove, e.g. 0.2/0.4/0.6")
    ap.add_argument("--output", required=True)
    ap.add_argument("--calib-dataset", default="Salesforce/wikitext")
    ap.add_argument("--calib-config", default="wikitext-103-raw-v1")
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP.keys()))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--protect-layers", type=int, default=0,
        help="exclude this many layers at the start AND end from pruning candidates "
             "(common practice to avoid degenerate first/last-layer removal; off by default "
             "to match the spec exactly -- turn on if sanity-check generations look broken)",
    )
    ap.add_argument("--num-generate", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    dtype = DTYPE_MAP[args.dtype] if device != "cpu" else torch.float32
    print(f"[prune] device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[prune] loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)

    total_params_before = sum(p.numel() for p in model.parameters())
    layers = get_decoder_layers(model)
    num_layers_before = len(layers)
    layer_params = [sum(p.numel() for p in layer.parameters()) for layer in layers]

    print(f"[prune] building calibration set ({args.calib_samples} x {args.seq_len} tokens) ...")
    batches = build_calibration_batches(
        tokenizer, args.calib_dataset, args.calib_config,
        args.calib_samples, args.seq_len, args.batch_size, args.seed,
    )

    print(f"[prune] scoring {num_layers_before} layers ...")
    t0 = time.time()
    scores = score_layers(model, layers, batches, device)
    print(f"[prune] scoring done in {time.time() - t0:.1f}s")
    for i, s in enumerate(scores):
        print(f"  layer {i:2d}  cosine_sim(in,out) = {s:.4f}")

    remove_idx = choose_layers_to_remove(layer_params, scores, args.ratio, args.protect_layers)
    print(f"[prune] removing layers: {remove_idx}")

    kept_indices = [i for i in range(num_layers_before) if i not in remove_idx]
    kept = [layers[i] for i in kept_indices]
    new_module_list = torch.nn.ModuleList(kept)
    reindex_layer_idx(new_module_list)
    reslice_per_layer_config(model.config, num_layers_before, kept_indices)
    model.model.layers = new_module_list
    model.config.num_hidden_layers = len(kept)

    total_params_after = sum(p.numel() for p in model.parameters())
    achieved_ratio = 1 - (total_params_after / total_params_before)
    print(f"[prune] params before={total_params_before:,} after={total_params_after:,} "
          f"achieved_ratio={achieved_ratio:.3f} (requested {args.ratio})")

    print(f"[prune] running sanity-check generations ...")
    sanity = sanity_generate(model, tokenizer, device, args.num_generate, args.max_new_tokens)
    for s in sanity:
        print(f"\n  PROMPT: {s['prompt']}\n  OUTPUT: {s['completion']}")

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    sidecar = {
        "base_model": args.model,
        "requested_ratio": args.ratio,
        "achieved_ratio": achieved_ratio,
        "num_layers_before": num_layers_before,
        "num_layers_after": len(kept),
        "removed_layer_indices": remove_idx,
        "protect_layers": args.protect_layers,
        "layer_scores": scores,
        "total_params_before": total_params_before,
        "total_params_after": total_params_after,
        "calibration": {
            "dataset": args.calib_dataset,
            "config": args.calib_config,
            "num_samples": args.calib_samples,
            "seq_len": args.seq_len,
            "seed": args.seed,
        },
        "sanity_check_generations": sanity,
    }
    with open(os.path.join(args.output, "prune_info.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[prune] saved pruned model + sidecar to {args.output}")


if __name__ == "__main__":
    main()
