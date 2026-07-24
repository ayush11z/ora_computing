"""Fixed-budget LoRA knowledge distillation: recover accuracy lost to pruning.

Teacher = original unpruned base model (frozen, fp32 "soft label" source).
Student = a pruned model (from prune.py), LoRA-adapted and trained to match
the teacher's output distribution via KL divergence, optionally blended
with a small hard-label cross-entropy term for stability.

FIXED BUDGET: the whole point of this ablation is that every ratio gets the
*same* compute budget, so it's expressed as a hard constant, FULL_BUDGET_STEPS
= 1000 gradient update steps -- not a per-run tunable. --steps defaults to
that constant; passing a different value (e.g. --steps 50 for a cheap
end-to-end smoke test) prints a loud warning so a test run can never be
silently mistaken for an official ablation run.

Token budget: with the default --batch-size 1 x --grad-accum-steps 4 x
--seq-len 512, that's 2048 tokens/step (one optimizer update, i.e. one
entry of "step" in FULL_BUDGET_STEPS, spans 4 micro-batches), ~2.05M
tokens over the full 1000-step budget. This (not ~10M) is the realized
token count for the default config -- documented here since the spec
allowed either steps or tokens as the fixed unit and we picked steps
(matching the spec's own CLI example). Micro-batch is kept at 1 rather
than upping --batch-size directly because Qwen2.5's ~152K vocabulary
makes the (batch, seq, vocab) tensors in the KD loss large (~1.16GB each
at batch=4/seq=512/fp32) -- gradient accumulation reaches the same
effective batch/step without holding that many at once, so it fits a
single T4's ~15GB.

Usage:
    python distill.py --student models/pruned_40 --teacher Qwen/Qwen2.5-1.5B-Instruct \
        --steps 1000 --output models/distilled_40

    # cheap smoke test before committing to the full budget:
    python distill.py --student models/pruned_40 --teacher Qwen/Qwen2.5-1.5B-Instruct \
        --steps 50 --output models/distilled_40_test

    # "before" baseline, zero recovery training:
    python distill.py --student models/pruned_40 --teacher Qwen/Qwen2.5-1.5B-Instruct \
        --no-distill --output models/baseline_40
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

FULL_BUDGET_STEPS = 1000
DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_training_batches(tokenizer, dataset_name, num_examples, seq_len, seed):
    """Stream chat-formatted examples, tokenize with the model's chat template,
    pad/truncate to seq_len. Returns a list of {input_ids, attention_mask} tensors."""
    ds = load_dataset(dataset_name, split="train_sft", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    examples = []
    pbar = tqdm(total=num_examples, desc="streaming training examples", unit="ex")
    for row in ds:
        messages = row.get("messages")
        if not messages:
            continue
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, truncation=True, max_length=seq_len, return_tensors="pt")
        if enc["input_ids"].shape[1] < 8:
            continue
        examples.append(enc)
        pbar.update(1)
        if len(examples) >= num_examples:
            break
    pbar.close()

    if len(examples) < num_examples:
        raise RuntimeError(
            f"Only collected {len(examples)} training examples, needed {num_examples}. "
            f"Try a smaller --batch-size/--steps or a different --data-dataset."
        )
    return examples


def collate(batch, pad_token_id):
    max_len = max(ex["input_ids"].shape[1] for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        L = ex["input_ids"].shape[1]
        input_ids[i, :L] = ex["input_ids"][0]
        attn[i, :L] = ex["attention_mask"][0]
    return input_ids, attn


def kd_loss(student_logits, teacher_logits, attention_mask, temperature):
    mask = attention_mask.bool().unsqueeze(-1)
    s = student_logits / temperature
    t = teacher_logits / temperature
    log_p_student = F.log_softmax(s, dim=-1)
    p_teacher = F.softmax(t, dim=-1)
    per_token_kl = F.kl_div(log_p_student, p_teacher, reduction="none").sum(-1, keepdim=True)
    per_token_kl = per_token_kl.masked_fill(~mask, 0.0)
    denom = mask.sum().clamp(min=1)
    return (per_token_kl.sum() / denom) * (temperature ** 2)


def ce_loss(student_logits, input_ids, attention_mask):
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    labels = shift_labels.masked_fill(shift_mask == 0, -100)
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1), ignore_index=-100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="path to a pruned model dir (output of prune.py)")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-distill", action="store_true",
                     help="skip training entirely; write a manifest marking this as the "
                          "zero-recovery 'before' baseline for this pruning ratio")
    ap.add_argument("--steps", type=int, default=FULL_BUDGET_STEPS,
                     help=f"gradient update steps; official ablation runs must use "
                          f"{FULL_BUDGET_STEPS} (the default) -- override only for smoke tests")
    ap.add_argument("--batch-size", type=int, default=1,
                     help="micro-batch size; Qwen2.5's ~152K vocab makes the (batch, seq, vocab) "
                          "KD-loss tensors large, so this stays small by default and "
                          "--grad-accum-steps makes up the effective batch instead, to fit a T4")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--ce-weight", type=float, default=0.0,
                     help="weight of hard-label CE term blended with KD loss (spec suggests ~0.1 "
                          "for stability); 0.0 = pure soft-label distillation")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-target-modules", nargs="+", default=DEFAULT_LORA_TARGETS)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--data-dataset", default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPE_MAP.keys()))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    dtype = DTYPE_MAP[args.dtype] if device != "cpu" else torch.float32
    print(f"[distill] device={device} dtype={dtype}")

    if args.steps != FULL_BUDGET_STEPS and not args.no_distill:
        print(f"[distill] WARNING: --steps={args.steps} differs from the fixed ablation "
              f"budget ({FULL_BUDGET_STEPS}). This run is NOT comparable to the official "
              f"ablation matrix -- use only for smoke-testing the pipeline.")

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.output, exist_ok=True)

    if args.no_distill:
        manifest = {
            "no_distill": True,
            "student_base_path": args.student,
            "teacher": args.teacher,
            "note": "zero-recovery baseline; load student_base_path directly, no adapter here",
        }
        with open(os.path.join(args.output, "distill_info.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[distill] --no-distill: wrote baseline manifest to {args.output}, no training run")
        return

    print(f"[distill] loading teacher {args.teacher} ...")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"[distill] loading student {args.student} ...")
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=dtype).to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_config)
    # keep LoRA adapter weights in fp32 for optimizer stability even though the
    # frozen base model stays fp16 -- autocast handles the mixed-dtype matmuls
    for p in student.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    student.print_trainable_parameters()

    num_examples = args.steps * args.batch_size * args.grad_accum_steps
    print(f"[distill] streaming {num_examples} training examples from {args.data_dataset} ...")
    examples = build_training_batches(
        tokenizer, args.data_dataset, num_examples, args.seq_len, args.seed
    )

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=args.lr
    )
    total_optimizer_steps = args.steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_optimizer_steps),
        num_training_steps=total_optimizer_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and device == "cuda"))
    # autocast on MPS only supports bf16 reliably; fp16 there is prone to NaNs.
    # fp32 (the recommended local default) never needs autocast at all.
    autocast_enabled = dtype != torch.float32 and (device == "cuda" or (device == "mps" and dtype == torch.bfloat16))
    autocast_device_type = device if device in ("cuda", "mps") else "cpu"
    autocast_ctx = torch.autocast(device_type=autocast_device_type, dtype=dtype, enabled=autocast_enabled)

    student.train()
    loss_history = []
    tokens_seen = 0
    example_ptr = 0
    t0 = time.time()

    pbar = tqdm(range(args.steps), desc="distilling", unit="step")
    for step in pbar:
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(args.grad_accum_steps):
            micro_batch = examples[example_ptr : example_ptr + args.batch_size]
            example_ptr += args.batch_size
            input_ids, attn = collate(micro_batch, tokenizer.pad_token_id)
            input_ids, attn = input_ids.to(device), attn.to(device)
            tokens_seen += int(attn.sum().item())

            with autocast_ctx:
                with torch.no_grad():
                    teacher_logits = teacher(input_ids=input_ids, attention_mask=attn).logits
                student_logits = student(input_ids=input_ids, attention_mask=attn).logits
                loss = kd_loss(student_logits, teacher_logits, attn, args.temperature)
                if args.ce_weight > 0:
                    loss = (1 - args.ce_weight) * loss + args.ce_weight * ce_loss(student_logits, input_ids, attn)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            step_loss += loss.item() * args.grad_accum_steps

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_history.append(step_loss)
        pbar.set_postfix(loss=f"{step_loss:.4f}", tokens=tokens_seen)

        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            tqdm.write(f"  step {step:4d}/{args.steps}  loss={step_loss:.4f}  "
                       f"tokens_seen={tokens_seen:,}  elapsed={elapsed:.0f}s")

    print(f"[distill] training done in {time.time() - t0:.0f}s, {tokens_seen:,} tokens seen")

    student.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    manifest = {
        "no_distill": False,
        "student_base_path": args.student,
        "teacher": args.teacher,
        "steps": args.steps,
        "is_official_budget_run": args.steps == FULL_BUDGET_STEPS,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "grad_accum_steps": args.grad_accum_steps,
        "tokens_seen": tokens_seen,
        "temperature": args.temperature,
        "ce_weight": args.ce_weight,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "target_modules": args.lora_target_modules},
        "lr": args.lr,
        "data_dataset": args.data_dataset,
        "seed": args.seed,
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
    }
    with open(os.path.join(args.output, "distill_info.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[distill] saved LoRA-adapted student + manifest to {args.output}")


if __name__ == "__main__":
    main()
