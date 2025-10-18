"""
Minimal training script for the CS336 adapters.py stack.

Features
- CLI to configure model + optimizer hyperparameters
- Memory‑efficient dataset loading with np.memmap / mmap_mode
- Cosine LR with warmup (uses run_get_lr_cosine_schedule)
- Gradient clipping (uses run_gradient_clipping)
- Periodic eval on a validation memmap
- Checkpoint save/load to a user path (run_save_checkpoint / run_load_checkpoint)
- Optional Weights & Biases logging (disabled by default)

Expected datasets: 1D arrays of integer token IDs (dtype int32/int64).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tests.adapters import (
    MyTransformerLM,
    get_adamw_cls,
    run_get_batch,
    run_cross_entropy,
    run_gradient_clipping,
    run_get_lr_cosine_schedule,
    run_save_checkpoint,
    run_load_checkpoint,
)


def str2dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp32", "float32"}: return torch.float32
    if name in {"bf16", "bfloat16"}: return torch.bfloat16
    if name in {"fp16", "float16"}: return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def load_tokens_memmap(path: str, dtype: str = "int64") -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        arr = np.load(path, mmap_mode="r")  # preserves dtype stored in file
    else:
        np_dtype = np.int64 if dtype == "int64" else np.int32
        arr = np.memmap(path, dtype=np_dtype, mode="r")
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def maybe_init_wandb(args) -> Optional[object]:
    if not args.wandb:
        return None
    try:
        import wandb  
    except Exception as e:
        print(f"[wandb] Disabled (import failed: {e})")
        return None
    run = wandb.init(project=args.wandb_project, name=args.run_name or None, config=vars(args))
    return run


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, 
    val_tokens: np.ndarray, 
    context_length: int, 
    batch_size: int, 
    device: str, 
    num_batches: int = 50
) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = run_get_batch(val_tokens, batch_size, context_length, device)
        ## x, y: (bsz, len)
        logits = model(x)  ## (bsz, len, voc)
        loss = run_cross_entropy(logits, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    parser = argparse.ArgumentParser(description="Train a tiny Transformer LM")

    # Data
    parser.add_argument("--train_path", type=str, required=True, help="Path to training tokens (memmap or .npy)")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation tokens (memmap or .npy)")
    parser.add_argument("--token_dtype", type=str, default="int64", choices=["int32", "int64"], help="dtype of raw memmap file when not .npy")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)

    # Model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Optim & schedule
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--total_iters", type=int, default=40000, help="Number of optimizer steps (also cosine cycle length)")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Logging / eval / ckpt
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--ckpt_interval", type=int, default=5000)
    parser.add_argument("--out_dir", type=str, default="save")
    parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume from")

    # System
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"])
    parser.add_argument("--run_name", type=str, default="train01")

    # W&B (optional)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="cs336")

    args = parser.parse_args()

    # Dirs & seed
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device / dtype
    device = args.device
    dtype = str2dtype(args.dtype)

    # Data (memory‑mapped)
    train_tokens = load_tokens_memmap(args.train_path, args.token_dtype)
    val_tokens = load_tokens_memmap(args.val_path, args.token_dtype)
    print(f"Loaded train tokens: {train_tokens.shape}, val tokens: {val_tokens.shape}")

    # Model & optimizer
    model = MyTransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        device=torch.device(device),
        dtype=dtype,
    )
    model.to(device=device, dtype=dtype)
    model.train()

    OptimCls = get_adamw_cls()
    optimizer = OptimCls(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        try:
            start_iter = int(run_load_checkpoint(args.resume, model, optimizer))
            print(f"Resumed from {args.resume} at iteration {start_iter}")
        except Exception as e:
            print(f"[warn] Failed to resume from {args.resume}: {e}")

    # Optional wandb
    wb_run = maybe_init_wandb(args)

    # Training loop
    tokens_per_step = args.batch_size * args.context_length
    running_loss: float = 0.0
    t_last = time.time()

    for it in range(start_iter, args.total_iters):
        # LR schedule
        lr = run_get_lr_cosine_schedule(
            it,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.total_iters,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = run_get_batch(train_tokens, args.batch_size, args.context_length, device)
        logits = model(x) 
        loss = run_cross_entropy(logits, y)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            run_gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        # Logging
        running_loss += float(loss.item())
        if args.log_interval != -1 and (it + 1) % args.log_interval == 0:
            now = time.time()
            dt = max(1e-9, now - t_last)
            t_last = now
            tok_per_s = tokens_per_step * args.log_interval / dt
            avg_loss = running_loss / args.log_interval
            running_loss = 0.0
            msg = f"iter {it+1:>8d} | lr {lr:.3e} | loss {avg_loss:.4f} | {tok_per_s:.0f} tok/s"
            print(msg)
            if wb_run is not None:
                try:
                    import wandb  # type: ignore
                    wandb.log({"iter": it + 1, "lr": lr, "train/loss": avg_loss, "throughput_tok_s": tok_per_s})
                except Exception:
                    pass

        # Eval
        if args.eval_interval != -1 and (it + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, val_tokens, args.context_length, args.batch_size, device, args.eval_batches)
            print(f"[eval] iter {it+1} | val/loss {val_loss:.4f}")
            if wb_run is not None:
                try:
                    import wandb  # type: ignore
                    wandb.log({"iter": it + 1, "val/loss": val_loss})
                except Exception:
                    pass

        # Checkpoint
        if (args.ckpt_interval != -1 and (it + 1) % args.ckpt_interval == 0) \
            or (it + 1) == args.total_iters:
            ckpt_path = out_dir / f"ckpt_iter_{it+1}.pt"
            run_save_checkpoint(model, optimizer, it + 1, str(ckpt_path))
            print(f"[ckpt] saved to {ckpt_path}")

    if wb_run is not None:
        try:
            wb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
