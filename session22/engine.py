import math
import time
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from tsai_gpt.model import GPT
from tsai_gpt.speed_monitor import (SpeedMonitorBase, estimate_flops,
                                    measure_flops)
from tsai_gpt.utils import chunked_cross_entropy


def get_lr(
    it: int, warmup_iters: int, learning_rate: float, min_lr: float, lr_decay_iters: int
) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def train(
    fabric: L.Fabric,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    speed_monitor: SpeedMonitorBase,
    args: Any,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        validate(fabric, model, val_dataloader)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * args.micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (args.micro_batch_size, model.max_seq_length))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    for state["iter_num"], train_data in enumerate(train_dataloader, state["iter_num"]):
        if state["iter_num"] >= args.max_iters:
            checkpoint_path = args.out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"], args.warmup_iters, args.learning_rate, args.min_lr, args.lr) if args.decay_lr else args.lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.max_seq_length].contiguous()
        targets = train_data[:, 1 : model.max_seq_length + 1].contiguous()

        is_accumulating = (state["iter_num"] + 1) % args.gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / args.gradient_accumulation_steps)

        # return

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        t1 = time.perf_counter()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (state["iter_num"] + 1) * args.micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        if state["iter_num"] % args.log_interval == 0:
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, LR: {lr:.6f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if (
            val_dataloader is not None
            and not is_accumulating
            and state["step_count"] % args.eval_interval == 0
        ):
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, args.eval_iters)
            t1 = time.perf_counter() - t0
            speed_monitor.eval_end(t1)
            fabric.print(
                f"step {state['iter_num']}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms"
            )
            fabric.barrier()
        if not is_accumulating and state["step_count"] % args.save_interval == 0:
            checkpoint_path = args.out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)
    # svaing the last checkpoint
    checkpoint_path = args.out_dir / f"last-iter-{state['iter_num']:06d}-ckpt.pth"
    fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
    fabric.save(checkpoint_path, state)


@torch.inference_mode()
def validate(
    fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, eval_iters: int
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        input_ids = val_data[:, 0 : model.max_seq_length].contiguous()
        targets = val_data[:, 1 : model.max_seq_length + 1].contiguous()
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits, targets, chunk_size=0)
    out = losses.mean()

    model.train()
    return out
