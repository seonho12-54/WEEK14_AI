# -*- coding: utf-8 -*-
"""GPT pretraining utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    model = model.to(device)
    loss, _ = model(input_batch, targets=target_batch)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    if len(data_loader) == 0:
        return float("nan")

    was_training = model.training
    model.eval()
    losses: list[float] = []
    max_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            losses.append(float(loss.item()))

    model.train(was_training)
    return sum(losses) / max(1, len(losses))


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": getattr(model, "config", None),
        },
        path_obj,
    )


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    idx = idx.to(device)
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -context_size:]
                logits = model(idx_cond)
                if isinstance(logits, tuple):
                    logits = logits[1]
                logits = logits[:, -1, :]

                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, top_k)
                    min_values = values[:, [-1]]
                    logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

                if temperature <= 0:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)

                idx = torch.cat((idx, next_id), dim=1)
                if eos_id is not None and torch.all(next_id == eos_id):
                    break
    finally:
        model.train(was_training)
    return idx


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    model.to(device)
    input_ids = tokenizer.encode(start_context, add_bos_eos=False)
    idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    out = generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=getattr(tokenizer, "get_eos_id", lambda: None)(),
    )
    print(tokenizer.decode(out[0].tolist(), skip_special=True))


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    model.to(device)
    train_losses: list[float] = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_losses: list[float] = []
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_losses.append(float(loss.item()))

            if eval_freq and global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                print(f"step {global_step}: train {train_loss:.4f}, val {val_loss:.4f}")

            if ckpt_freq and global_step % ckpt_freq == 0:
                save_checkpoint(model, optimizer, epoch=epoch, global_step=global_step, path=f"checkpoint_step_{global_step}.pt")

        train_losses.append(sum(epoch_losses) / max(1, len(epoch_losses)))
        if start_context:
            generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """Plot training/validation losses."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
