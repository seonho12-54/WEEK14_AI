# -*- coding: utf-8 -*-
"""
Mini GPT pretraining experiment runner.

This file is intentionally separate from gpt-lab.ipynb and src/.
Use it to change model/training settings, compare train/validation loss,
and save a dense loss graph for reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from bpe import BPETokenizer  # noqa: E402
from dataset import create_dataloader  # noqa: E402
from model import GPTModel  # noqa: E402
from train import calc_loss_batch, calc_loss_loader, generate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mini GPT pretraining experiment.")

    parser.add_argument("--vocab-path", type=Path, default=ROOT / "data" / "nsmc_bpe_vocab_3000.json")
    parser.add_argument("--train-text", type=Path, default=ROOT / "data" / "nsmc_lm_train.txt")
    parser.add_argument("--val-text", type=Path, default=ROOT / "data" / "nsmc_lm_val.txt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "outputs")

    parser.add_argument("--train-text-limit", type=int, default=200_000)
    parser.add_argument("--val-text-limit", type=int, default=50_000)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--qkv-bias", action="store_true")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)

    parser.add_argument("--sample-context", type=str, default="이 영화는")
    parser.add_argument("--sample-contexts", nargs="*", default=None)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--sample-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--show-plot", action="store_true")

    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def read_text(path: Path, limit: int | None) -> str:
    text = path.read_text(encoding="utf-8")
    if limit is not None and limit > 0:
        return text[:limit]
    return text


def load_tokenizer(path: Path) -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.load(path)
    return tokenizer


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def moving_average(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return []

    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def plot_loss_curves(
    train_steps: list[int],
    train_losses: list[float],
    val_steps: list[int],
    val_losses: list[float],
    out_path: Path,
    show_plot: bool = False,
) -> None:
    plt.figure(figsize=(11, 6))

    train_smooth = moving_average(train_losses, window=5)
    plt.plot(train_steps, train_losses, color="#4e79a7", alpha=0.25, linewidth=1, label="Train loss raw")
    plt.plot(train_steps, train_smooth, color="#4e79a7", linewidth=2, label="Train loss smooth")

    if val_losses:
        plt.plot(val_steps, val_losses, color="#f28e2b", linestyle="--", marker="o", markersize=3, label="Validation loss")

        train_at_val = []
        for step in val_steps:
            if step in train_steps:
                train_at_val.append(train_losses[train_steps.index(step)])
            else:
                nearest_idx = min(range(len(train_steps)), key=lambda i: abs(train_steps[i] - step))
                train_at_val.append(train_losses[nearest_idx])

        plt.fill_between(
            val_steps,
            train_at_val,
            val_losses,
            color="#f28e2b",
            alpha=0.16,
            label="Train/Val gap",
        )

    plt.title("Mini GPT Pretraining Loss")
    plt.xlabel("Global step")
    plt.ylabel("Cross entropy loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    print(f"Saved loss graph: {out_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_sample(
    model: GPTModel,
    tokenizer: BPETokenizer,
    device: torch.device,
    context: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    model.eval()
    idx = torch.tensor(tokenizer.encode(context), dtype=torch.long, device=device).unsqueeze(0)
    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=model.config["context_length"],
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.get_eos_id(),
    )
    model.train()
    return tokenizer.decode(out[0].tolist())


def print_samples(
    model: GPTModel,
    tokenizer: BPETokenizer,
    device: torch.device,
    contexts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    title: str,
) -> None:
    print(f"\n{title}")
    for context in contexts:
        sample = generate_sample(
            model=model,
            tokenizer=tokenizer,
            device=device,
            context=context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        print(f"[prompt] {context}")
        print(sample)
        print()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.vocab_path)
    sample_contexts = args.sample_contexts if args.sample_contexts else [args.sample_context]
    train_text = read_text(args.train_text, args.train_text_limit)
    val_text = read_text(args.val_text, args.val_text_limit)

    print(f"Device: {device}")
    print(f"Tokenizer vocab: {len(tokenizer.id_to_token):,}")
    print(f"Train chars: {len(train_text):,} / Val chars: {len(val_text):,}")

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    print(f"Train tokens: {len(train_ids):,} / Val tokens: {len(val_ids):,}")

    train_loader = create_dataloader(
        train_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=args.stride,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=args.stride,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    config = {
        "vocab_size": len(tokenizer.id_to_token),
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
    }

    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Model params: {count_parameters(model):,}")
    print(f"Train batches per epoch: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")

    train_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []

    global_step = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % args.log_every == 0:
                train_steps.append(global_step)
                train_losses.append(loss.item())
                print(f"step {global_step:5d} | train loss {loss.item():.4f}")

            if global_step % args.eval_every == 0:
                train_loss_for_gap = loss.item()
                val_loss = calc_loss_loader(
                    val_loader,
                    model,
                    device,
                    num_batches=args.eval_batches,
                )
                gap = val_loss - train_loss_for_gap
                val_steps.append(global_step)
                val_losses.append(val_loss)
                print(
                    f"step {global_step:5d} | "
                    f"train loss {train_loss_for_gap:.4f} | "
                    f"val loss {val_loss:.4f} | "
                    f"gap {gap:+.4f}"
                )

            if args.sample_every is not None and args.sample_every > 0 and global_step % args.sample_every == 0:
                print_samples(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    contexts=sample_contexts,
                    max_new_tokens=args.sample_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    title=f"Samples at step {global_step}",
                )

            if args.max_steps is not None and args.max_steps > 0 and global_step >= args.max_steps:
                break

        print_samples(
            model=model,
            tokenizer=tokenizer,
            device=device,
            contexts=sample_contexts,
            max_new_tokens=args.sample_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            title=f"Samples after epoch {epoch}",
        )

        if args.max_steps is not None and args.max_steps > 0 and global_step >= args.max_steps:
            break

    graph_path = args.out_dir / "pretrain_loss_curve.png"
    plot_loss_curves(train_steps, train_losses, val_steps, val_losses, graph_path, show_plot=args.show_plot)

    checkpoint_path = args.out_dir / "pretrain_experiment_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "global_step": global_step,
            "train_steps": train_steps,
            "train_losses": train_losses,
            "val_steps": val_steps,
            "val_losses": val_losses,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
