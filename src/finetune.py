# -*- coding: utf-8 -*-
"""NSMC sentiment fine-tuning utilities."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def _clean_text(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _read_nsmc_tsv(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = _clean_text(row.get("document"))
            label = row.get("label")
            if not text or label not in {"0", "1"}:
                continue
            rows.append({"text": text, "label": int(label)})
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    rows = _read_nsmc_tsv(train_tsv_path)
    rng = random.Random(seed)
    rng.shuffle(rows)

    if rows and val_ratio > 0:
        val_size = max(1, int(len(rows) * val_ratio))
    else:
        val_size = 0
    val_size = min(val_size, len(rows))
    val_data = rows[:val_size]
    train_data = rows[val_size:]
    test_data = _read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    if output_dir is not None:
        output_dir = Path(output_dir)
        _write_jsonl(output_dir / "nsmc_sentiment_train.jsonl", train_data)
        _write_jsonl(output_dir / "nsmc_sentiment_val.jsonl", val_data)
        _write_jsonl(output_dir / "nsmc_sentiment_test.jsonl", test_data)

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """Dataset returning padded review token IDs and one sentiment label."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.data[idx]
        ids = self.tokenizer.encode(str(item["text"]), add_bos_eos=True)
        ids = ids[: self.max_length]
        if len(ids) < self.max_length:
            ids = ids + [self.pad_id] * (self.max_length - len(ids))
        label = int(item["label"])
        return torch.tensor(ids, dtype=torch.long), label


class GPTForSequenceClassification(nn.Module):
    """GPT backbone with a small classification head."""

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        emb_dim = gpt_model.config["emb_dim"]
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.gpt, "forward_features"):
            hidden = self.gpt.forward_features(input_ids)
        else:
            hidden = self.gpt.embedding(input_ids)
            for block in self.gpt.blocks:
                hidden = block(hidden, causal_mask=True)
            hidden = self.gpt.final_norm(hidden)

        pooled = hidden[:, -1, :]
        logits = self.classifier(self.dropout(pooled))
        if labels is None:
            return logits

        labels = labels.to(logits.device).long()
        loss = F.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.to(device)
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device).long()
        optimizer.zero_grad(set_to_none=True)
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total += batch_size

    return total_loss / max(1, total), correct / max(1, total)


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    model.to(device)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device).long()
            loss, logits = model(input_ids, labels=labels)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += batch_size

    model.train(was_training)
    return total_loss / max(1, total), correct / max(1, total)
