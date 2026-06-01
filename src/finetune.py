# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import csv
import json
import random

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel

#TSV파일 하나를 읽고 깨끗한 데이터 리스트로 바꾸는 함수
def _read_sentiment_tsv(path: str | Path) -> list[dict]:
    #text, label로 변환한 출력용 dict리스트
    rows = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        #파일 하나 열어서 한 줄씩 읽기 시작
        for row in reader:
            text = row.get("document", "")
            label = row.get("label", "")
            #document가 비었다면 건너뜀
            if text is None:
                continue
            text = text.strip()
            if not text:
                continue

            if label not in {"0", "1"}:
                continue

            rows.append({
                "text": text,
                "label": int(label),
            })
    return rows

#데이터 dict리스트를 train/val/test로 나누고 필요하면 저장하는 함수
def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    #raise NotImplementedError("make_sentiment_dataset을 구현하세요.")
    #_read_sentiment_tsv함수로 train파일을 읽고 
    train_rows = _read_sentiment_tsv(train_tsv_path)
    #데이터를 섞음
    rng = random.Random(seed)
    rng.shuffle(train_rows)
    #validation크기를 계산해서 val_data와 test_data로 나눔(val_ratio가 있어서 이렇게 함)
    val_size = max(1, int(len(train_rows) * val_ratio))
    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]
    #test data는 별도 파일에서 읽어옴
    if test_tsv_path is not None:
        test_data = _read_sentiment_tsv(test_tsv_path)
    else:
        test_data = []
    #저장할 폴더가 지정되어 있다면 아래 코드를 실행, 없다면 그냥 반환
    if output_dir is not None:
        #문자열 경로를 Path객체로 바꾸고
        output_dir = Path(output_dir)
        #폴더가 없으면 만들고, 이미 있다면 그냥 넘어감
        output_dir.mkdir(parents=True, exist_ok=True)
        #train_data 안의 dict들을 한 줄씩 JOSN으로 저장
        with open(output_dir / "nsmc_sentiment_train.jsonl", "w", encoding="utf-8") as f:
            for row in train_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(output_dir / "nsmc_sentiment_val.jsonl", "w", encoding="utf-8") as f:
            for row in val_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(output_dir / "nsmc_sentiment_test.jsonl", "w", encoding="utf-8") as f:
            for row in test_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    return train_data, val_data, test_data

class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

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

    #리뷰 하나를 꺼내서 token id tensor로 만들고 max_length로 맞춘 뒤, label과 함께 반환
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        #raise NotImplementedError("ReviewSentimentDataset.__getitem__을 구현하세요.")
        #idx번째 리뷰 꺼내기
        item = self.data[idx]
        #text와 label 분리
        text = item["text"]
        label = int(item["label"])
        #text를 token id 로 encode
        input_ids = self.tokenizer.encode(text, add_bos_eos=True)
        #길이가 길다면 max_length까지 자르기
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        #짧다면 나머지 부분 pad_id로 채우기
        else:
            pad_len = self.max_length - len(input_ids)
            input_ids = input_ids + [self.pad_id]*self.pad_len
        #torch.long tensor로 변환
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids, label

class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        raise NotImplementedError("GPTForSequenceClassification.__init__을 구현하세요.")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        raise NotImplementedError("GPTForSequenceClassification.forward를 구현하세요.")


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    raise NotImplementedError("train_epoch_sentiment를 구현하세요.")


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    raise NotImplementedError("evaluate_sentiment를 구현하세요.")
