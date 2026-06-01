# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch

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
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    #raise NotImplementedError("calc_loss_batch를 구현하세요.")

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    #target이 있으므로 model에서 loss계산 가능
    loss, logits = model(input_batch, target_batch)

    return loss

def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    #raise NotImplementedError("calc_loss_loader를 구현하세요.")
    #평가모드로 전환, dropout 유무
    model.eval()

    total_loss = 0.0
    #num_batches : 몇개의 batch를 볼지, batch_size : 하나의 배치에 문장 몇개인지
    if num_batches is None:
        #len(data_loader) : DataLoader가 만들어낼 전체 배치 개수
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    #평가중이므로 gradient 계산을 끄고, 작업이 끝나면 다시 켜짐
    with torch.no_grad():
        #enumerate : 반복문에서 원소와 함께 index도 꺼내주는 함수
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            #하나의 batch에 대한 loss계산 후 total에 저장(평균 구하기용)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()

    model.train()
    #순회했던 batch들의 loss의 평균을 반환
    return total_loss / num_batches

def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    #raise NotImplementedError("save_checkpoint를 구현하세요.")
    #모델의 학습된 파라미터, optimizer 내부 상태, 몇번째 epoch까지 했는지, 전체 배치 update가 몇번인지 저장
    checkpoint = {
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "epoch" : epoch,
        "global_step" : global_step,
    }

    torch.save(checkpoint, path)

def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    #raise NotImplementedError("load_checkpoint를 구현하세요.")
    #저장했던 checkpoint파일을 읽어옴,map_location=device : 저장 당시 gpu였더라도 현재 cpu/gpu환경에 맞게 불러옴
    checkpoint = torch.load(path, map_location=device)
    #저장했던 모델 파라미터를 현재 모델에 다시 넣음
    model.load_state_dict(checkpoint["model_state_dict"])
    #optimizer 상태 복원, 생성만 할 때는 optimizer가 필요없어서 none 조건문으로
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #어디까지 학습했는지 불러와서 반환
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    return epoch, global_step

def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    raise NotImplementedError("generate를 구현하세요.")


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
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    raise NotImplementedError("generate_and_print_sample을 구현하세요.")


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
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    raise NotImplementedError("train_model을 구현하세요.")


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
