"""Training / inference loops, aware of Inception v3's auxiliary classifier."""
import copy
import time

import torch
from tqdm import tqdm

from .model import forward_logits, forward_train


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp,
                    aux_weight: float = 0.4):
    """Standard Inception v3 recipe: total_loss = main_loss + aux_weight * aux_loss."""
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, aux_logits = forward_train(model, x)
            loss = criterion(logits, y)
            if aux_logits is not None:
                loss = loss + aux_weight * criterion(aux_logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += x.size(0)
    return loss_sum / seen, correct / seen


@torch.no_grad()
def predict_loader(model, loader, device, criterion=None):
    """Returns (probabilities [N,C], labels [N], mean loss or None)."""
    model.eval()
    probs, labels, loss_sum, n = [], [], 0.0, 0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = forward_logits(model, x)  # discard the aux head, timm returns it even in eval()
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * x.size(0)
            n += x.size(0)
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(y.cpu())
    return (torch.cat(probs).numpy(), torch.cat(labels).numpy(),
            loss_sum / n if n else None)


@torch.no_grad()
def measure_latency(model, device, img_size, runs=50, warmup=10):
    model = copy.deepcopy(model).to(device).eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs):
        forward_logits(model, x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000
