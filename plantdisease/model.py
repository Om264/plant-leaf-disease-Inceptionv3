"""Inception v3 creation, checkpoint I/O.

Inception v3 has a quirk the other timm models don't: an auxiliary classifier
(`AuxLogits`) attached partway through the network. During training with
`aux_logits=True`, a forward pass returns a namedtuple `(logits, aux_logits)`
instead of a single tensor. Training on the aux head as well as the main head
is part of the original Inception v3 recipe and is one of the reasons it can
out-perform simpler backbones on fine-grained problems - it acts as a
regularizer and improves gradient flow into the early layers. This module
centralizes that handling so the rest of the code can mostly ignore it.
"""
from typing import Optional

import timm
import torch
from timm.data import resolve_data_config

DEFAULT_IMG_SIZE = 299  # Inception v3's native training resolution
MIN_IMG_SIZE = 224  # smaller inputs collapse to nothing by the last Inception block


def create_model(name: str = "inception_v3", num_classes: int = 1000,
                 pretrained: bool = True, dropout: float = 0.2, aux_logits: bool = True):
    kwargs = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
    if "inception_v3" in name:
        kwargs["aux_logits"] = aux_logits
    return timm.create_model(name, **kwargs)


def data_config(model, img_size: Optional[int] = None) -> dict:
    cfg = resolve_data_config({}, model=model)
    size = img_size or cfg["input_size"][1] or DEFAULT_IMG_SIZE
    if size < MIN_IMG_SIZE:
        raise ValueError(
            f"--img_size {size} is too small for Inception v3: its stride-2/pooling "
            f"stages collapse the feature map to nothing below ~{MIN_IMG_SIZE}px. "
            f"Use --img_size {DEFAULT_IMG_SIZE} (the default/native size) or at least {MIN_IMG_SIZE}."
        )
    return {
        "img_size": int(size),
        "mean": [float(x) for x in cfg["mean"]],
        "std": [float(x) for x in cfg["std"]],
    }


def has_aux_logits(model) -> bool:
    return getattr(model, "AuxLogits", None) is not None


def set_backbone_frozen(model, frozen: bool) -> None:
    """Freeze everything except the main classifier head (and AuxLogits.fc if present)."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    if has_aux_logits(model):
        head_ids |= {id(p) for p in model.AuxLogits.fc.parameters()}
    for p in model.parameters():
        p.requires_grad = (not frozen) or (id(p) in head_ids)


def forward_train(model, x):
    """Run a training forward pass. Returns (main_logits, aux_logits_or_None)."""
    out = model(x)
    if isinstance(out, (tuple, list)):
        return out[0], out[1]
    return out, None


def forward_logits(model, x):
    """Run an inference forward pass, returning only the main-head logits.

    Unlike torchvision's Inception v3, timm's returns the (logits, aux) tuple
    whenever the model was built with aux_logits=True - in both train() and
    eval() mode. Use this helper (instead of calling model(x) directly) for
    any inference/prediction code so the aux head is always discarded here.
    """
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out


def save_checkpoint(path, model, model_name, class_names, cfg, extra=None):
    torch.save({
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "data_cfg": cfg,
        "aux_logits": has_aux_logits(model),
        "extra": extra or {},
    }, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = create_model(ckpt["model_name"], len(ckpt["class_names"]), pretrained=False,
                         dropout=0.0, aux_logits=ckpt.get("aux_logits", False))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
