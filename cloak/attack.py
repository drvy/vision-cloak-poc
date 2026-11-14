"""PGD attack with EOT over an ensemble of encoders."""
import io
import random

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .models import Encoder


def _jpeg_roundtrip(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Apply real JPEG compression with a straight-through gradient."""
    buf = io.BytesIO()
    img = Image.fromarray((x[0].detach().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy())
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    arr = torch.from_numpy(
        __import__("numpy").asarray(Image.open(buf).convert("RGB")).copy()
    ).permute(2, 0, 1)[None].to(x.device, x.dtype) / 255
    return x + (arr - x).detach()


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random differentiable transform that mimics resizing, noise and recompression."""
    h, w = x.shape[-2:]
    scale = random.uniform(0.5, 1.0)
    x = F.interpolate(x, size=(max(64, int(h * scale)), max(64, int(w * scale))), mode="bilinear", align_corners=False)
    if random.random() < 0.5:
        k = random.choice([3, 5])
        x = F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, stride=1)
    ch, cw = int(x.shape[-2] * random.uniform(0.8, 1.0)), int(x.shape[-1] * random.uniform(0.8, 1.0))
    top, left = random.randint(0, x.shape[-2] - ch), random.randint(0, x.shape[-1] - cw)
    x = x[..., top:top + ch, left:left + cw]
    x = x + torch.randn_like(x) * random.uniform(0.0, 0.02)
    if random.random() < 0.5:
        x = _jpeg_roundtrip(x, random.randint(50, 95))
    return x.clamp(0, 1)


def cloak(
    image: torch.Tensor,
    encoders: list[Encoder],
    eps: float = 8 / 255,
    steps: int = 100,
    step_size: float | None = None,
    decoy: torch.Tensor | None = None,
    eot: bool = True,
    progress: bool = True,
) -> torch.Tensor:
    """Return an L-inf bounded perturbed copy of `image` (1, 3, H, W in [0, 1]).

    Without a decoy the loss pushes embeddings away from the original (untargeted).
    With `decoy` (one embedding per encoder) it pulls them toward the decoy (targeted).
    """
    step_size = step_size or eps / 8
    device = image.device
    with torch.no_grad():
        originals = [e.embed(image) for e in encoders]
    delta = (torch.rand_like(image) * 2 - 1) * eps * 0.1
    delta.requires_grad_(True)
    bar = tqdm(range(steps), disable=not progress, desc="cloak")
    for _ in bar:
        adv = (image + delta).clamp(0, 1)
        view = augment(adv) if eot else adv
        loss = 0.0
        for i, enc in enumerate(encoders):
            emb = enc.embed(view)
            sim = (emb * originals[i]).sum(-1).mean()
            if decoy is not None:
                loss = loss + (emb * decoy[i]).sum(-1).mean() - sim
            else:
                loss = loss - sim
        loss.backward()
        with torch.no_grad():
            delta += step_size * delta.grad.sign()
            delta.clamp_(-eps, eps)
            delta.copy_((image + delta).clamp(0, 1) - image)
            delta.grad = None
        bar.set_postfix(loss=f"{loss.item():.3f}")
    return (image + delta).detach().clamp(0, 1)
