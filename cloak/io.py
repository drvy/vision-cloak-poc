"""Image loading and saving helpers."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(path: Path, max_side: int, device: str) -> torch.Tensor:
    """Load an RGB image as a (1, 3, H, W) float tensor in [0, 1], downscaled to `max_side`."""
    img = Image.open(path).convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def save_image(x: torch.Tensor, path: Path) -> None:
    """Save a (1, 3, H, W) tensor in [0, 1] as a lossless PNG."""
    arr = (x[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path, format="PNG")
