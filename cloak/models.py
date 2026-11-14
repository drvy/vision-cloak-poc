"""Loading of the frozen CLIP-style encoder ensemble."""
from dataclasses import dataclass

import open_clip
import torch
import torch.nn.functional as F

DEFAULT_MODELS = [
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-B-16", "laion2b_s34b_b88k"),
]
LARGE_MODELS = DEFAULT_MODELS + [("ViT-L-14", "laion2b_s32b_b82k")]


@dataclass
class Encoder:
    """A frozen image encoder with its preprocessing constants."""

    name: str
    model: torch.nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    size: int
    tokenizer: object

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings for images in [0, 1], shape (B, 3, H, W)."""
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).to(next(self.model.parameters()).dtype)
        return F.normalize(self.model.encode_image(x).float(), dim=-1)

    @torch.no_grad()
    def embed_text(self, prompts: list[str], device: str) -> torch.Tensor:
        """Return L2-normalized text embeddings."""
        tokens = self.tokenizer(prompts).to(device)
        return F.normalize(self.model.encode_text(tokens).float(), dim=-1)


def load_ensemble(specs: list[tuple[str, str]], device: str, half: bool) -> list[Encoder]:
    """Load the encoders in eval mode with gradients disabled on their weights."""
    encoders = []
    for arch, pretrained in specs:
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=device
        )
        model.eval().requires_grad_(False)
        if half:
            model.half()
        norm = next(t for t in preprocess.transforms if t.__class__.__name__ == "Normalize")
        shape = (1, 3, 1, 1)
        encoders.append(
            Encoder(
                name=f"{arch}/{pretrained}",
                model=model,
                mean=torch.tensor(norm.mean, device=device).view(shape),
                std=torch.tensor(norm.std, device=device).view(shape),
                size=model.visual.image_size[0]
                if isinstance(model.visual.image_size, tuple)
                else model.visual.image_size,
                tokenizer=open_clip.get_tokenizer(arch),
            )
        )
    return encoders
