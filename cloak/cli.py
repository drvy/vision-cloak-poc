"""Command line interface: `python -m cloak.cli protect|evaluate`."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from .attack import augment, cloak
from .io import load_image, save_image
from .models import DEFAULT_MODELS, LARGE_MODELS, SIGLIP_MODELS, load_ensemble

LABELS = ["a photo of a person", "a photo of a dog", "a photo of a cat", "a photo of a car",
          "a photo of a building", "a photo of food", "a photo of a landscape",
          "a screenshot of a document", "a painting", "a photo of an animal"]


def _setup(args):
    """Resolve device and load the encoder ensemble."""
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    specs = list(LARGE_MODELS if args.large else DEFAULT_MODELS)
    if args.siglip:
        specs += SIGLIP_MODELS
    return device, load_ensemble(specs, device, half=device == "cuda")


def protect(args) -> None:
    """Cloak every image in the input path and write PNGs to the output dir."""
    device, encoders = _setup(args)
    src = Path(args.input)
    files = sorted(src.glob("*")) if src.is_dir() else [src]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        x = load_image(f, args.max_side, device)
        decoy = None
        if args.decoy:
            decoy = [e.embed_text([args.decoy], device) for e in encoders]
        y = cloak(x, encoders, eps=args.eps / 255,
                  steps=args.steps, decoy=decoy, eot=not args.no_eot)
        save_image(y.float(), out / f"{f.stem}_cloaked.png")
        print(f"{f.name}: Linf={float((y - x).abs().max()) * 255:.1f}/255")


@torch.no_grad()
def evaluate(args) -> None:
    """Compare zero-shot CLIP predictions on the original vs the cloaked image."""
    device, encoders = _setup(args)
    orig = load_image(Path(args.original), args.max_side, device)
    adv = load_image(Path(args.cloaked), args.max_side, device)
    if adv.shape != orig.shape:
        adv = F.interpolate(adv, size=orig.shape[-2:], mode="bilinear", align_corners=False)
    for e in encoders:
        text = e.embed_text(LABELS, device)
        for name, img in (("original", orig), ("cloaked", adv), ("cloaked+aug", augment(adv))):
            probs = (100 * e.embed(img) @ text.T).softmax(-1)[0]
            top = probs.argmax().item()
            print(f"{e.name:34s} {name:12s} -> {LABELS[top]} ({probs[top]:.2f})")
        cos = (e.embed(orig) * e.embed(adv)).sum().item()
        print(f"{e.name:34s} cosine(original, cloaked) = {cos:.3f}")


def obscure(args) -> None:
    """Pixelate or blur an image (or a region) so its content is no longer recognizable."""
    img = Image.open(args.input).convert("RGB")
    box = tuple(int(v) for v in args.region.split(",")) if args.region else (0, 0, *img.size)
    x, y, w, h = box
    part = img.crop((x, y, x + w, y + h))
    if args.mode == "pixelate":
        small = part.resize((max(1, w // args.block), max(1, h // args.block)), Image.BOX)
        part = small.resize((w, h), Image.NEAREST)
    else:
        part = part.filter(ImageFilter.GaussianBlur(args.block))
    img.paste(part, (x, y))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output, format="PNG")


def main() -> None:
    """Parse arguments and dispatch."""
    p = argparse.ArgumentParser(prog="cloak")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--large", action="store_true", help="add ViT-L/14 to the ensemble")
    p.add_argument("--siglip", action="store_true", help="add SigLIP ViT-B/16 to the ensemble")
    p.add_argument("--max-side", type=int, default=512)
    sub = p.add_subparsers(required=True)
    a = sub.add_parser("protect")
    a.add_argument("input")
    a.add_argument("output")
    a.add_argument("--eps", type=float, default=8, help="L-inf budget out of 255")
    a.add_argument("--steps", type=int, default=100)
    a.add_argument("--decoy", help="text of a decoy concept (targeted attack)")
    a.add_argument("--no-eot", action="store_true")
    a.set_defaults(fn=protect)
    b = sub.add_parser("evaluate")
    b.add_argument("original")
    b.add_argument("cloaked")
    b.set_defaults(fn=evaluate)
    c = sub.add_parser("obscure")
    c.add_argument("input")
    c.add_argument("output")
    c.add_argument("--mode", choices=["pixelate", "blur"], default="pixelate")
    c.add_argument("--block", type=int, default=32, help="pixel block size or blur radius")
    c.add_argument("--region", help="x,y,w,h (default: whole image)")
    c.set_defaults(fn=obscure)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
