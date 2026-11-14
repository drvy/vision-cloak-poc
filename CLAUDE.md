# cloak

Adversarial image cloaking: PGD + EOT over an ensemble of CLIP encoders (open_clip) so vision models misread the image. Not Nightshade/Glaze (those target generative training); this targets recognition encoders.

## Commands
- `scripts/setup_env.sh [cpu]`: create `.venv` (default torch 2.7.1 cu126; cu128+ dropped Pascal sm_61).
- `python -m cloak.cli [--cpu] [--large] [--max-side 512] protect IN OUT [--eps 8] [--steps 100] [--decoy "text"] [--no-eot]`
- `python -m cloak.cli evaluate ORIGINAL CLOAKED`: zero-shot CLIP labels + cosine on original vs cloaked (and with augmentation).
- `scripts/deploy.sh <host>`: rsync to `<host>:~/cloak/` (excludes .venv, out*, samples).

## Gotchas
- Dev is local (CPU, no NVIDIA). Run on a remote GPU box (tested: GTX 1050 Ti 4 GB, Python 3.13, fp16 on CUDA; needs `python3-venv`). `--large` (ViT-L/14) may not fit in 4 GB with backprop.
- Output is PNG (lossless); transfer to closed VLMs is partial, so always verify with `evaluate`.
