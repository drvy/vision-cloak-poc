# cloak

Adversarial image cloaking: PGD + EOT over an ensemble of CLIP encoders (open_clip) so vision models misread the image. Not Nightshade/Glaze (those target generative training); this targets recognition encoders.

## Commands
- `scripts/setup_env.sh [cpu]`: create `.venv` (default torch 2.7.1 cu126; cu128+ dropped Pascal sm_61).
- `python -m cloak.cli [--cpu] [--large] [--siglip] [--max-side 512] protect IN OUT [--eps 8] [--steps 100] [--decoy "text"] [--no-eot]`
- `python -m cloak.cli evaluate ORIGINAL CLOAKED`: zero-shot CLIP labels + cosine on original vs cloaked (and with augmentation).
- `python -m cloak.cli obscure IN OUT [--mode pixelate|blur] [--block 32] [--region x,y,w,h]`: visible obfuscation.
- `scripts/deploy.sh <host>`: rsync to `<host>:~/cloak/` (excludes .venv, out*, samples).

## Gotchas
- Dev is local (CPU, no NVIDIA). Run on a remote GPU box (tested: GTX 1050 Ti 4 GB, Python 3.13, fp16 on CUDA; needs `python3-venv`). `--large` (ViT-L/14) may not fit in 4 GB with backprop.
- Output is PNG (lossless); transfer to closed VLMs is partial, so always verify with `evaluate`.

## Findings (2026-11-14)
- CLIP ensemble (B-32, B-16, L-14) at eps 8 and 16/255, 100-200 steps, EOT: embeddings fully desaligned (cosine < 0) but Gemini still described the scene correctly in a clean chat (no original in context). No transfer to frontier VLMs.
- Test hygiene: use a fresh chat with only the cloaked file; a prior copy of the original in context invalidates the test.
- 1050 Ti 4 GB: `--large` fits at 512 px (~1.1 s/it); base ensemble ~4.4 it/s.
- Also tested: + SigLIP B/16, ViT-L/14, eps 12, decoy text, stronger EOT (blur/crop/resize): Gemini still recognizes it; SigLIP barely moved (cosine 0.41). Conclusion: no transfer to frontier VLMs with imperceptible noise.
- `obscure` (pixelate/blur): block 48 makes the image unrecognizable for humans and AI alike; there is no middle ground. Use `--region` to hide only sensitive parts.
- Next idea if resumed: Nightshade-style poisoning against the SD 1.5 VAE (training protection), a different goal from recognition evasion.
