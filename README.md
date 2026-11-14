# vision-cloak-poc

Proof of concept: can an imperceptible adversarial perturbation stop vision models from recognizing what is in an image?

It runs a PGD attack with EOT (random resize, blur, crop, noise, JPEG) against an ensemble of frozen CLIP/SigLIP image encoders (`open_clip`) and pushes the image embedding away from the original (or toward a text decoy). It also ships a plain `obscure` command (pixelate/blur) as a baseline.

This is **not** Nightshade or Glaze. Those poison *generative* model training; this targets *recognition* encoders.

## Result (negative)

- Against the attacked encoders it works: cosine similarity to the original drops to ~0 or below and zero-shot labels change, even after augmentation.
- It does **not** transfer to a frontier closed VLM: with eps 8, 12 and 16/255 and ensembles up to B/32 + B/16 + L/14 + SigLIP, the model still described the scene correctly in a clean chat.
- Pixelating strongly (block 48) hides the content, but from humans as well. There is no middle ground with these tools.

Treat it as a test bench, not as a protection tool.

## Usage

```bash
scripts/setup_env.sh          # or: scripts/setup_env.sh cpu
.venv/bin/python -m cloak.cli protect photo.png out --eps 8 --steps 100
.venv/bin/python -m cloak.cli evaluate photo.png out/photo_cloaked.png
.venv/bin/python -m cloak.cli obscure photo.png out/photo_pix.png --block 48
```

Flags: `--large` adds ViT-L/14, `--siglip` adds SigLIP B/16, `--decoy "text"` makes the attack targeted, `--cpu` forces CPU. The default torch build is cu126 because cu128+ dropped Pascal GPUs (sm_61). It fits on a 4 GB GTX 1050 Ti at 512 px.

When testing against a hosted VLM, use a fresh chat containing only the cloaked file; a copy of the original in context invalidates the test.

## License

MIT
