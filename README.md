# PhotoShield

**Status: Phases 1–4 and 6 done; Phase 5 (real-editor evaluation) code done,
run in progress. The PGD + ArcFace identity loss works on the VAE proxy and is
measured. No claim yet that a real editor fails: that number comes from
Phase 5.**

PhotoShield is a defensive tool for protecting **your own** photos from AI image
editing (diffusion img2img / instruction editing such as "make me Ghibli" or
"swap my gender"). It computes a small, imperceptible adversarial perturbation
and saves a protected copy you can post instead of the original.

## What it does and does not do

- **Does:** target *open* diffusion editors (SD 1.5 img2img, InstructPix2Pix) and
  measure, with numbers, how much identity similarity drops after an edit.
- **Does not:** protect against closed APIs (ChatGPT, Gemini, Midjourney), against
  face-recognition/training-data scraping, or against a determined human who
  crops and re-encodes. There is no guarantee. See `docs_private/05_results_and_limits.md`
  once Phase 5 runs.
- **Ethics:** only use photos you own or have consent/licence for. This is not a
  tool for evading identity verification.

## Install

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1     Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Use

```bash
python -m pytest -q                       # 68 passed, 3 gated/skipped
python -m photoshield.cli demo            # smoke test, no photos needed
python -m photoshield.cli protect me.jpg --eps-255 4 --size 256 --steps 40 --w-identity 2
python -m photoshield.cli metrics me.jpg me_protected.png
```

`--w-identity 2` requires a detectable face and downloads `buffalo_l`
(~350 MB) on first use. It also adds ~2-8 min on CPU. Set it to `0` to keep the
Phase 2 latent-only attack (no face detection, no ArcFace).

The protected file is written at the working size (512 px long side), not at the
original resolution — see `docs_private/07_decision_log.md` D13.

Measured on a CPU-only laptop, 256 px, eps 4/255, 40 steps:

| `w_identity` | aligned VAE identity cosine | published-photo cosine | PSNR | LPIPS |
|---|---|---|---|---|
| 0 (latent only) | 0.6063 | 0.9526 | 38.02 | 0.0023 |
| 2 | **-0.1133** | 0.9604 | 40.07 | **0.0031** |
| 10 | **-0.1337** | 0.9551 | 41.21 | 0.0048 |

The aligned VAE cosine collapses while the published photo stays recognizable.
A random-noise control at the same eps leaves the aligned cosine at 0.9827.
These identity numbers are **256 px**; 512 px identity is not measured yet
(CPU cost). **This is a proxy, not proof that an edit fails.** Real SD img2img and
InstructPix2Pix results are Phase 5 and have not been run yet. The full
before/after investigation (the first loss was optimising the wrong crop) is in
`docs_private/05_results_and_limits.md` §3.3–3.4.

### Web UI (Phase 6)

```bash
python -m photoshield.app            # http://127.0.0.1:7860
python -m photoshield.app --share    # temporary public link (Colab)
```

Upload a photo, choose the budget/steps/size, press **Protect**, compare side by
side (plus a ×20 view of the perturbation), and download the protected **PNG**.
On a CPU the defaults drop to 256 px / 30 steps.

### Evaluate against real editors (Phase 5)

```bash
python -m photoshield.evaluate --editor img2img \
    --original me.jpg --protected me_protected.png \
    --prompt "portrait of the person, detailed, studio lighting" \
    --steps 20 --strength 0.5 --out-dir results/phase5/img2img

python -m photoshield.evaluate --editor ip2p \
    --original me.jpg --protected me_protected.png \
    --prompt "turn this into a Studio Ghibli style painting" \
    --out-dir results/phase5/ip2p \
    --postprocess none jpeg75 resize50 crop90
```

Each run edits the original, the protected photo and a random-noise control
(same L-inf), after each post-process (none, JPEG q=75, resize 50 %, crop 90 %),
with the same prompt and seed. It writes `results.csv`, `results.md`,
`results.json` and every edited PNG to `--out-dir`. First use downloads the fp16
editor weights (~2.6 GB each). A row only counts when the unprotected edit still
matches the source face (cosine ≥ 0.5).

LPIPS needs a one-time 233 MB trunk download:

```bash
python -m photoshield.cli fetch-lpips
```

## Layout

```
photoshield/       library (data, metrics, protect/)
tests/             pytest suite
assets/samples/    your own test photos (gitignored)
results/           CSV + markdown from evaluation (gitignored)
docs_private/      learning notes (gitignored, never committed)
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | skeleton, I/O, PSNR/SSIM/LPIPS | done |
| 2 | PGD on the SD VAE encoder + LPIPS penalty | done |
| 3 | ArcFace identity loss + aligned VAE surrogate | done on VAE proxy; real-editor check pending |
| 4 | EOT (JPEG/resize/crop/blur) | code + tests present; not measured (paused for Phase 3 verification) |
| 5 | evaluation vs open editors + CSV | code done; run in progress (results pending) |
| 6 | Gradio UI | done |
| 7 | optional DCT perturbation | optional |

## Licence / use

For personal photo protection and research/education. You are responsible for
the images you process and for complying with platform terms.
