"""Phase 3 measurement sweep — w_identity ablation on the astronaut image.

Runs 4 weight values (0, 2, 10, 30) at 256 px / 40 steps / eps 4/255.
Per run:
  - PSNR, SSIM, LPIPS (image quality)
  - cos_measured  : re-detect face on the PROTECTED output, embed with ONNX
                    (measures what the attack achieved on a *new* face crop)
  - cos_loop      : history[-1]["cos_id"] — the surrogate value from the PGD
                    loop (only meaningful when w_identity > 0)
  - latent_rms    : last step latent distance

Total expected runtime on CPU: ~2.5 min/run × 4 runs ≈ 10 min.
Results saved to results/phase3_sweep.json.

Usage:
    .venv\\Scripts\\python.exe scripts\\sweep_phase3.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make sure the project root is on sys.path so photoshield imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from photoshield.config import EPS_4_255, RESULTS_DIR, ProtectConfig
from photoshield.data import load_image
from photoshield.metrics import metric_report
from photoshield.models.arcface import IdentityModel
from photoshield.protect.pgd_engine import PGDProtector

SAMPLE = ROOT / "assets" / "samples" / "astronaut_pd.png"
OUT_DIR = RESULTS_DIR / "phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W_IDENTITY_VALUES = [0, 2, 10, 30]


def measure_cos(identity_model: IdentityModel, protected: Image.Image, original_emb) -> float | None:
    """Re-detect face on the protected image and compute cosine to original."""
    try:
        prot_emb, _ = identity_model.embed_image(protected)
    except ValueError:
        return None
    from photoshield.models.arcface import cosine
    return cosine(original_emb, prot_emb)


def run_one(w_identity: float, source: Image.Image, identity_model: IdentityModel, src_emb) -> dict:
    print(f"\n{'='*60}")
    print(f"  w_identity = {w_identity}   (size=256, steps=40, eps=4/255)")
    print(f"{'='*60}")

    cfg = ProtectConfig(
        eps=EPS_4_255,
        steps=40,
        size=256,
        seed=0,
        device="cpu",
        w_latent=1.0,
        w_lpips=2.0,
        w_identity=float(w_identity),
    )
    protector = PGDProtector(cfg=cfg, verbose=True)

    t0 = time.perf_counter()
    result = protector.protect(source)
    elapsed = time.perf_counter() - t0

    protected = result.image  # PIL Image at 256 px
    working = protector.working_image(source)  # same size, for metrics

    # Image quality vs the *working* (not original) image, since protected is 256px
    quality = metric_report(working, protected, device="cpu", with_lpips=True)

    # Identity: re-detect on protected output (independent of training path)
    cos_m = measure_cos(identity_model, protected, src_emb)

    # Surrogate cos from training loop
    history = result.info.get("history", [])
    cos_loop = history[-1].get("cos_id") if history else None
    latent_rms = history[-1].get("latent_rms") if history else None

    row = {
        "w_identity": w_identity,
        "psnr_db": quality["psnr_db"],
        "ssim": quality["ssim"],
        "lpips": quality["lpips"],
        "linf_255": quality["linf_255"],
        "cos_measured": cos_m,
        "cos_loop": cos_loop,
        "latent_rms_final": latent_rms,
        "seconds": round(elapsed, 1),
    }

    print(f"\n  PSNR={row['psnr_db']:.2f}  SSIM={row['ssim']:.4f}  LPIPS={row['lpips']:.4f}")
    if cos_m is not None:
        print(f"  cos_measured={cos_m:.4f}")
    else:
        print("  cos_measured=None (no face!)")
    print(f"  cos_loop={cos_loop!r}  latent_rms={latent_rms!r}")
    print(f"  time={elapsed:.1f}s")

    # Save individual protected image
    out_img = OUT_DIR / f"astronaut_w{w_identity}.png"
    protected.save(str(out_img))
    row["output_png"] = str(out_img)
    return row


def main():
    if not SAMPLE.exists():
        print(f"ERROR: sample not found at {SAMPLE}")
        print("Run:  python -c \"from skimage.data import astronaut; from PIL import Image; Image.fromarray(astronaut()).save('assets/samples/astronaut_pd.png')\"")
        sys.exit(1)

    print(f"Loading source: {SAMPLE}")
    source = load_image(SAMPLE)

    print("Loading ONNX ArcFace for measurement (needs weights/buffalo_l/w600k_r50.onnx)...")
    try:
        identity_model = IdentityModel(device="cpu", torch_embedder=False)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("Embedding source face (reference)...")
    try:
        src_emb, src_meta = identity_model.embed_image(source)
        print(f"  Source face detected: score={src_meta['score']:.3f}  box={src_meta['box']}")
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    rows = []
    for w in W_IDENTITY_VALUES:
        row = run_one(w, source, identity_model, src_emb)
        rows.append(row)
        # Save incrementally so a crash doesn't lose all data
        (OUT_DIR / "sweep_partial.json").write_text(json.dumps(rows, indent=2))

    out_path = RESULTS_DIR / "phase3_sweep.json"
    out_path.write_text(json.dumps({"sweep": rows}, indent=2))
    print(f"\n\nDone. Results saved to {out_path}")

    # Print markdown table
    print("\n## Markdown table (paste into 05_results_and_limits.md)\n")
    print("| w_identity | PSNR | SSIM | LPIPS | cos_measured | cos_loop | latent_rms |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        cos_m = f"{r['cos_measured']:.4f}" if r['cos_measured'] is not None else "—"
        cos_l = f"{r['cos_loop']:.4f}" if r['cos_loop'] is not None else "—"
        lrms  = f"{r['latent_rms_final']:.4f}" if r['latent_rms_final'] is not None else "—"
        print(f"| {r['w_identity']} | {r['psnr_db']:.2f} | {r['ssim']:.4f} | {r['lpips']:.4f} | {cos_m} | {cos_l} | {lrms} |")


if __name__ == "__main__":
    main()
