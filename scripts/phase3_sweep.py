"""Phase 3 measurement: sweep `w_identity` on one licensed face.

Why a committed script instead of a scratch REPL command: the first attempt at
this sweep was interrupted and produced *zero* rows. A script writes each row to
JSON immediately after the run, so an interruption costs one weight, not the
whole experiment.

What is measured per weight:
    quality      : PSNR / SSIM / LPIPS vs the working-size original
    cos_measured : ArcFace-ONNX cosine between the *re-detected* face in the
                   original and in the protected image (independent check)
    cos_loop     : last per-step cosine from the torch surrogate (same box,
                   VAE round-trip) - what the optimiser actually saw
    linf         : effective budget after clamping

`cos_measured` deliberately re-detects the face. The training path reuses one
box, so measuring with that same box would be grading its own homework.

Run (CPU, ~2.5 min per weight at 256 px):
    .\\.venv\\Scripts\\python.exe scripts\\phase3_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make `import photoshield` work when this file is run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from photoshield.config import RESULTS_DIR, SAMPLES_DIR, ProtectConfig  # noqa: E402
from photoshield.data import load_image, save_png  # noqa: E402
from photoshield.metrics import metric_report  # noqa: E402
from photoshield.models.arcface import IdentityModel, cosine  # noqa: E402
from photoshield.protect.pgd_engine import PGDProtector  # noqa: E402


def embed_onnx(model: IdentityModel, img) -> object:
    """Face -> 512-d ONNX embedding. Raises if the face is no longer detectable."""
    emb, _ = model.embed_image(img)
    return emb


def run_weight(
    weight: float,
    image_path: Path,
    size: int,
    steps: int,
    eps_255: int,
    out_dir: Path,
    tag: str,
    ref: IdentityModel,
) -> dict:
    torch.manual_seed(0)
    cfg = ProtectConfig(
        eps=eps_255 / 255.0,
        steps=steps,
        size=size,
        seed=0,
        device="cpu",
        w_identity=weight,
    )
    protector = PGDProtector(cfg)
    original = load_image(image_path)
    work = protector.working_image(original)  # the image the attack actually sees

    t0 = time.perf_counter()
    result = protector(original)
    seconds = time.perf_counter() - t0

    quality = metric_report(work, result.image, device="cpu", with_lpips=True)

    # Independent identity check: detect + align + embed again on both images.
    cos_measured = cosine(embed_onnx(ref, work), embed_onnx(ref, result.image))

    history = result.info["history"]
    row = {
        "w_identity": weight,
        "cos_measured": cos_measured,
        "cos_loop": history[-1].get("cos_id") if history else None,
        "latent_rms": history[-1].get("latent_rms") if history else None,
        "psnr_db": quality["psnr_db"],
        "ssim": quality["ssim"],
        "lpips": quality["lpips"],
        "linf_255": quality["linf_255"],
        "mean_abs_255": quality["mean_abs_255"],
        "seconds": round(seconds, 2),
        "protector_seconds": result.info["seconds"],
        "working_size": result.info["working_size"],
        "config": cfg.as_dict(),
    }

    # Tag by output file so a second run cannot silently overwrite the evidence
    # of the first one (this happened: the aligned run clobbered the broken run's
    # PNGs). Default --out keeps the historical names.
    out_png = out_dir / f"{tag}_astronaut_w{weight:g}.png"
    save_png(result.image, out_png)
    row["protected_png"] = str(out_png.relative_to(RESULTS_DIR.parent))
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", default=str(SAMPLES_DIR / "astronaut_pd.png"))
    p.add_argument("--weights", type=float, nargs="+", default=[0.0, 2.0, 10.0, 30.0])
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--eps-255", type=int, default=4)
    p.add_argument("--out", default=str(RESULTS_DIR / "phase3_sweep_aligned.json"))
    p.add_argument("--smoke", action="store_true", help="2 steps, one weight: wiring check")
    args = p.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 2

    weights = [2.0] if args.smoke else args.weights
    steps = 2 if args.smoke else args.steps
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref = IdentityModel(device="cpu", torch_embedder=False)
    rows: list[dict] = []
    meta = {
        "image": str(image_path),
        "image_size": list(load_image(image_path).size),
        "size": args.size,
        "steps": steps,
        "eps_255": args.eps_255,
        "torch": torch.__version__,
        "device": "cpu",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tag = out_path.stem.replace("phase3_sweep", "phase3").strip("_") or "phase3"
    print(f"image={image_path.name} size={args.size} steps={steps} eps={args.eps_255}/255")
    print(f"weights={weights}  out={out_path}  png_tag={tag}\n")

    for weight in weights:
        print(f"--- w_identity={weight:g} ---", flush=True)
        # Write after every weight: interruption must not lose completed rows.
        try:
            row = run_weight(
                weight, image_path, args.size, steps, args.eps_255, RESULTS_DIR, tag, ref
            )
        except Exception as exc:
            print(f"FAILED w={weight:g}: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"w_identity": weight, "error": f"{type(exc).__name__}: {exc}"})
            out_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
            raise
        rows.append(row)
        out_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
        print(
            f"    cos_measured={row['cos_measured']:.4f} "
            f"cos_loop={row['cos_loop']} "
            f"PSNR={row['psnr_db']:.2f} LPIPS={row['lpips']:.4f} "
            f"({row['seconds']:.0f}s)",
            flush=True,
        )

    print("\n| w_identity | cos_measured | cos_loop | PSNR | SSIM | LPIPS | latent | time |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['w_identity']:g} | ERROR: {r['error']} | | | | | | |")
            continue
        loop = "—" if r["cos_loop"] is None else f"{r['cos_loop']:.4f}"
        print(
            f"| {r['w_identity']:g} | {r['cos_measured']:.4f} | {loop} | "
            f"{r['psnr_db']:.2f} | {r['ssim']:.4f} | {r['lpips']:.4f} | "
            f"{r['latent_rms']:.4f} | {r['seconds']:.0f}s |"
        )
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
