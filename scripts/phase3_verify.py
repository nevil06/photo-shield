"""Phase 3 verification: is the identity loss real, or is it gaming the crop?

Run AFTER `scripts/phase3_sweep.py`. It never changes the loss; it only measures.

Step 1  preprocessing consistency
    Same 112x112 crop through the torch embedder (used by the loss) and the ONNX
    embedder (used by evaluation). If these disagree, RGB/BGR/range/model are
    mismatched. Then compare the *crude* padded-box crop against the *aligned*
    landmark crop on the same image: this isolates the crop as the mismatch.

Step 2  aligned-face measurement
    VAE round-trip of original and protected, then:
      crude          = loss's padded box + bilinear resize (torch ArcFace)
      aligned_fixed  = original landmarks applied to both (ONNX ArcFace)
      aligned_redetect = detect + align each independently (ONNX ArcFace)
    Also report the raw published image (no VAE) for context.

Step 4  random-noise control
    Same L-inf budget, uniform noise, same three measurements.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\phase3_verify.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from photoshield.config import RESULTS_DIR, SAMPLES_DIR, ProtectConfig  # noqa: E402
from photoshield.data import load_image, save_png, to_pil, to_tensor  # noqa: E402
from photoshield.models.arcface import (  # noqa: E402
    ARCFACE_TEMPLATE,
    IdentityModel,
    cosine,
    umeyama_similarity,
)
from photoshield.models.vae import load_vae  # noqa: E402
from photoshield.protect.losses import crop_resize  # noqa: E402
from photoshield.protect.pgd_engine import PGDProtector  # noqa: E402


def warp_with(img: Image.Image, M: np.ndarray, size: int = 112) -> Image.Image:
    """Apply a known 2x3 transform. Same bilinear/border policy as align_face."""
    arr = np.asarray(img.convert("RGB"))
    out = cv2.warpAffine(
        arr, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return Image.fromarray(out)


def roundtrip(vae, img: Image.Image) -> torch.Tensor:
    """Float (1,3,H,W) round-trip.

    Kept as a tensor (not uint8 PIL) so the crude-crop cosine reproduces the
    loss exactly; the aligned measurement converts to PIL where evaluation does.
    """
    with torch.no_grad():
        return vae.decode(vae.encode(to_tensor(img)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", default=str(SAMPLES_DIR / "astronaut_pd.png"))
    p.add_argument("--sweep", default=str(RESULTS_DIR / "phase3_sweep_aligned.json"))
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--out", default=str(RESULTS_DIR / "phase3_verify_aligned.json"))
    p.add_argument("--no-noise", action="store_true")
    args = p.parse_args(argv)

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        print(f"error: missing {sweep_path}; run phase3_sweep.py first", file=sys.stderr)
        return 2
    sweep = json.loads(sweep_path.read_text())
    rows = [r for r in sweep["rows"] if "protected_png" in r and "error" not in r]
    if not rows:
        print("error: no completed sweep rows", file=sys.stderr)
        return 2

    image_path = Path(args.image)
    work = PGDProtector(ProtectConfig(size=args.size, device="cpu")).working_image(
        load_image(image_path)
    )

    # Evaluation path: exact ONNX ArcFace.
    ref = IdentityModel(device="cpu", torch_embedder=False)
    src_face = ref.detector.largest(work)
    if src_face is None:
        print("error: no face in the original working image", file=sys.stderr)
        return 2
    M = umeyama_similarity(src_face["kps"], ARCFACE_TEMPLATE)

    # Optimisation path: torch ArcFace + the same padded box the loss uses.
    ident_t = IdentityModel(device="cpu", torch_embedder=True)
    box = ident_t.crop_box(work)
    if box is None:
        print("error: crop_box returned None", file=sys.stderr)
        return 2

    def torch_embed_t(x: torch.Tensor, use_box=True):
        view = crop_resize(x, box, 112) if use_box else torch.nn.functional.interpolate(
            x, size=(112, 112), mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            return ident_t.embedder(view)[0].numpy()

    def torch_embed(img: Image.Image, use_box=True):
        return torch_embed_t(to_tensor(img), use_box)

    def onnx_embed_t(x: torch.Tensor, use_box=True):
        if use_box:
            img = to_pil(crop_resize(x, box, 112))
        else:
            img = warp_with(to_pil(x), M)
        return ref.embedder.embed_aligned(img)

    def onnx_embed(img: Image.Image, use_box=True):
        return onnx_embed_t(to_tensor(img), use_box)

    # ------------------------------------------------------------ Step 1
    print("=" * 78)
    print("STEP 1 — preprocessing consistency (same image, no perturbation)")
    print("=" * 78)
    aligned = warp_with(work, M)
    emb_aligned = ref.embedder.embed_aligned(aligned)
    c_aligned = cosine(emb_aligned, torch_embed(aligned, use_box=False))
    c_crude = cosine(onnx_embed(work, use_box=True), torch_embed(work, use_box=True))
    c_views_torch = cosine(
        torch_embed(aligned, use_box=False), torch_embed(work, use_box=True)
    )
    c_views_onnx = cosine(emb_aligned, onnx_embed(work, use_box=True))

    x = to_tensor(work)
    x0, y0, x1, y1 = box
    crop_t = np.asarray(to_pil(crop_resize(x, box, 112))).astype(int)
    crop_cv = cv2.resize(
        np.asarray(work)[y0:y1, x0:x1], (112, 112), interpolation=cv2.INTER_LINEAR
    ).astype(int)
    resize_diff = float(np.abs(crop_t - crop_cv).mean())

    print(f"  torch vs ONNX on the SAME aligned crop : cos = {c_aligned:.6f}")
    print(f"  torch vs ONNX on the SAME crude crop   : cos = {c_crude:.6f}")
    print(f"  crude vs aligned crop, torch embedder  : cos = {c_views_torch:.6f}")
    print(f"  crude vs aligned crop, ONNX embedder   : cos = {c_views_onnx:.6f}")
    print(f"  torch bilinear vs cv2.resize, mean |diff| = {resize_diff:.2f}/255")
    print(f"  crude box = {box} (side {x1 - x0}x{y1 - y0}); face box = "
          f"({src_face['box'][0]:.0f},{src_face['box'][1]:.0f},"
          f"{src_face['box'][2]:.0f},{src_face['box'][3]:.0f})")
    print("  alignment = Umeyama similarity to the ArcFace 5-point template;")
    print("  this is the same affine insightface.norm_crop computes.")
    print("  RGB/BGR + range + backbone are identical if the first two rows are ~1.0.")
    print("  A low third/fourth row means the loss and the metric look at different crops.\n")

    # ------------------------------------------------------------ Step 2
    print("=" * 78)
    print("STEP 2 — VAE round-trip: crude crop vs aligned crop")
    print("=" * 78)

    vae = load_vae(device="cpu")
    rt_src = roundtrip(vae, work)
    rt_src_pil = to_pil(rt_src)
    crude_src = torch_embed_t(rt_src)
    aligned_src = warp_with(rt_src_pil, M)
    emb_aligned_src = ref.embedder.embed_aligned(aligned_src)
    face_rt_src = ref.detector.largest(rt_src_pil)
    aligned_redetect_src = ref.embedder.embed_aligned(
        warp_with(rt_src_pil, umeyama_similarity(face_rt_src["kps"], ARCFACE_TEMPLATE))
    )
    raw_aligned_src = ref.embedder.embed_aligned(warp_with(work, M))

    def measure(img: Image.Image) -> dict:
        rt = roundtrip(vae, img)
        rt_pil = to_pil(rt)
        crude = cosine(crude_src, torch_embed_t(rt))
        fixed = cosine(emb_aligned_src, ref.embedder.embed_aligned(warp_with(rt_pil, M)))
        face = ref.detector.largest(rt_pil)
        redetect = (
            cosine(
                aligned_redetect_src,
                ref.embedder.embed_aligned(
                    warp_with(rt_pil, umeyama_similarity(face["kps"], ARCFACE_TEMPLATE))
                ),
            )
            if face is not None
            else None
        )
        raw_fixed = cosine(raw_aligned_src, ref.embedder.embed_aligned(warp_with(img, M)))
        face_raw = ref.detector.largest(img)
        kp_shift = (
            float(np.linalg.norm(face_raw["kps"] - src_face["kps"], axis=1).mean())
            if face_raw is not None
            else float("nan")
        )
        return {
            "crude_torch": crude,
            "aligned_fixed": fixed,
            "aligned_redetect": redetect,
            "raw_aligned_fixed": raw_fixed,
            "kp_shift_px": kp_shift,
        }

    out_rows = []
    print("\n| w | cos_loop (hist) | crude torch | aligned fixed | aligned re-detect | raw aligned | kp shift |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        prot_path = Path(row["protected_png"])
        if not prot_path.is_absolute():
            prot_path = sweep_path.parent.parent / prot_path
        m = measure(load_image(prot_path))
        m["w_identity"] = row["w_identity"]
        m["cos_loop"] = row["cos_loop"]
        m["cos_measured_raw"] = row["cos_measured"]
        out_rows.append(m)
        loop_s = "—" if m["cos_loop"] is None else f"{m['cos_loop']:.4f}"
        redetect_s = "—" if m["aligned_redetect"] is None else f"{m['aligned_redetect']:.4f}"
        print(
            f"| {m['w_identity']:g} | {loop_s} | {m['crude_torch']:.4f} | "
            f"{m['aligned_fixed']:.4f} | {redetect_s} | "
            f"{m['raw_aligned_fixed']:.4f} | {m['kp_shift_px']:.2f} |"
        )

    # ------------------------------------------------------------ Step 4
    if not args.no_noise:
        print("\n" + "=" * 78)
        print("STEP 4 — random-noise control at the same L-inf budget")
        print("=" * 78)
        eps = float(rows[0]["config"]["eps"])
        gen = torch.Generator().manual_seed(0)
        delta = (torch.rand(to_tensor(work).shape, generator=gen) * 2 - 1) * eps
        noise = to_pil((to_tensor(work) + delta).clamp(0, 1))
        noise_path = RESULTS_DIR / "phase3_noise_control.png"
        save_png(noise, noise_path)
        m = measure(noise)
        m["w_identity"] = "noise"
        m["cos_loop"] = None
        m["cos_measured_raw"] = None
        out_rows.append(m)
        redetect_s = "—" if m["aligned_redetect"] is None else f"{m['aligned_redetect']:.4f}"
        print(
            f"| noise | — | {m['crude_torch']:.4f} | {m['aligned_fixed']:.4f} | "
            f"{redetect_s} | {m['raw_aligned_fixed']:.4f} | {m['kp_shift_px']:.2f} |"
        )
        print(f"  saved: {noise_path}")

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(
            {
                "source": str(image_path),
                "box": list(box),
                "step1": {
                    "torch_vs_onnx_aligned": c_aligned,
                    "torch_vs_onnx_crude": c_crude,
                    "crude_vs_aligned_torch": c_views_torch,
                    "crude_vs_aligned_onnx": c_views_onnx,
                    "resize_mean_abs_diff_255": resize_diff,
                },
                "rows": out_rows,
            },
            indent=2,
        )
    )
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
