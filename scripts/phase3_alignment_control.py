"""Phase 3 control: separate the cosine numbers that get confused.

DEPRECATED for new work: `scripts/phase3_verify.py` supersedes this file and adds
Step 1 preprocessing diagnostics plus the random-noise control. Kept because it
was the first script that separated `cos_measured` from `cos_rt_fixed`.

Comparing the wrong pair makes the result look like a failure (or a success):

    cos_measured : raw original vs raw protected, re-detected
                   -> SHOULD stay high. The published photo must still look like
                      the user; a big drop here is a usability bug, not success.
    cos_fixed    : raw original vs raw protected, same landmark transform
                   -> removes detector jitter from the number above.
    cos_loop     : the surrogate the optimiser minimises (unaligned padded crop
                   of the VAE round-trip, torch ArcFace)
    cos_rt_fixed : ALIGNED VAE round-trip, ONNX ArcFace
                   -> the honest check of the surrogate. If the loss collapses
                      cos_loop but not this, it is gaming the unaligned crop.

`cos_rt_fixed` is the number that matters for Phase 3a's claim: the editor's
encoder/decoder path should see a different face while the published pixels keep
the user's identity.

Reads the sweep JSON, so run it after `phase3_sweep.py`.

    .\\.venv\\Scripts\\python.exe scripts\\phase3_alignment_control.py
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
from photoshield.data import load_image, to_pil, to_tensor  # noqa: E402
from photoshield.models.arcface import (  # noqa: E402
    ARCFACE_TEMPLATE,
    IdentityModel,
    cosine,
    umeyama_similarity,
)
from photoshield.models.vae import load_vae  # noqa: E402
from photoshield.protect.pgd_engine import PGDProtector  # noqa: E402


def warp_with(img: Image.Image, M: np.ndarray, size: int = 112) -> Image.Image:
    """Apply a known 2x3 transform (no re-detection). Same border policy as align_face."""
    arr = np.asarray(img.convert("RGB"))
    out = cv2.warpAffine(
        arr, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return Image.fromarray(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", default=str(SAMPLES_DIR / "astronaut_pd.png"))
    p.add_argument("--sweep", default=str(RESULTS_DIR / "phase3_sweep_aligned.json"))
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--out", default=str(RESULTS_DIR / "phase3_alignment_control.json"))
    p.add_argument("--no-vae", action="store_true", help="skip the round-trip column")
    args = p.parse_args(argv)

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        print(f"error: run phase3_sweep.py first, missing {sweep_path}", file=sys.stderr)
        return 2
    sweep = json.loads(sweep_path.read_text())
    rows = [r for r in sweep["rows"] if "protected_png" in r and "error" not in r]
    if not rows:
        print("error: no completed sweep rows", file=sys.stderr)
        return 2

    image_path = Path(args.image)
    protector = PGDProtector(ProtectConfig(size=args.size, device="cpu"))
    work = protector.working_image(load_image(image_path))

    ref = IdentityModel(device="cpu", torch_embedder=False)
    src_face = ref.detector.largest(work)
    if src_face is None:
        print("error: no face in the original working image", file=sys.stderr)
        return 2

    # One transform from the ORIGINAL landmarks, reused for every protected image.
    M = umeyama_similarity(src_face["kps"], ARCFACE_TEMPLATE)
    emb_src = ref.embedder.embed_aligned(warp_with(work, M))

    vae = None
    emb_rt_src = None
    if not args.no_vae:
        vae = load_vae(device="cpu")
        with torch.no_grad():
            rt_src = to_pil(vae.decode(vae.encode(to_tensor(work))))
        emb_rt_src = ref.embedder.embed_aligned(warp_with(rt_src, M))

    print(f"image={image_path.name} work={work.size[0]}x{work.size[1]}")
    print(f"source landmarks mean x,y = {src_face['kps'].mean(axis=0).round(2)}")
    print(
        "\n| w | cos_measured | cos_fixed | cos_loop | cos_rt_fixed | kp shift px | reading |"
    )
    print("|---|---|---|---|---|---|---|")

    out_rows = []
    for row in rows:
        prot_path = Path(row["protected_png"])
        if not prot_path.is_absolute():
            prot_path = sweep_path.parent.parent / prot_path
        protected = load_image(prot_path)

        emb_fixed = ref.embedder.embed_aligned(warp_with(protected, M))
        cos_fixed = cosine(emb_src, emb_fixed)

        cos_rt = None
        if vae is not None:
            with torch.no_grad():
                rt_prot = to_pil(vae.decode(vae.encode(to_tensor(protected))))
            emb_rt = ref.embedder.embed_aligned(warp_with(rt_prot, M))
            cos_rt = cosine(emb_rt_src, emb_rt)

        # How much did the detector move between the two images?
        face_p = ref.detector.largest(protected)
        kp_shift = (
            float(np.linalg.norm(face_p["kps"] - src_face["kps"], axis=1).mean())
            if face_p is not None
            else float("nan")
        )

        cos_measured = row["cos_measured"]
        cos_loop = row["cos_loop"]
        if face_p is None:
            reading = "face no longer detected"
        elif cos_rt is not None and cos_loop is not None and cos_loop < 0.5 and cos_rt > 0.9:
            reading = "surrogate gamed by unaligned crop"
        elif cos_rt is not None and cos_rt < 0.8:
            reading = "VAE path identity moved"
        elif cos_measured < 0.9 and cos_fixed > 0.95:
            reading = "detector jitter only"
        else:
            reading = "published face intact, weak VAE effect"

        out_rows.append(
            {
                "w_identity": row["w_identity"],
                "cos_measured": cos_measured,
                "cos_fixed": cos_fixed,
                "cos_loop": cos_loop,
                "cos_rt_fixed": cos_rt,
                "kp_shift_px": kp_shift,
                "reading": reading,
            }
        )
        loop_s = "—" if cos_loop is None else f"{cos_loop:.4f}"
        rt_s = "—" if cos_rt is None else f"{cos_rt:.4f}"
        print(
            f"| {row['w_identity']:g} | {cos_measured:.4f} | {cos_fixed:.4f} | "
            f"{loop_s} | {rt_s} | {kp_shift:.2f} | {reading} |"
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"source": str(image_path), "rows": out_rows}, indent=2))
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
