"""Step 3: evaluate with a held-out face-recognition backbone.

The optimisation loss and the `phase3_verify.py` table both use
`buffalo_l/w600k_r50.onnx`. A drop measured with the same model is less
convincing than a drop measured with a model that never saw the attack.

This script takes a second ONNX recognition model (e.g. `w600k_mbf` from
`buffalo_s`) and measures the same aligned VAE round-trip cosine. It reads the
sweep JSON and, if available, `phase3_verify_aligned.json` for the used-model column.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\phase3_heldout.py ^
        --rec weights\\buffalo_s\\w600k_mbf.onnx
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
    arr = np.asarray(img.convert("RGB"))
    out = cv2.warpAffine(
        arr, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return Image.fromarray(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", default=str(SAMPLES_DIR / "astronaut_pd.png"))
    p.add_argument("--sweep", default=str(RESULTS_DIR / "phase3_sweep_aligned.json"))
    p.add_argument("--verify", default=str(RESULTS_DIR / "phase3_verify_aligned.json"))
    p.add_argument("--rec", required=True, help="second ArcFace ONNX (not used in the loss)")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--out", default=str(RESULTS_DIR / "phase3_heldout.json"))
    args = p.parse_args(argv)

    rec_path = Path(args.rec)
    if not rec_path.exists():
        print(f"error: held-out model not found: {rec_path}", file=sys.stderr)
        print("Download buffalo_s (128 MB) and extract w600k_mbf.onnx first.", file=sys.stderr)
        return 2

    sweep = json.loads(Path(args.sweep).read_text())
    rows = [r for r in sweep["rows"] if "protected_png" in r and "error" not in r]
    if not rows:
        print("error: no completed sweep rows", file=sys.stderr)
        return 2

    used_by_w = {}
    verify_path = Path(args.verify)
    if verify_path.exists():
        for r in json.loads(verify_path.read_text())["rows"]:
            used_by_w[r["w_identity"]] = r.get("aligned_fixed")

    work = PGDProtector(ProtectConfig(size=args.size, device="cpu")).working_image(
        load_image(Path(args.image))
    )
    held = IdentityModel(device="cpu", rec_path=rec_path, torch_embedder=False)
    inp = held.embedder.session.get_inputs()[0]
    out = held.embedder.session.get_outputs()[0]
    print(f"held-out model: {rec_path.name}  input={inp.name}{inp.shape}  output={out.shape}")
    print("used model    : w600k_r50.onnx (the one in the identity loss)\n")

    src_face = held.detector.largest(work)
    if src_face is None:
        print("error: no face in the original working image", file=sys.stderr)
        return 2
    M = umeyama_similarity(src_face["kps"], ARCFACE_TEMPLATE)

    vae = load_vae(device="cpu")
    with torch.no_grad():
        rt_src = to_pil(vae.decode(vae.encode(to_tensor(work))))
    emb_rt_src = held.embedder.embed_aligned(warp_with(rt_src, M))
    emb_raw_src = held.embedder.embed_aligned(warp_with(work, M))
    face_rt_src = held.detector.largest(rt_src)
    M_rt_src = umeyama_similarity(face_rt_src["kps"], ARCFACE_TEMPLATE)
    emb_rt_redetect_src = held.embedder.embed_aligned(warp_with(rt_src, M_rt_src))

    print("| w | used w600k_r50 | held-out fixed | held-out re-detect | held-out raw | delta vs used |")
    print("|---|---|---|---|---|---|")

    out_rows = []
    for row in rows:
        prot_path = Path(row["protected_png"])
        if not prot_path.is_absolute():
            prot_path = sweep.parent.parent / prot_path
        protected = load_image(prot_path)
        with torch.no_grad():
            rt = to_pil(vae.decode(vae.encode(to_tensor(protected))))

        emb_rt = held.embedder.embed_aligned(warp_with(rt, M))
        cos_fixed = cosine(emb_rt_src, emb_rt)

        face_rt = held.detector.largest(rt)
        cos_redetect = None
        if face_rt is not None:
            M_rt = umeyama_similarity(face_rt["kps"], ARCFACE_TEMPLATE)
            cos_redetect = cosine(
                emb_rt_redetect_src, held.embedder.embed_aligned(warp_with(rt, M_rt))
            )

        cos_raw = cosine(emb_raw_src, held.embedder.embed_aligned(warp_with(protected, M)))
        used = used_by_w.get(row["w_identity"])
        delta = (used - cos_fixed) if used is not None else None

        out_rows.append(
            {
                "w_identity": row["w_identity"],
                "used_w600k_r50": used,
                "heldout_fixed": cos_fixed,
                "heldout_redetect": cos_redetect,
                "heldout_raw": cos_raw,
                "delta_vs_used": delta,
            }
        )
        used_s = "—" if used is None else f"{used:.4f}"
        re_s = "—" if cos_redetect is None else f"{cos_redetect:.4f}"
        d_s = "—" if delta is None else f"{delta:+.4f}"
        print(
            f"| {row['w_identity']:g} | {used_s} | {cos_fixed:.4f} | {re_s} | "
            f"{cos_raw:.4f} | {d_s} |"
        )

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps({"heldout_model": str(rec_path), "rows": out_rows}, indent=2)
    )
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
