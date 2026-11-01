"""Phase 5 pre-check: does the editor keep identity on the UNPROTECTED photo?

If edit(original) already loses the face (cos < 0.5), no protection result means
anything. This sweeps editor settings on the original only and prints the
ArcFace cosine of each edit to the source, so the real evaluation can use a
setting where the baseline is valid.

    python scripts/phase5_baseline.py --image assets/samples/astronaut_face512.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photoshield.data import load_image, save_png  # noqa: E402
from photoshield.evaluate import face_embedding, load_editor, run_edit  # noqa: E402
from photoshield.models.arcface import IdentityModel, cosine  # noqa: E402

PROMPTS = {
    "img2img": "portrait photo of a woman, detailed, studio lighting",
    "ip2p": "turn her into an oil painting",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--editors", nargs="+", default=["img2img", "ip2p"])
    p.add_argument("--strengths", nargs="+", type=float, default=[0.3, 0.5])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="results/phase5/baseline")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img = load_image(args.image)
    ref = IdentityModel(device="cpu")
    src = face_embedding(ref, img)
    if src is None:
        raise SystemExit("no face in the source image")

    rows = []
    for kind in args.editors:
        pipe = load_editor(kind, "cpu")
        settings = args.strengths if kind == "img2img" else [None]
        for s in settings:
            t0 = time.perf_counter()
            edited = run_edit(pipe, kind, img, PROMPTS[kind], args.steps, 0,
                              s or 0.5, 7.5, 1.5)
            emb = face_embedding(ref, edited)
            row = {"editor": kind, "strength": s, "prompt": PROMPTS[kind],
                   "cos_to_source": cosine(src, emb) if emb is not None else None,
                   "seconds": round(time.perf_counter() - t0, 1)}
            save_png(edited, out / f"{kind}_s{s}.png")
            rows.append(row)
            print(row, flush=True)
            (out / "baseline.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        del pipe
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
