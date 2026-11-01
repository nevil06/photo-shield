"""Phase 5: does a real open editor preserve identity after protection?

This is the only honest test of the product claim. Phase 2/3 measure proxies
(latent distance, VAE round-trip identity). Here we actually run an editor on

    original    - the untouched photo (baseline edit)
    protected   - our PGD output
    noise       - random +-eps sign noise, same budget (control: is it just noise?)

each optionally after a social-media style post-process (none, JPEG q=75,
resize 50% and back, centre crop 90%), with the same prompt, seed and strength,
and compare every edited face to the *source* face with ArcFace.

A useful result needs all of:
    1. edit(original) still looks like the source (cos >= 0.5), otherwise the
       editor is too destructive and the comparison is meaningless,
    2. edit(protected) has clearly lower cosine than edit(original),
    3. edit(noise) does NOT drop much (otherwise any noise would do),
    4. we report when no face is found at all, instead of silently skipping.

Weights are not bundled: the first run downloads the fp16 editor weights
(~2.6 GB per editor) to the Hugging Face cache.

    python -m photoshield.evaluate --editor img2img \
        --original assets/samples/astronaut_pd.png \
        --protected results/phase5/astronaut512_eps4_w2.png \
        --prompt "portrait of the person, detailed, studio lighting" \
        --out-dir results/phase5/img2img

Outputs: <out-dir>/results.csv, <out-dir>/results.md, <out-dir>/results.json
and every edited PNG.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path

import torch
from PIL import Image

from .config import RESULTS_DIR, resolve_device
from .data import load_image, save_png, to_pil, to_tensor
from .metrics import lpips_distance, psnr
from .models.arcface import IdentityModel, cosine

MODEL_IDS = {
    "img2img": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "ip2p": "timbrooks/instruct-pix2pix",
}

# Cosine >= this between edit(original) and the source = "editor keeps identity".
# 0.5 is far above the usual ArcFace verification threshold (~0.3-0.4).
BASELINE_COS = 0.5

POSTPROCESS = ("none", "jpeg75", "resize50", "crop90")


# ---------------------------------------------------------------- post-processing
def postprocess(img: Image.Image, kind: str) -> Image.Image:
    """What a platform might do to the photo *before* someone edits it.

    Real (non-differentiable) operations, not the EOT approximations: this is
    the test, so it must use the real JPEG codec. Output size == input size so
    every editor run sees the same resolution.
    """
    if kind == "none":
        return img
    if kind == "jpeg75":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "resize50":  # downscale (platform) then upscale (editor input)
        w, h = img.size
        small = img.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
        return small.resize((w, h), Image.BICUBIC)
    if kind == "crop90":  # keep the central 90% of each side, scale back
        w, h = img.size
        cw, ch = round(w * 0.9), round(h * 0.9)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BICUBIC)
    raise ValueError(f"unknown post-process {kind!r}; choose from {POSTPROCESS}")


def sign_noise(img: Image.Image, eps: float, seed: int = 0) -> Image.Image:
    """Random-noise control: every pixel moves by exactly +-eps (same L-inf and
    roughly the same energy as a PGD sign-step perturbation)."""
    x = to_tensor(img)
    gen = torch.Generator().manual_seed(seed)
    s = torch.randint(0, 2, x.shape, generator=gen).float() * 2 - 1
    return to_pil((x + eps * s).clamp(0, 1))


def measured_eps(original: Image.Image, protected: Image.Image) -> float:
    """L-inf of the protection actually applied, so the control uses the same budget."""
    return float((to_tensor(original) - to_tensor(protected)).abs().max())


# ---------------------------------------------------------------- editor
def load_editor(kind: str, device: str):
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInstructPix2PixPipeline,
    )

    # Only fp16 weights are downloaded (half the size); on CPU we upcast to fp32
    # because fp16 matmuls on CPU are slow or unsupported.
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = dict(torch_dtype=dtype, variant="fp16", safety_checker=None, requires_safety_checker=False)
    if kind == "img2img":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_IDS[kind], **kwargs)
    elif kind == "ip2p":
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(MODEL_IDS[kind], **kwargs)
    else:
        raise ValueError(f"unknown editor {kind!r}; choose img2img or ip2p")
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_edit(pipe, kind: str, image: Image.Image, prompt: str, steps: int, seed: int,
             strength: float, guidance: float, image_guidance: float) -> Image.Image:
    # CPU generator on every device: identical noise on CPU and GPU runs.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = dict(prompt=prompt, image=image, num_inference_steps=steps, generator=gen,
                  guidance_scale=guidance)
    if kind == "img2img":
        kwargs["strength"] = strength
    else:
        kwargs["image_guidance_scale"] = image_guidance
    with torch.no_grad():
        return pipe(**kwargs).images[0]


def face_embedding(model: IdentityModel, img: Image.Image):
    """Embedding, or None when no face survives (counted, never silently skipped)."""
    try:
        return model.embed_image(img)[0]
    except ValueError:
        return None


# ---------------------------------------------------------------- report
COLUMNS = ["postprocess", "input", "cos_input_to_source", "cos_edit_to_source", "face_found",
           "lpips_edit_vs_edit_original", "psnr_input_vs_original", "seconds", "output_png"]


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def markdown_table(rows: list[dict], cols: list[str] | None = None) -> str:
    cols = cols or [c for c in COLUMNS if c != "output_png"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def summarize(rows: list[dict]) -> list[dict]:
    """Per post-process: identity drop of protected vs original, and of noise vs original."""
    out = []
    for pp in dict.fromkeys(r["postprocess"] for r in rows):
        by = {r["input"]: r for r in rows if r["postprocess"] == pp}
        o, p, n = (by.get(k, {}).get("cos_edit_to_source") for k in ("original", "protected", "noise"))
        drop = (o - p) if (o is not None and p is not None) else None
        ndrop = (o - n) if (o is not None and n is not None) else None
        out.append({
            "postprocess": pp,
            "cos_edit_original": o,
            "cos_edit_protected": p,
            "cos_edit_noise": n,
            "drop_protected": drop,
            "drop_noise": ndrop,
            "baseline_valid": o is not None and o >= BASELINE_COS,
        })
    return out


# ---------------------------------------------------------------- main
def run_suite(args) -> dict:
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / "phase5" / args.editor
    out_dir.mkdir(parents=True, exist_ok=True)

    protected = load_image(args.protected)
    original = load_image(args.original)
    if original.size != protected.size:  # protector works at a resized/cropped size
        from .data import resize_long_side

        original = resize_long_side(original, max(protected.size)).crop((0, 0, *protected.size))
    if original.size != protected.size:
        raise SystemExit(f"size mismatch: original {original.size} vs protected {protected.size}")

    eps = measured_eps(original, protected)
    inputs = {"original": original, "protected": protected}
    if not args.no_noise:
        inputs["noise"] = sign_noise(original, eps, seed=args.seed)

    ref = IdentityModel(device="cpu", torch_embedder=False)
    src = face_embedding(ref, original)
    if src is None:
        raise SystemExit("no face detected in the original; cannot evaluate identity")

    print(f"editor {args.editor} ({MODEL_IDS[args.editor]}) device={device}")
    print(f"prompt '{args.prompt}' steps={args.steps} strength={args.strength} seed={args.seed}")
    print(f"size {original.size}, measured eps {eps * 255:.2f}/255", flush=True)
    t0 = time.perf_counter()
    pipe = load_editor(args.editor, device)
    print(f"editor loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    rows: list[dict] = []
    for pp in args.postprocess:
        edits: dict[str, Image.Image] = {}
        for label, img in inputs.items():
            x = postprocess(img, pp)
            t0 = time.perf_counter()
            edited = run_edit(pipe, args.editor, x, args.prompt, args.steps, args.seed,
                              args.strength, args.guidance, args.image_guidance)
            sec = time.perf_counter() - t0
            png = save_png(edited, out_dir / f"{pp}_{label}.png")
            save_png(x, out_dir / f"{pp}_{label}_input.png")
            emb_in, emb = face_embedding(ref, x), face_embedding(ref, edited)
            edits[label] = edited
            rows.append({
                "postprocess": pp,
                "input": label,
                "cos_input_to_source": cosine(src, emb_in) if emb_in is not None else None,
                "cos_edit_to_source": cosine(src, emb) if emb is not None else None,
                "face_found": emb is not None,
                "lpips_edit_vs_edit_original": (
                    0.0 if label == "original" else lpips_distance(edits["original"], edited)
                ),
                "psnr_input_vs_original": psnr(to_tensor(original), to_tensor(x)) if label != "original" or pp != "none" else None,
                "seconds": round(sec, 1),
                "output_png": str(png),
            })
            r = rows[-1]
            print(f"  [{pp:8s}] {label:9s} cos_edit={_fmt(r['cos_edit_to_source'])} "
                  f"face={_fmt(r['face_found'])} lpips_vs_orig_edit={_fmt(r['lpips_edit_vs_edit_original'])} "
                  f"{sec:.0f}s", flush=True)
        # Write after every post-process so a crash/timeout keeps partial results.
        _write(out_dir, args, eps, device, rows)
    return _write(out_dir, args, eps, device, rows)


def _write(out_dir: Path, args, eps: float, device: str, rows: list[dict]) -> dict:
    summary = summarize(rows)
    payload = {
        "editor": args.editor, "model_id": MODEL_IDS[args.editor], "prompt": args.prompt,
        "steps": args.steps, "strength": args.strength, "guidance": args.guidance,
        "image_guidance": args.image_guidance, "seed": args.seed, "device": device,
        "original": args.original, "protected": args.protected, "eps_255": eps * 255,
        "rows": rows, "summary": summary,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    md = [
        f"# Phase 5 — {args.editor}",
        "",
        f"- model: `{MODEL_IDS[args.editor]}`, device: {device}",
        f"- prompt: \"{args.prompt}\"",
        f"- steps {args.steps}, strength {args.strength}, guidance {args.guidance}, "
        f"image guidance {args.image_guidance}, seed {args.seed}",
        f"- protected: `{args.protected}` (measured L-inf {eps * 255:.2f}/255)",
        "",
        "## Summary (ArcFace cosine of the edited face to the source face)",
        "",
        markdown_table(summary, list(summary[0].keys()) if summary else []),
        "",
        "## All runs",
        "",
        markdown_table(rows),
        "",
    ]
    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--editor", choices=sorted(MODEL_IDS), required=True)
    p.add_argument("--original", required=True)
    p.add_argument("--protected", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--strength", type=float, default=0.5, help="img2img only")
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--image-guidance", type=float, default=1.5, help="ip2p only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--postprocess", nargs="+", default=list(POSTPROCESS), choices=POSTPROCESS)
    p.add_argument("--no-noise", action="store_true", help="skip the random-noise control")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="")
    args = p.parse_args(argv)

    payload = run_suite(args)
    print("\nsummary (cos of edited face to source; drop = original - protected)")
    print(markdown_table(payload["summary"], list(payload["summary"][0].keys())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
