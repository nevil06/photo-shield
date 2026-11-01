"""Command line entry point.

    protect   add a perturbation to a photo and save a protected PNG
    metrics   compare two images (original vs protected)
    demo      end-to-end smoke test on a synthetic image (no photos needed)
    fetch-lpips  pre-download the LPIPS trunk so nothing downloads mid-run
    check-face   detect the face and save the aligned 112x112 crop
    eval      run a real editor on original / protected / noise (Phase 5)
    ui        start the Gradio web UI (Phase 6)

Run with:  python -m photoshield.cli <command> --help
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

from .config import ProtectConfig, resolve_device
from .data import load_image, linf, save_png, synthetic_image, to_tensor
from .metrics import format_report, metric_report


def _print_report(title: str, report: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(format_report(report))
    print()


# ---------------------------------------------------------------- commands
def cmd_protect(args: argparse.Namespace) -> int:
    """Protect one photo. Quality is measured against the *working-size* original,
    because that is the image the attack actually sees (see docs 05: resolution policy)."""
    from .protect.pgd_engine import PGDProtector

    device = resolve_device(args.device)
    cfg = ProtectConfig(
        eps=args.eps_255 / 255.0,
        steps=args.steps,
        size=args.size,
        seed=args.seed,
        device=device,
        w_identity=args.w_identity,
        eot_k=args.eot_k,
        latent_target=args.target,
    )
    img = load_image(args.image)
    print(f"input : {args.image}  {img.size[0]}x{img.size[1]}")
    print(f"config: eps={args.eps_255}/255 steps={args.steps} size={args.size} device={device}")
    if args.w_identity > 0:
        print(f"        w_identity={args.w_identity} (needs a detectable face)")
    if args.eot_k > 0:
        print(f"        eot_k={args.eot_k} (EOT: gradients averaged over {args.eot_k} augmented views)")

    protector = PGDProtector(cfg, verbose=not args.quiet)
    result = protector(img)

    out = Path(args.output) if args.output else Path(args.image).with_suffix("").with_name(
        Path(args.image).stem + "_protected.png"
    )
    save_png(result.image, out)

    info = result.info
    print(f"\nlatent distance : {info['loss_start']:.4f} -> {info['loss_end']:.4f}")
    if args.w_identity > 0:
        hist = info["history"]
        print(
            f"identity cosine : {hist[0]['cos_id']:.4f} -> {hist[-1]['cos_id']:.4f} "
            f"(lower = less recognisable as the same person)"
        )
    print(f"L-inf           : {info['linf_255']:.2f}/255 (budget {args.eps_255}/255)")
    print(f"working size    : {info['working_size']}, {info['seconds']}s")
    print(f"saved           : {out}")

    report = metric_report(
        protector.working_image(img), result.image, device=device, with_lpips=not args.no_lpips
    )
    _print_report("quality vs working-size original:", report)
    print("note: 'does the edit fail?' is measured in Phase 5, not here.\n")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    orig = load_image(args.original)
    prot = load_image(args.protected)
    if orig.size != prot.size:
        print(
            f"error: sizes differ ({orig.size} vs {prot.size}); resize before comparing",
            file=sys.stderr,
        )
        return 2
    report = metric_report(orig, prot, device=args.device, with_lpips=not args.no_lpips)
    _print_report(f"quality: {args.original} -> {args.protected}", report)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Smoke test: add bounded random noise, then check the metric pipeline.

    This is *not* an attack. It proves three things at once: I/O round-trips,
    the L-inf budget holds, and every metric reacts in the expected direction.
    """
    cfg = ProtectConfig(eps=args.eps, size=args.size)
    img = synthetic_image(size=args.size, seed=0)
    print(f"synthetic image: {img.size[0]}x{img.size[1]}, eps={cfg.eps_grey_levels}/255")

    clean = to_tensor(img)
    gen = torch.Generator().manual_seed(cfg.seed)
    delta = (torch.rand(clean.shape, generator=gen) * 2 - 1) * cfg.eps
    # clamp both sides: delta to the budget, pixels to the legal range
    prot = (clean + delta).clamp(0, 1)
    delta = prot - clean  # the delta that was *really* applied after clamping
    assert linf(delta, torch.zeros_like(delta)) <= cfg.eps + 1e-6, "budget violated!"

    report = metric_report(clean, prot, device=args.device, with_lpips=not args.no_lpips)
    _print_report("identity check (same image, delta=0):", metric_report(clean, clean, device=args.device, with_lpips=False))
    _print_report(f"random noise control at L-inf<={cfg.eps_grey_levels}/255:", report)
    print("expected: PSNR high (>=35 dB), SSIM close to 1, LPIPS small.")
    print("if PSNR is low or L-inf > eps, something is broken - do not proceed.")
    return 0


def cmd_fetch_lpips(args: argparse.Namespace) -> int:
    from .metrics import get_lpips, lpips_distance

    print("downloading/loading LPIPS trunk (torchvision AlexNet, ~233 MB, cached once)...")
    get_lpips(device="cpu")
    img = synthetic_image(size=128, seed=0)
    d_same = lpips_distance(img, img)
    print(f"self-test: LPIPS(identical) = {d_same:.6f}  (must be 0.0)")
    return 0 if math.isclose(d_same, 0.0, abs_tol=1e-8) else 1


def cmd_check_face(args: argparse.Namespace) -> int:
    """Is this photo usable for the identity loss, and where is the face?

    Exists because "no face detected" is the most likely failure on real photos,
    and seeing the 112x112 aligned crop is the fastest way to tell a bad crop
    from a missing detection.
    """
    from .models.arcface import IdentityModel

    img = load_image(args.image)
    model = IdentityModel(device=args.device)
    faces = model.detector.detect(img)
    print(f"image : {args.image}  {img.size[0]}x{img.size[1]}")
    print(f"faces : {len(faces)}")
    for i, f in enumerate(faces):
        x, y, w, h = f["box"]
        print(f"  [{i}] box=({x:.0f},{y:.0f},{w:.0f},{h:.0f}) score={f['score']:.3f}")
    if not faces:
        print("\nnone found. The identity loss cannot run on this photo (w_identity must be 0).")
        return 1

    emb, meta = model.embed_image(img)
    box = model.crop_box(img)
    print(f"\naligned crop: 112x112, embedding dim={emb.size}")
    print(f"loss box     : {box}")
    out = Path(args.image).with_name(Path(args.image).stem + "_face112.png")
    meta["aligned"].save(out)
    print(f"saved        : {out}")
    return 0


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="photoshield", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("protect", help="add a perturbation and save a protected PNG")
    r.add_argument("image")
    r.add_argument("-o", "--output", help="output PNG (default: <name>_protected.png)")
    r.add_argument("--eps-255", type=int, default=4, choices=range(1, 33), metavar="{1..32}",
                   help="L-inf budget in grey levels; 4 = invisible, 8 = strongest allowed")
    r.add_argument("--steps", type=int, default=50, help="PGD iterations (50-100)")
    r.add_argument("--size", type=int, default=512, help="working long side (256 for CPU)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="auto")
    r.add_argument("--no-lpips", action="store_true", help="skip LPIPS in the report")
    r.add_argument("--w-identity", type=float, default=ProtectConfig().w_identity,
                   help="weight of the ArcFace identity term (needs a face; 0 = off)")
    r.add_argument("-q", "--quiet", action="store_true", help="no per-step logging")
    r.add_argument("--eot-k", type=int, default=0, metavar="K",
                   help="EOT: average gradient over K random augmented views (0=off, 4=recommended)")
    r.add_argument("--target", choices=("none", "gray", "pattern"), default="none",
                   help="latent objective: none = push away from the photo, gray/pattern = "
                        "pull toward a target image (PhotoGuard/Mist style)")
    r.set_defaults(func=cmd_protect)

    c = sub.add_parser("check-face", help="detect the face and save the 112x112 aligned crop")
    c.add_argument("image")
    c.add_argument("--device", default="cpu")
    c.set_defaults(func=cmd_check_face)

    m = sub.add_parser("metrics", help="compare original vs protected image")
    m.add_argument("original")
    m.add_argument("protected")
    m.add_argument("--device", default="cpu", help="cpu or cuda (LPIPS runs there)")
    m.add_argument("--no-lpips", action="store_true", help="skip LPIPS (no 233 MB download)")
    m.set_defaults(func=cmd_metrics)

    d = sub.add_parser("demo", help="synthetic smoke test")
    d.add_argument("--size", type=int, default=512)
    d.add_argument("--eps", type=float, default=ProtectConfig().eps)
    d.add_argument("--device", default=resolve_device("auto"))
    d.add_argument("--no-lpips", action="store_true")
    d.set_defaults(func=cmd_demo)

    f = sub.add_parser("fetch-lpips", help="pre-download the LPIPS AlexNet trunk")
    f.set_defaults(func=cmd_fetch_lpips)

    # Listed for --help only; main() dispatches them before argparse sees them.
    sub.add_parser("eval", help="Phase 5: edit original/protected/noise with a real editor (see evaluate --help)")
    sub.add_parser("ui", help="Phase 6: start the Gradio web UI (--share for a public link)")
    return p


# Passthrough commands: their own argparse lives in the module, so the flags
# documented in evaluate.py / app.py are exactly the flags accepted here.
PASSTHROUGH = {"eval": "photoshield.evaluate", "ui": "photoshield.app"}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in PASSTHROUGH:
        import importlib

        return int(importlib.import_module(PASSTHROUGH[argv[0]]).main(argv[1:]) or 0)
    args = build_parser().parse_args(argv)
    torch.manual_seed(0)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
