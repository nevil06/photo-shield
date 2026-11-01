"""Phase 6: Gradio UI - upload, protect, compare side by side, download PNG.

    python -m photoshield.app            # opens http://127.0.0.1:7860
    python -m photoshield.app --share    # public link (Colab)

The UI is a thin shell over `PGDProtector`: every number it shows comes from
the same code path as the CLI, so a screenshot of the UI and a CLI log agree.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from PIL import Image

from .config import ProtectConfig, resolve_device
from .data import save_png
from .metrics import metric_report

DISCLAIMER = (
    "Only protect photos you own or have consent for. Protection is **not "
    "guaranteed**: it is measured against open models (Stable Diffusion VAE / "
    "img2img / InstructPix2Pix). Closed services (ChatGPT, Gemini) are not "
    "targeted directly, and heavy re-compression can weaken the effect."
)


def protect_image(
    image: Image.Image | None,
    eps_255: int,
    steps: int,
    size: int,
    w_identity: float,
    eot_k: int,
):
    """Gradio callback. Returns (protected, amplified perturbation, PNG path, report)."""
    if image is None:
        raise ValueError("upload a photo first")
    from .protect.pgd_engine import PGDProtector

    device = resolve_device("auto")
    cfg = ProtectConfig(
        eps=int(eps_255) / 255.0,
        steps=int(steps),
        size=int(size),
        device=device,
        w_identity=float(w_identity),
        eot_k=int(eot_k),
    )
    protector = PGDProtector(cfg)
    image = image.convert("RGB")
    try:
        result = protector(image)
    except ValueError as e:  # e.g. w_identity > 0 but no face found: show it in the UI
        import gradio as gr

        raise gr.Error(str(e)) from e

    work = protector.working_image(image)
    report = metric_report(work, result.image, device=device, with_lpips=True)

    # Perturbation x20 around grey: makes the invisible noise visible for the demo.
    from .data import to_pil

    amp = to_pil((0.5 + result.delta * 20).clamp(0, 1))

    out = Path(tempfile.mkdtemp(prefix="photoshield_")) / "protected.png"
    save_png(result.image, out)  # PNG: lossless, so the perturbation survives saving

    info = result.info
    lines = [
        "| metric | value |",
        "|---|---|",
        f"| device | {info['device']} |",
        f"| working size | {info['working_size']} |",
        f"| time | {info['seconds']} s |",
        f"| L-inf | {info['linf_255']:.2f}/255 (budget {int(eps_255)}/255) |",
        f"| latent distance | {info['loss_start']:.4f} → {info['loss_end']:.4f} |",
        f"| PSNR | {report['psnr_db']:.2f} dB |",
        f"| SSIM | {report['ssim']:.4f} |",
        f"| LPIPS | {report['lpips']:.4f} |",
    ]
    hist = info["history"]
    if w_identity > 0 and hist and "cos_id" in hist[0]:
        lines.append(f"| identity cos (VAE proxy) | {hist[0]['cos_id']:.4f} → {hist[-1]['cos_id']:.4f} |")
    lines.append("")
    lines.append(
        "Higher latent distance / lower identity cosine = stronger proxy effect. "
        "This is **not** proof that a given editor fails; see Phase 5 results."
    )
    return result.image, amp, str(out), "\n".join(lines)


def build_app():
    import gradio as gr

    cpu = resolve_device("auto") == "cpu"
    with gr.Blocks(title="PhotoShield") as demo:
        gr.Markdown("# PhotoShield\nAdd an imperceptible perturbation to your own photo before posting it.")
        gr.Markdown(DISCLAIMER)
        if cpu:
            gr.Markdown("_No GPU detected: defaults are reduced (256 px, 30 steps). Expect 1-10 minutes._")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Your photo")
                eps = gr.Slider(2, 16, value=4, step=1, label="Budget eps (grey levels /255)")
                steps = gr.Slider(10, 200, value=30 if cpu else 100, step=10, label="PGD steps")
                size = gr.Radio([256, 384, 512], value=256 if cpu else 512, label="Working size (long side, px)")
                w_id = gr.Slider(0, 10, value=0, step=0.5, label="Identity weight (needs a visible face; slow on CPU)")
                eot = gr.Slider(0, 8, value=0, step=1, label="EOT samples (robustness to JPEG/resize; slower)")
                btn = gr.Button("Protect", variant="primary")
            with gr.Column():
                out_img = gr.Image(type="pil", label="Protected (download PNG below)")
                amp = gr.Image(type="pil", label="Perturbation ×20 (for visualisation only)")
                file = gr.File(label="Download protected PNG")
                report = gr.Markdown()
        btn.click(protect_image, [inp, eps, steps, size, w_id, eot], [out_img, amp, file, report])
    return demo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PhotoShield web UI")
    p.add_argument("--share", action="store_true", help="public gradio link (Colab)")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args(argv)
    build_app().launch(share=args.share, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
