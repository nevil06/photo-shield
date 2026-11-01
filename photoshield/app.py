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

# ---------------------------------------------------------------- look
# Cyanotype palette: Prussian-blue ink on cool white paper, inverted for dark.
# Gradio 6 takes theme/css/js at launch(), so they travel together in UI_KW.

LOGO_SVG = """
<svg class="ps-logo" viewBox="0 0 32 36" aria-hidden="true">
  <defs><clipPath id="ps-shield"><path d="M16 1 30 6v11c0 9-6 15-14 18C8 32 2 26 2 17V6z"/></clipPath></defs>
  <path d="M16 1 30 6v11c0 9-6 15-14 18C8 32 2 26 2 17V6z" fill="currentColor" opacity=".14"/>
  <g clip-path="url(#ps-shield)" fill="currentColor">
    <rect x="4" y="7" width="5" height="5" opacity=".9"/><rect x="10" y="7" width="5" height="5" opacity=".35"/>
    <rect x="16" y="7" width="5" height="5" opacity=".7"/><rect x="22" y="7" width="5" height="5" opacity=".2"/>
    <rect x="4" y="13" width="5" height="5" opacity=".4"/><rect x="10" y="13" width="5" height="5" opacity=".8"/>
    <rect x="16" y="13" width="5" height="5" opacity=".25"/><rect x="22" y="13" width="5" height="5" opacity=".95"/>
    <rect x="4" y="19" width="5" height="5" opacity=".75"/><rect x="10" y="19" width="5" height="5" opacity=".2"/>
    <rect x="16" y="19" width="5" height="5" opacity=".9"/><rect x="22" y="19" width="5" height="5" opacity=".45"/>
    <rect x="10" y="25" width="5" height="5" opacity=".6"/><rect x="16" y="25" width="5" height="5" opacity=".3"/>
  </g>
  <path d="M16 1 30 6v11c0 9-6 15-14 18C8 32 2 26 2 17V6z" fill="none" stroke="currentColor" stroke-width="1.6"/>
</svg>"""

TOGGLE_HTML = (
    '<button id="ps-theme" type="button" aria-label="Switch light or dark theme" title="Light / dark">'
    '<svg class="ps-sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/>'
    '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
    '<svg class="ps-moon" viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/></svg>'
    "</button>"
)

CSS = """
:root { --ps-accent: #2447b8; --ps-paper: #f3f5fa; --ps-muted: #5b6475; --ps-line: #d9dfea; }
.dark { --ps-accent: #8aa2ff; --ps-paper: #0b1020; --ps-muted: #8d97ad; --ps-line: #26314b; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
#ps-header { display: flex; align-items: center; gap: 14px; padding: 6px 0 18px;
             border-bottom: 1px solid var(--ps-line); margin-bottom: 4px; }
#ps-header .ps-logo { width: 38px; height: 42px; color: var(--ps-accent); flex: none; }
#ps-header .ps-word { font-family: 'Instrument Serif', Georgia, serif; font-size: 34px; line-height: 1;
                      letter-spacing: -0.01em; color: var(--body-text-color); }
#ps-header .ps-word em { color: var(--ps-accent); }
#ps-header .ps-tag { color: var(--ps-muted); font-size: 14px; margin-top: 4px; }
#ps-header .ps-spacer { flex: 1; }
.ps-chip { font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: var(--ps-muted);
           border: 1px solid var(--ps-line); border-radius: 999px; padding: 4px 10px; white-space: nowrap; }
.ps-chip b { color: var(--ps-accent); font-weight: 600; }
#ps-theme { width: 38px; height: 38px; border-radius: 10px; border: 1px solid var(--ps-line);
            background: transparent; color: var(--body-text-color); cursor: pointer; display: grid; place-items: center; }
#ps-theme:hover { border-color: var(--ps-accent); color: var(--ps-accent); }
#ps-theme:focus-visible { outline: 2px solid var(--ps-accent); outline-offset: 2px; }
#ps-theme svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; }
#ps-theme .ps-moon { display: none; } .dark #ps-theme .ps-moon { display: block; } .dark #ps-theme .ps-sun { display: none; }
#ps-note, #ps-note .prose, #ps-note p { font-size: 12.5px !important; color: var(--ps-muted) !important; }
#ps-note p { margin: 0; }
#ps-note strong { color: var(--body-text-color); }
#ps-report table { font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; }
#ps-protect { font-size: 15px; letter-spacing: 0.02em; }
@media (max-width: 640px) {
  #ps-header { flex-wrap: wrap; } #ps-header .ps-word { font-size: 28px; }
  .ps-chip { order: 5; }
}
"""

# Remember the light/dark choice across reloads; click handling is delegated
# because gr.HTML strips inline handlers.
JS = """
() => {
  const saved = localStorage.getItem('ps-theme');
  if (saved) document.body.classList.toggle('dark', saved === 'dark');
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#ps-theme')) return;
    const dark = document.body.classList.toggle('dark');
    localStorage.setItem('ps-theme', dark ? 'dark' : 'light');
  });
}
"""


def _theme():
    import gradio as gr

    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_md,
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#f3f5fa",
        body_background_fill_dark="#0b1020",
        body_text_color="#101828",
        body_text_color_dark="#e6eaf2",
        block_background_fill="#ffffff",
        block_background_fill_dark="#121a2e",
        block_border_color="#d9dfea",
        block_border_color_dark="#26314b",
        border_color_primary_dark="#26314b",
        block_label_text_color="#5b6475",
        input_background_fill_dark="#0e1528",
        slider_color="#2447b8",
        slider_color_dark="#8aa2ff",
        button_primary_background_fill="#2447b8",
        button_primary_background_fill_hover="#1c3893",
        button_primary_background_fill_dark="#4a68e8",
        button_primary_background_fill_hover_dark="#5b78f0",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
    )


def ui_kwargs() -> dict:
    """theme/css/js/head for `demo.launch(**ui_kwargs())` (Gradio 6 takes them there)."""
    head = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">'
    )
    return {"theme": _theme(), "css": CSS, "js": JS, "head": head}


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
    chip = (
        '<span class="ps-chip" title="No GPU detected: defaults are reduced. Expect 1-10 minutes.">'
        "<b>CPU</b> · 256 px · 30 steps · 1-10 min</span>"
        if cpu
        else '<span class="ps-chip"><b>GPU</b> · 512 px · 100 steps</span>'
    )
    with gr.Blocks(title="PhotoShield") as demo:
        gr.HTML(
            f'<header id="ps-header">{LOGO_SVG}<div><div class="ps-word">Photo<em>Shield</em></div>'
            '<div class="ps-tag">Add an imperceptible perturbation to your own photo before posting it.</div></div>'
            f'<div class="ps-spacer"></div>{chip}{TOGGLE_HTML}</header>'
        )
        gr.Markdown(DISCLAIMER, elem_id="ps-note")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Your photo")
                eps = gr.Slider(2, 16, value=4, step=1, label="Budget eps (grey levels /255)")
                steps = gr.Slider(10, 200, value=30 if cpu else 100, step=10, label="PGD steps")
                size = gr.Radio([256, 384, 512], value=256 if cpu else 512, label="Working size (long side, px)")
                w_id = gr.Slider(0, 10, value=0, step=0.5, label="Identity weight (needs a visible face; slow on CPU)")
                eot = gr.Slider(0, 8, value=0, step=1, label="EOT samples (robustness to JPEG/resize; slower)")
                btn = gr.Button("Protect", variant="primary", elem_id="ps-protect")
            with gr.Column():
                out_img = gr.Image(type="pil", label="Protected (download PNG below)")
                amp = gr.Image(type="pil", label="Perturbation ×20 (for visualisation only)")
                file = gr.File(label="Download protected PNG")
                report = gr.Markdown(elem_id="ps-report")
        btn.click(protect_image, [inp, eps, steps, size, w_id, eot], [out_img, amp, file, report])

    # Gradio 6 applies theme/css/js at launch(); bake them in so callers such as
    # the Colab notebook's `build_app().launch(share=True)` get the same look.
    plain_launch = demo.launch
    demo.launch = lambda *a, **kw: plain_launch(*a, **{**ui_kwargs(), **kw})
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
