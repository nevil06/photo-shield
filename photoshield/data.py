"""Image I/O and the tensor convention used by the whole project.

Convention (memorise this, everything else depends on it):
    tensor : torch.float32, shape (1, 3, H, W), values in [0, 1], RGB order
    PIL    : PIL.Image in mode "RGB"  <->  PNG on disk

Two deliberate privacy choices:
1. `load_image` applies EXIF rotation then throws the EXIF away, so a phone
   photo is not silently sideways when we save it back out.
2. `save_png` always writes PNG and never carries EXIF/GPS, so the protected
   file does not leak where the photo was taken.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------- load / save
def load_image(path: str | Path) -> Image.Image:
    """Open an image as RGB, upright, with metadata stripped."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)  # phone cameras store rotation in EXIF
        im = im.convert("RGB")
        im.info = {}  # drop exif / icc_profile so it cannot be re-saved by accident
        im.load()
        return im


def save_png(img: Image.Image | torch.Tensor, path: str | Path) -> Path:
    """Write a lossless PNG. Pixel values are exact, no metadata is written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, torch.Tensor):
        img = to_pil(img)
    img.save(path, format="PNG", optimize=True)
    return path


# ---------------------------------------------------------------- geometry
def resize_long_side(img: Image.Image, size: int) -> Image.Image:
    """Scale so max(w, h) == size, keeping the aspect ratio."""
    w, h = img.size
    if max(w, h) == size:
        return img
    s = size / max(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


def round_to_multiple(n: int, m: int = 8) -> int:
    """Round *down* to a multiple of m (VAE downsamples by 8).

    Down, not nearest: padding invents pixels that were never in the photo,
    while dropping the last <=7 columns is invisible. Floor also avoids
    Python's banker's rounding (round(64.5) == 64), which surprises everyone.
    """
    return max(m, (n // m) * m)


# ---------------------------------------------------------------- conversions
def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> (1, 3, H, W) float32 in [0, 1]."""
    # copy=True: PIL's buffer is read-only and torch warns about sharing it
    arr = np.array(img, dtype=np.uint8, copy=True)
    if arr.ndim == 2:  # grayscale
        arr = np.stack([arr] * 3, axis=-1)
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    return t.div_(255.0).unsqueeze(0)


def to_pil(t: torch.Tensor) -> Image.Image:
    """(1|3, 3, H, W) float in [0, 1] -> PIL RGB, rounding to nearest uint8."""
    if t.dim() == 4:
        t = t[0]
    arr = t.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8)
    return Image.fromarray(arr.permute(1, 2, 0).numpy(), "RGB")


def linf(a: torch.Tensor, b: torch.Tensor) -> float:
    """L-inf distance = the largest single-channel change. This is our budget."""
    return float((a - b).abs().max())


# ---------------------------------------------------------------- test fixture
def synthetic_image(size: int = 512, seed: int = 0) -> Image.Image:
    """Deterministic fake photo: smooth gradient + hard edges + light noise.

    Exists so tests and the CLI demo never touch your real photos. Hard edges
    matter: PSNR/SSIM/LPIPS all behave differently on flat vs detailed areas.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    xn, yn = xx / (size - 1), yy / (size - 1)
    img = np.stack(
        [
            0.30 + 0.55 * xn,  # red ramp, left -> right
            0.30 + 0.55 * yn,  # green ramp, top -> bottom
            0.55 + 0.25 * np.sin(6.0 * xn),  # blue waves
        ],
        axis=-1,
    )
    # two "faces": a bright disc and a dark square, so edges exist
    cy, cx, r = size // 3, size // 3, size // 6
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
    img[disc] = (0.92, 0.80, 0.72)
    q = size // 8
    img[size // 2 : size // 2 + q, size // 2 : size // 2 + q] = 0.05
    img += rng.normal(0.0, 0.01, img.shape).astype(np.float32)  # sensor-ish noise
    return to_pil(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).clamp(0, 1))
