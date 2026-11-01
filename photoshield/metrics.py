"""Quality metrics: how much did we change the photo?

Three numbers, three different questions (all "higher = more visible change"
except the last two, which we want *small*):

    PSNR (dB)  : average pixel error, log scale. Crude but standard.
    SSIM  [0,1]: local structure/luminance/contrast similarity. 1 = identical.
    LPIPS [0,~]: distance in AlexNet feature space. Best proxy for "do humans
                 notice?", which is why we report it next to PSNR.

All functions accept a PIL image, an HWC numpy array, a CHW tensor, or an
NCHW tensor, and internally normalise to (1, 3, H, W) float32 in [0, 1].
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from .data import to_tensor

# ---------------------------------------------------------------- conversion
def _to_nchw(x) -> torch.Tensor:
    """Best-effort normalisation of any supported input to (1, 3, H, W) [0,1]."""
    if isinstance(x, Image.Image):
        return to_tensor(x)

    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.integer):  # uint8 arrays arrive as 0..255
            x = x.astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        if x.ndim == 2:  # H, W
            x = x.unsqueeze(-1)
        if x.ndim == 3:  # H, W, C  (ambiguous with C,H,W; we assume HWC for numpy)
            x = x.permute(2, 0, 1)

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"unsupported image type: {type(x)!r}")

    x = x.detach().float()
    if x.ndim == 2:
        x = x.unsqueeze(0)  # H,W -> 1,H,W
    if x.ndim == 3:
        x = x.unsqueeze(0)  # C,H,W -> 1,C,H,W
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)  # grayscale -> RGB
    if x.shape[1] == 4:
        x = x[:, :3]  # drop alpha
    if x.shape[1] != 3:
        raise ValueError(f"expected 1/3/4 channels, got shape {tuple(x.shape)}")
    return x.clamp(0, 1) if float(x.max()) > 1.0 + 1e-6 else x


def _check_same(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")


# ---------------------------------------------------------------- PSNR
def mse(a, b) -> float:
    """Mean squared error over every pixel and channel."""
    a, b = _to_nchw(a), _to_nchw(b)
    _check_same(a, b)
    return float(((a - b) ** 2).mean())


def psnr(a, b, data_range: float = 1.0) -> float:
    """10*log10(peak^2 / MSE). Identical images -> +inf.

    Rule of thumb: >40 dB is hard to see, ~35 dB is "looks fine", <30 dB is
    visibly lossy. We target >=35 dB at eps=4/255.
    """
    m = mse(a, b)
    if m == 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range**2) / m)


# ---------------------------------------------------------------- SSIM
def ssim(a, b, data_range: float = 1.0) -> float:
    """Structural similarity (scikit-image, 11x11 Gaussian window, 7x7 skip).

    We push pixels through numpy because scikit-image is numpy-native; the
    float64 cast avoids a subtle loss of precision in the covariance terms.
    """
    from skimage.metrics import structural_similarity

    A = _to_nchw(a)[0].permute(1, 2, 0).numpy().astype(np.float64)
    B = _to_nchw(b)[0].permute(1, 2, 0).numpy().astype(np.float64)
    if A.shape[0] < 11 or A.shape[1] < 11:
        raise ValueError("SSIM needs at least 11x11 pixels")
    # channel_axis=-1: compute per channel, then average -> standard RGB SSIM
    return float(structural_similarity(A, B, data_range=data_range, channel_axis=-1))


# ---------------------------------------------------------------- LPIPS
_LPIPS_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


def get_lpips(net: str = "alex", device: str = "cpu") -> torch.nn.Module:
    """Load (and cache) the LPIPS model. Frozen, eval mode, no gradients.

    Note: the lpips package ships its *linear* weights, but the AlexNet trunk
    comes from torchvision, so the very first call downloads ~233 MB into the
    torch hub cache. That is why this is lazy and why the CLI has
    `fetch-lpips` - we never download anything silently.
    """
    key = (net, device)
    if key in _LPIPS_CACHE:
        return _LPIPS_CACHE[key]

    # `--device auto` reaches here from the CLI; resolve before torch sees it.
    if str(device) == "auto":
        from .config import resolve_device

        device = resolve_device("auto")

    from lpips import LPIPS  # imported here so `metrics` works without torchvision

    try:
        model = LPIPS(net=net, verbose=False).to(device).eval()
    except Exception as exc:  # usually a URLError from torch.hub
        raise RuntimeError(
            f"LPIPS failed to load (device={device!r}). If the trunk is missing "
            f"(~233 MB), run once:  python -m photoshield.cli fetch-lpips\n"
            f"underlying error: {exc}"
        ) from exc

    for p in model.parameters():
        p.requires_grad_(False)  # we only ever *measure* with LPIPS
    _LPIPS_CACHE[key] = model
    return model


def lpips_distance(a, b, net: str = "alex", device: str = "cpu") -> float:
    """Perceptual distance. 0 = identical. ~0.01 is already hard to see."""
    A, B = _to_nchw(a), _to_nchw(b)
    _check_same(A, B)
    h, w = A.shape[-2:]
    if min(h, w) < 64:
        raise ValueError(f"LPIPS needs >=64px per side, got {w}x{h}")
    model = get_lpips(net=net, device=device)
    with torch.no_grad():
        # LPIPS is defined on [-1, 1] inputs, ours are [0, 1].
        d = model(A.to(device) * 2 - 1, B.to(device) * 2 - 1)
    return float(d.mean())


# ---------------------------------------------------------------- report
def metric_report(orig, prot, device: str = "cpu", with_lpips: bool = True) -> dict:
    """All quality numbers for one (original, protected) pair.

    `linf` is included because it is the *constraint* we promised, not a
    quality proxy: if linf > eps, the attack silently broke its own contract.
    """
    A, B = _to_nchw(orig), _to_nchw(prot)
    _check_same(A, B)
    diff = (A - B).abs()
    report = {
        "psnr_db": psnr(A, B),
        "ssim": ssim(A, B),
        "linf": float(diff.max()),
        "linf_255": float(diff.max()) * 255.0,
        "mean_abs_255": float(diff.mean()) * 255.0,
        "size": f"{A.shape[-1]}x{A.shape[-2]}",
    }
    if with_lpips:
        report["lpips"] = lpips_distance(A, B, device=device)
    return report


def format_report(report: dict) -> str:
    """Pretty multi-line summary for the CLI."""
    lines = []
    for key, val in report.items():
        if key == "psnr_db":
            lines.append(f"  PSNR      : {'inf' if math.isinf(val) else f'{val:6.2f}'} dB")
        elif key == "ssim":
            lines.append(f"  SSIM      : {val:6.4f}  (1.0 = identical)")
        elif key == "lpips":
            lines.append(f"  LPIPS     : {val:6.4f}  (0.0 = identical)")
        elif key == "linf":
            lines.append(f"  L-inf     : {val:6.4f}  = {report['linf_255']:.2f}/255")
        elif key in ("linf_255", "size"):
            continue
        elif key == "mean_abs_255":
            lines.append(f"  mean |d|  : {val:6.2f}/255")
        else:
            lines.append(f"  {key:10s}: {val}")
    return "\n".join(lines)
