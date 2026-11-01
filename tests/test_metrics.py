"""Phase 1 tests: the metrics must be correct, not just plausible."""

import math
import os

import numpy as np
import pytest
import torch
from PIL import Image

from photoshield.data import synthetic_image, to_tensor
from photoshield.metrics import (
    lpips_distance,
    metric_report,
    mse,
    psnr,
    ssim,
)

RUN_LPIPS = os.environ.get("PHOTOSHIELD_RUN_LPIPS") == "1"
needs_lpips = pytest.mark.skipif(
    not RUN_LPIPS, reason="set PHOTOSHIELD_RUN_LPIPS=1 (downloads 233 MB AlexNet trunk once)"
)


def _const(value: int, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_mse_and_psnr_identical():
    img = synthetic_image(64)
    assert mse(img, img) == 0.0
    assert math.isinf(psnr(img, img))
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_psnr_known_value_one_grey_level():
    """A uniform +1/255 shift must give exactly 10*log10(255^2) = 48.13 dB."""
    a, b = _const(128), _const(129)
    assert psnr(a, b) == pytest.approx(10 * math.log10(255**2), abs=1e-6)
    assert psnr(a, b) == pytest.approx(48.1308, abs=1e-3)


def test_psnr_scales_with_error():
    base = _const(100)
    assert psnr(base, _const(102)) < psnr(base, _const(101))


def test_ssim_drops_with_noise_and_stays_in_range():
    img = synthetic_image(128)
    t = to_tensor(img)
    noisy = (t + 0.15 * torch.randn_like(t)).clamp(0, 1)
    value = ssim(img, noisy)
    assert 0.0 <= value < 0.95
    assert ssim(img, img) > value


def test_all_input_types_agree():
    """PIL, HWC uint8, CHW float and NCHW float must give the same numbers."""
    a = synthetic_image(64)
    b = to_tensor(a) + 0.01
    via_pil = psnr(a, b)
    assert psnr(np.asarray(a), b.numpy()) == pytest.approx(via_pil, rel=1e-6)
    assert psnr(b[0], to_tensor(a)[0]) == pytest.approx(via_pil, rel=1e-6)
    assert ssim(a, b) == pytest.approx(ssim(to_tensor(a), b), rel=1e-6)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        psnr(synthetic_image(64), synthetic_image(96))


def test_metric_report_contains_budget_fields():
    a = synthetic_image(64)
    b = to_tensor(a) + 0.5 / 255
    rep = metric_report(a, b, with_lpips=False)
    assert set(rep) == {"psnr_db", "ssim", "linf", "linf_255", "mean_abs_255", "size"}
    assert rep["linf"] == pytest.approx(0.5 / 255, abs=1e-7)
    assert rep["linf_255"] == pytest.approx(0.5, abs=1e-4)
    assert rep["psnr_db"] > 45


def test_lpips_rejects_tiny_images_before_downloading():
    """Guard order matters: we must fail on size, never on a network call."""
    a = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError, match="64"):
        lpips_distance(a, a)


@needs_lpips
def test_lpips_is_zero_for_identical_and_grows_with_noise():
    a = synthetic_image(128)
    b = (to_tensor(a) + 0.05 * torch.randn_like(to_tensor(a))).clamp(0, 1)
    assert lpips_distance(a, a) == pytest.approx(0.0, abs=1e-8)
    assert lpips_distance(a, b) > 0.005
