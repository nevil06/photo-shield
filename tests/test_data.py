"""Phase 1 tests: I/O fidelity and the L-inf bookkeeping."""

import numpy as np
import pytest
import torch
from PIL import Image

from photoshield.data import (
    linf,
    load_image,
    resize_long_side,
    round_to_multiple,
    save_png,
    synthetic_image,
    to_pil,
    to_tensor,
)


def test_tensor_shape_dtype_range():
    t = to_tensor(synthetic_image(64))
    assert t.shape == (1, 3, 64, 64)
    assert t.dtype == torch.float32
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


def test_roundtrip_is_bit_exact():
    """uint8 -> float -> uint8 must not drift, or PSNR would lie to us."""
    img = synthetic_image(96, seed=3)
    back = to_pil(to_tensor(img))
    assert np.array_equal(np.asarray(img), np.asarray(back))


def test_linf_zero_for_identical():
    t = to_tensor(synthetic_image(64))
    assert linf(t, t) == 0.0
    # float32: 1/255 isn't exact and values near 1.0 have ~6e-8 spacing,
    # so a 1e-6 relative tolerance is too tight. 1e-4 still pins 4/255 vs 5/255.
    assert linf(t, t + 1 / 255) == pytest.approx(1 / 255, rel=1e-4)


def test_save_png_writes_no_metadata(tmp_path):
    """Our own leakage check: the protected file must not carry EXIF/GPS."""
    path = save_png(synthetic_image(64), tmp_path / "sub" / "out.png")
    assert path.exists()
    with Image.open(path) as im:
        assert im.format == "PNG"
        assert dict(im.getexif()) == {}


def test_load_image_converts_grayscale_to_rgb(tmp_path):
    p = tmp_path / "grey.png"
    Image.new("L", (32, 24), color=90).save(p)
    img = load_image(p)
    assert img.mode == "RGB" and img.size == (32, 24)


def test_resize_long_side_keeps_aspect():
    img = Image.new("RGB", (800, 400))
    out = resize_long_side(img, 512)
    assert out.size == (512, 256)


def test_round_to_multiple():
    assert round_to_multiple(512) == 512
    assert round_to_multiple(514) == 512
    assert round_to_multiple(519) == 512
    assert round_to_multiple(3) == 8  # never return 0


def test_synthetic_image_is_deterministic():
    assert np.array_equal(np.asarray(synthetic_image(64, seed=7)), np.asarray(synthetic_image(64, seed=7)))
    assert not np.array_equal(np.asarray(synthetic_image(64, seed=7)), np.asarray(synthetic_image(64, seed=8)))
