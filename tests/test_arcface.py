"""Phase 3 tests for the parts that need no model weights.

Alignment is exact maths, so it is tested exactly. The identity loss is tested
with a 3-number "embedder" (mean colour), which is enough to prove the gradient
flows and the sign is right. The ArcFace cross-check (ONNX vs torch) is gated
because it needs both weight files.
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from photoshield.config import ProtectConfig
from photoshield.data import synthetic_image, to_pil, to_tensor
from photoshield.models.arcface import (
    ARCFACE_TEMPLATE,
    affine_theta,
    align_face,
    apply_affine,
    cosine,
    umeyama_similarity,
)
from photoshield.protect.losses import (
    CombinedLoss,
    IdentityLoss,
    align_crop,
    crop_resize,
)
from photoshield.protect.pgd_engine import PGDProtector

RUN_ARCFACE = os.environ.get("PHOTOSHIELD_RUN_ARCFACE") == "1"
needs_arcface = pytest.mark.skipif(
    not RUN_ARCFACE, reason="set PHOTOSHIELD_RUN_ARCFACE=1 (needs buffalo_l ONNX in weights/)"
)


# ---------------------------------------------------------------- alignment
def test_umeyama_recovers_a_known_similarity():
    angle, scale, shift = 0.3, 1.7, np.array([12.0, -5.0])
    R = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = ARCFACE_TEMPLATE.astype(np.float64)
    dst = src @ R.T + shift

    M = umeyama_similarity(src, dst)
    assert np.allclose(apply_affine(M, src), dst, atol=1e-6)
    assert np.allclose(M[:, :2], R, atol=1e-6)


def test_umeyama_rejects_bad_input():
    with pytest.raises(ValueError):
        umeyama_similarity(np.zeros((3, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        umeyama_similarity(np.zeros((1, 2)), np.zeros((1, 2)))


def test_align_with_template_landmarks_is_almost_identity():
    """Catches a transposed matrix, a flipped axis or an RGB/BGR mix-up."""
    arr = (np.random.default_rng(0).random((112, 112, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    out = np.asarray(align_face(img, ARCFACE_TEMPLATE.copy()))
    assert out.shape == (112, 112, 3)
    assert np.abs(out.astype(int) - arr.astype(int)).mean() < 2.0


def test_align_output_is_always_template_size():
    img = synthetic_image(256)
    kps = ARCFACE_TEMPLATE * 2  # any 5 points, placed differently
    assert align_face(img, kps, size=112).size == (112, 112)


# ---------------------------------------------------------------- cosine
def test_cosine_properties():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, b) == pytest.approx(0.0)
    assert cosine(a, 3.5 * a) == pytest.approx(1.0)  # scale-invariant
    assert cosine(a, -a) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cosine(a, np.zeros(3))


# ---------------------------------------------------------------- loss plumbing
def test_crop_resize_selects_the_right_region():
    x = torch.zeros(1, 3, 64, 64)
    x[..., 32:64, 32:64] = 1.0  # bottom-right quadrant is white
    out = crop_resize(x, (32, 32, 64, 64), size=16)
    assert out.shape == (1, 3, 16, 16)
    assert float(out.min()) == pytest.approx(1.0)  # whole crop is white
    assert float(crop_resize(x, (0, 0, 32, 32), size=16).max()) == pytest.approx(0.0)


def test_affine_theta_matches_cv2_warp_affine():
    """The differentiable crop must be the same view as evaluation's norm_crop."""
    import cv2

    img = synthetic_image(160)
    kps = np.array(
        [[50, 40], [110, 42], [80, 75], [55, 110], [105, 112]], dtype=np.float32
    )
    M = umeyama_similarity(kps, ARCFACE_TEMPLATE)
    theta = affine_theta(M, img.size, 112)

    out_t = align_crop(to_tensor(img), theta, 112)
    out_cv = cv2.warpAffine(
        np.asarray(img),
        M,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    diff = np.abs(np.asarray(to_pil(out_t)).astype(int) - out_cv.astype(int)).mean()
    assert diff < 2.0, f"torch vs cv2 aligned crop differs by {diff:.2f}/255"


def test_align_crop_passes_gradients_to_pixels():
    x = torch.rand(1, 3, 128, 128, requires_grad=True)
    theta = affine_theta(
        np.array([[0.8, 0, 10], [0, 0.8, 10]], dtype=np.float32), (128, 128), 112
    )
    align_crop(x, theta, 112).mean().backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0


def test_identity_loss_requires_exactly_one_crop_spec():
    with pytest.raises(ValueError):
        IdentityLoss(embed=MeanEmbed())
    with pytest.raises(ValueError):
        IdentityLoss(
            embed=MeanEmbed(), box=(0, 0, 10, 10), theta=np.zeros((2, 3), np.float32)
        )


class MeanEmbed(torch.nn.Module):
    """Stand-in embedder: 3-dim mean colour, L2-normalised. No weights needed."""

    def forward(self, v):
        return torch.nn.functional.normalize(v.mean(dim=(2, 3)), dim=-1)


def test_identity_loss_pushes_similarity_down_and_respects_budget():
    img = synthetic_image(128, seed=0)
    ident = IdentityLoss(embed=MeanEmbed(), box=(16, 16, 96, 96), size=112)
    cfg = ProtectConfig(eps=8 / 255, steps=8, size=128, device="cpu", w_lpips=0.0)
    prot = PGDProtector(cfg, loss_fn=CombinedLoss({"identity": (ident, 1.0)}))
    res = prot(img)

    hist = res.info["history"]
    assert hist[0]["cos_id"] == pytest.approx(1.0, abs=1e-4)  # delta=0 -> same image
    assert hist[-1]["cos_id"] < hist[0]["cos_id"]  # the optimiser did its job
    assert res.info["linf"] <= 8 / 255 + 1e-6


def test_identity_loss_with_aligned_crop_optimises_and_respects_budget():
    """Same loop, but through the differentiable landmark-aligned view."""
    img = synthetic_image(128, seed=0)
    theta = affine_theta(
        np.array([[0.8, 0, 10], [0, 0.8, 10]], dtype=np.float32), img.size, 112
    )
    ident = IdentityLoss(embed=MeanEmbed(), theta=theta, size=112)
    cfg = ProtectConfig(eps=8 / 255, steps=10, size=128, device="cpu", w_lpips=0.0)
    res = PGDProtector(cfg, loss_fn=CombinedLoss({"identity": (ident, 1.0)}))(img)

    hist = res.info["history"]
    assert hist[0]["cos_id"] == pytest.approx(1.0, abs=1e-4)
    assert hist[-1]["cos_id"] < hist[0]["cos_id"]
    assert res.info["linf"] <= 8 / 255 + 1e-6


def test_identity_loss_calls_the_surrogate_when_given_encode_decode():
    calls = {"n": 0}

    def encode(x):
        calls["n"] += 1
        return x

    def decode(z):
        calls["n"] += 1
        return z

    loss = IdentityLoss(embed=MeanEmbed(), box=(0, 0, 64, 64), encode=encode, decode=decode)
    x = torch.rand(1, 3, 128, 128)
    loss.prepare(x)  # 2 calls (encode + decode)
    loss(x, x + 0.01)  # 2 more
    assert calls["n"] == 4


@needs_arcface
def test_onnx_and_torch_arcface_agree():
    """The attack must optimise the same model we report. Cosine >= 0.999."""
    from photoshield.models.arcface import DEFAULT_REC_MODEL, ArcFaceONNX, TorchArcFace

    img = synthetic_image(112)  # no real face needed: we compare embeddings, not identity
    onnx_emb = ArcFaceONNX(DEFAULT_REC_MODEL).embed_aligned(img)
    with torch.no_grad():
        torch_emb = TorchArcFace(DEFAULT_REC_MODEL)(
            torch.nn.functional.interpolate(to_tensor(img), size=(112, 112))
        )[0].numpy()
    assert cosine(onnx_emb, torch_emb) > 0.999
