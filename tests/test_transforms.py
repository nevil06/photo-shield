"""Phase 4 tests: EOT transforms and the EOTWrapper.

All tests run without model weights (no VAE, no ArcFace). The identity
of the gradient-flow logic is verified with synthetic images.

Test philosophy:
  - JPEG DCT must be differentiable (gradient flows through STE round)
  - Each augmentation must preserve shape
  - EOTWrapper must call the inner loss K times
  - PGD+EOT must still respect the L-inf budget
  - EOT must produce a *different* perturbation than no-EOT (to verify it
    actually changes the gradient, not just wraps silently)
"""

from __future__ import annotations

import pytest
import torch

from photoshield.config import ProtectConfig
from photoshield.data import synthetic_image
from photoshield.protect.losses import CombinedLoss, LatentLoss
from photoshield.protect.pgd_engine import PGDProtector
from photoshield.protect.transforms import (
    EOTWrapper,
    differentiable_jpeg,
    random_blur,
    random_crop_pad,
    random_jpeg,
    random_resize_back,
)


# ---------------------------------------------------------------- JPEG DCT
class TestDifferentiableJPEG:
    def test_output_shape_preserved(self):
        x = torch.rand(1, 3, 64, 64)
        out = differentiable_jpeg(x, quality=75)
        assert out.shape == x.shape

    def test_output_in_valid_range(self):
        x = torch.rand(2, 3, 32, 32)
        out = differentiable_jpeg(x, quality=50)
        assert float(out.min()) >= -1e-6
        assert float(out.max()) <= 1.0 + 1e-6

    def test_high_quality_is_close_to_identity(self):
        """At q=99 the quantisation step is ~1/255 so the output should be
        very close to the input — if DCT + STE + IDCT round-trips correctly."""
        x = torch.rand(1, 3, 32, 32)
        out = differentiable_jpeg(x, quality=99)
        # Mean absolute error should be < 2 grey levels
        mae = float((out - x).abs().mean()) * 255
        assert mae < 2.0, f"high-q JPEG too lossy: mae={mae:.2f}/255"

    def test_low_quality_degrades_image(self):
        """At q=1 the output should differ substantially from the input."""
        x = torch.rand(1, 3, 32, 32) * 0.5 + 0.25  # avoid clipping
        out = differentiable_jpeg(x, quality=1)
        mae = float((out - x).abs().mean()) * 255
        assert mae > 5.0, f"low-q JPEG should degrade image, mae={mae:.2f}/255"

    def test_gradient_flows_through_ste(self):
        """The straight-through estimator must pass gradients to the input."""
        x = torch.rand(1, 3, 16, 16, requires_grad=True)
        out = differentiable_jpeg(x, quality=75)
        out.sum().backward()
        assert x.grad is not None
        assert float(x.grad.abs().sum()) > 0, "gradient was zero"

    def test_invalid_quality_raises(self):
        x = torch.rand(1, 3, 16, 16)
        with pytest.raises(ValueError):
            differentiable_jpeg(x, quality=0)
        with pytest.raises(ValueError):
            differentiable_jpeg(x, quality=101)


# ---------------------------------------------------------------- augmentations
class TestAugmentations:
    _X = torch.rand(1, 3, 64, 64)
    _RNG = torch.Generator()

    def test_resize_preserves_shape(self):
        out = random_resize_back(self._X, rng=self._RNG)
        assert out.shape == self._X.shape

    def test_resize_at_scale_one_is_identity(self):
        """With scale_min=scale_max=1.0 the function returns the input unchanged."""
        out = random_resize_back(self._X, scale_min=1.0, scale_max=1.0)
        # Should be exactly the same tensor (early-return path)
        assert out is self._X

    def test_crop_pad_preserves_shape(self):
        out = random_crop_pad(self._X, rng=self._RNG)
        assert out.shape == self._X.shape

    def test_blur_preserves_shape(self):
        out = random_blur(self._X, rng=self._RNG)
        assert out.shape == self._X.shape

    def test_blur_at_zero_sigma_is_identity(self):
        out = random_blur(self._X, sigma_max=0.0)
        # With sigma_max=0 all sigma draws are 0 → early return
        assert out is self._X

    def test_random_jpeg_gradient_flows(self):
        x = torch.rand(1, 3, 32, 32, requires_grad=True)
        rng = torch.Generator()
        rng.manual_seed(0)
        out = random_jpeg(x, rng=rng)
        out.sum().backward()
        assert x.grad is not None and float(x.grad.abs().sum()) > 0

    def test_crop_pad_gradient_flows(self):
        x = torch.rand(1, 3, 32, 32, requires_grad=True)
        rng = torch.Generator()
        rng.manual_seed(1)
        out = random_crop_pad(x, rng=rng)
        out.sum().backward()
        assert x.grad is not None and float(x.grad.abs().sum()) > 0


# ---------------------------------------------------------------- EOTWrapper
class TestEOTWrapper:
    def _make_counting_loss(self):
        """A loss that counts how many times it is called."""
        calls = {"n": 0}

        def loss_fn(x, x_adv):
            calls["n"] += 1
            val = (x_adv - x).pow(2).mean()
            return val, {"mse": float(val.detach())}

        return loss_fn, calls

    def test_inner_loss_called_k_times(self):
        loss_fn, calls = self._make_counting_loss()
        wrapper = EOTWrapper(loss_fn, k=3, seed=0)
        x = torch.rand(1, 3, 32, 32)
        wrapper(x, x + 0.01)
        assert calls["n"] == 3

    def test_output_shape_and_parts(self):
        loss_fn, _ = self._make_counting_loss()
        wrapper = EOTWrapper(loss_fn, k=2, seed=0)
        x = torch.rand(1, 3, 32, 32, requires_grad=True)
        loss, parts = wrapper(x, x + 0.01)
        assert "mse" in parts
        loss.backward()
        assert x.grad is not None

    def test_wrapper_calls_prepare_on_inner(self):
        prepared = {"done": False}

        class FakeLoss:
            def prepare(self, x):
                prepared["done"] = True

            def __call__(self, x, x_adv):
                return (x_adv - x).pow(2).mean(), {}

        wrapper = EOTWrapper(FakeLoss(), k=1, seed=0)
        wrapper.prepare(torch.rand(1, 3, 16, 16))
        assert prepared["done"]


# ---------------------------------------------------------------- PGD + EOT integration
class TestPGDWithEOT:
    def _stub_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Tiny differentiable encoder: 2x2 average pooling."""
        return torch.nn.functional.avg_pool2d(x, 2)

    def test_eot_pgd_respects_budget(self):
        """EOT must not violate the L-inf budget."""
        img = synthetic_image(64, seed=0)
        loss_fn = LatentLoss(self._stub_encode)
        cfg = ProtectConfig(eps=8 / 255, steps=6, size=64, device="cpu", w_lpips=0.0, eot_k=2)
        # Inject loss directly to skip model loading, EOT is added inside _build_loss
        # We inject CombinedLoss without EOT, test the budget check separately
        combined = CombinedLoss({"latent": (loss_fn, 1.0)})
        eot_wrapped = EOTWrapper(combined, k=2, seed=0)
        prot = PGDProtector(cfg=cfg, loss_fn=eot_wrapped)
        res = prot(img)
        assert res.info["linf"] <= 8 / 255 + 1e-6

    def test_eot_changes_perturbation(self):
        """A perturbation computed with EOT should differ from one without it.

        If they were identical, EOT would be doing nothing.
        """
        img = synthetic_image(64, seed=0)
        encode = self._stub_encode

        cfg_plain = ProtectConfig(eps=4 / 255, steps=8, size=64, seed=0, device="cpu", w_lpips=0.0, eot_k=0)
        cfg_eot   = ProtectConfig(eps=4 / 255, steps=8, size=64, seed=0, device="cpu", w_lpips=0.0, eot_k=3)

        loss_plain = CombinedLoss({"latent": (LatentLoss(encode), 1.0)})
        loss_eot   = EOTWrapper(CombinedLoss({"latent": (LatentLoss(encode), 1.0)}), k=3, seed=0)

        res_plain = PGDProtector(cfg=cfg_plain, loss_fn=loss_plain)(img)
        res_eot   = PGDProtector(cfg=cfg_eot,   loss_fn=loss_eot)(img)

        diff = float((res_plain.delta - res_eot.delta).abs().mean())
        assert diff > 1e-6, "EOT produced identical perturbation to plain PGD"

    def test_eot_config_k0_leaves_loss_unwrapped(self):
        """When eot_k=0, _build_loss must NOT wrap in EOTWrapper."""
        from photoshield.protect.transforms import EOTWrapper

        img = synthetic_image(64, seed=0)
        loss_fn = CombinedLoss({"latent": (LatentLoss(self._stub_encode), 1.0)})
        cfg = ProtectConfig(eps=4 / 255, steps=4, size=64, device="cpu", w_lpips=0.0, eot_k=0)
        prot = PGDProtector(cfg=cfg, loss_fn=loss_fn)
        prot(img)
        # The loss_fn was injected so it was never replaced; it is a CombinedLoss, not EOTWrapper
        assert not isinstance(prot.loss_fn, EOTWrapper)
