"""Phase 2 tests: does PGD actually run PGD, and does it keep its promise?

No model weights are needed: `StubEncoder` is a tiny differentiable network with
the same input/output contract as `SDVAE.encode`. One gated test uses the real
VAE (`PHOTOSHIELD_RUN_VAE=1`).
"""

import math
import os

import pytest
import torch

from photoshield.config import ProtectConfig
from photoshield.data import synthetic_image, to_tensor
from photoshield.protect.losses import CombinedLoss, LatentLoss, latent_rms
from photoshield.protect.pgd_engine import PGDProtector

RUN_VAE = os.environ.get("PHOTOSHIELD_RUN_VAE") == "1"
needs_vae = pytest.mark.skipif(not RUN_VAE, reason="set PHOTOSHIELD_RUN_VAE=1 (needs the SD VAE)")


class StubEncoder(torch.nn.Module):
    """Stands in for a VAE encoder: downsampling + a nonlinearity, differentiable."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, stride=2, padding=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(8, 4, 3, stride=1, padding=1),
        )

    def forward(self, x01):
        return self.net(x01 * 2 - 1)


def make(eps=4 / 255, steps=10, size=128, seed=0):
    enc = StubEncoder()
    cfg = ProtectConfig(eps=eps, steps=steps, size=size, seed=seed, device="cpu")
    return PGDProtector(cfg, loss_fn=LatentLoss(enc)), enc


def test_budget_is_respected_at_4_and_8_of_255():
    for eps in (4 / 255, 8 / 255):
        prot, _ = make(eps=eps)
        res = prot(synthetic_image(128))
        assert res.info["linf"] <= eps + 1e-6, f"eps={eps}: {res.info['linf']}"
        assert res.linf <= eps + 1e-6  # same number, from the tensor itself


def test_output_keeps_working_size_and_is_divisible_by_8():
    prot, _ = make(size=128)
    img = synthetic_image(300)  # 300 -> cropped to 296
    res = prot(img)
    w, h = res.image.size
    assert w % 8 == 0 and h % 8 == 0
    assert max(w, h) == 128 or max(w, h) == 128 - 0  # long side hits the target
    assert res.info["input_size"] == "300x300"


def test_attack_moves_the_latent_further_than_random_noise():
    """The control that kills 'you just added noise'."""
    img = synthetic_image(128)
    prot, enc = make(steps=15)
    res = prot(img)

    x = to_tensor(img)
    x_adv = to_tensor(res.image)
    with torch.no_grad():
        z0 = enc(x)
        d_attack = float(latent_rms(enc(x_adv), z0))

        gen = torch.Generator().manual_seed(0)
        noise = (torch.rand(x.shape, generator=gen) * 2 - 1) * (4 / 255)
        d_random = float(latent_rms(enc((x + noise).clamp(0, 1)), z0))

    assert d_attack > d_random, f"attack {d_attack:.4f} <= random {d_random:.4f}"


def test_deterministic_for_same_seed():
    """Same weights + same cfg + same photo -> identical bytes.

    Note: the encoder must be *shared* between the two runs; two `StubEncoder()`
    instances have different random init, which says nothing about determinism.
    """
    enc = StubEncoder()
    img = synthetic_image(128)

    def run(seed: int) -> bytes:
        cfg = ProtectConfig(eps=4 / 255, steps=10, size=128, seed=seed, device="cpu")
        return PGDProtector(cfg, loss_fn=LatentLoss(enc))(img).image.tobytes()

    assert run(3) == run(3)


def test_history_is_logged_and_loss_grows():
    prot, _ = make(steps=8)
    res = prot(synthetic_image(128))
    hist = res.info["history"]
    assert len(hist) == 8
    assert res.info["loss_end"] > res.info["loss_start"]
    assert all(math.isfinite(h["latent_rms"]) for h in hist)


def test_does_not_mutate_the_input_image():
    img = synthetic_image(128)
    before = img.tobytes()
    make()[0](img)
    assert img.tobytes() == before


def test_combined_loss_applies_weights_without_duplicating_keys():
    """The sign of every term is decided in losses.py, so it is tested there."""

    class Term:
        def __init__(self, value, key):
            self.value, self.key = value, key

        def __call__(self, x, x_adv):
            return torch.tensor(self.value, requires_grad=True), {self.key: self.value}

    loss = CombinedLoss({"latent": (Term(2.0, "latent_rms"), 1.0), "lpips": (Term(0.5, "lpips"), 3.0)})
    total, parts = loss(torch.zeros(1), torch.zeros(1))
    assert float(total.detach()) == pytest.approx(2.0 + 3.0 * 0.5)
    assert parts == {"latent_rms": 2.0, "lpips": 0.5}  # no duplicate "latent": -2.0


def test_combined_loss_forwards_prepare():
    called = {}

    class Term:
        def prepare(self, x):
            called["x"] = True

        def __call__(self, x, x_adv):
            return torch.tensor(0.0, requires_grad=True), {}

    CombinedLoss({"t": (Term(), 1.0)}).prepare(torch.zeros(1))
    assert called.get("x") is True


@needs_vae
def test_real_vae_smoke():
    """Gated: needs the 330 MB SD VAE. 5 steps at 128 px is a few seconds on CPU."""
    from photoshield.models.vae import load_vae

    vae = load_vae(device="cpu")
    cfg = ProtectConfig(eps=4 / 255, steps=5, size=128, device="cpu")
    prot = PGDProtector(cfg, loss_fn=LatentLoss(vae.encode), verbose=True)
    res = prot(synthetic_image(128))

    assert res.info["linf"] <= 4 / 255 + 1e-6
    assert res.info["loss_end"] > 0.0
    assert res.image.size == (128, 128)
    # round-trip sanity: the frozen decoder must still reconstruct the clean image
    with torch.no_grad():
        rec = vae.decode(vae.encode(to_tensor(synthetic_image(128)))) - to_tensor(synthetic_image(128))
    assert float(rec.abs().mean()) < 0.05


def test_target_image_shapes_and_range():
    from photoshield.protect.losses import target_image

    for kind in ("gray", "pattern"):
        t = target_image(kind, 40, 56)
        assert t.shape == (1, 3, 40, 56)
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0
    with pytest.raises(ValueError):
        target_image("noise", 8, 8)


def test_targeted_attack_moves_latent_toward_target():
    from photoshield.protect.losses import target_image

    enc = StubEncoder()
    img = synthetic_image(128)
    target = target_image("pattern", 128, 128)
    cfg = ProtectConfig(eps=8 / 255, steps=15, size=128, device="cpu")
    res = PGDProtector(cfg, loss_fn=LatentLoss(enc, target=target))(img)
    hist = res.info["history"]
    assert hist[-1]["to_target"] < hist[0]["to_target"]
    assert res.info["linf"] <= 8 / 255 + 1e-6


def test_config_rejects_unknown_target():
    with pytest.raises(ValueError):
        ProtectConfig(latent_target="noise")
