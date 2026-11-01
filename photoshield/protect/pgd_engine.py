"""PGD on the VAE encoder - the Phase 2 baseline.

Algorithm (every line matters, this is the viva answer):

    delta = 0
    repeat K times:
        x_adv = clamp(x + delta, 0, 1)          # stay a legal image
        loss  = -latent_distance(E(x_adv), E(x)) # maximise the latent shift
        g     = d loss / d delta                 # which way is uphill
        delta = clamp(delta - alpha*sign(g), -eps, eps)   # step + PROJECT
    return x + delta

`sign(g)` makes every pixel move by exactly alpha: maximum change per unit of
budget, and the result is high-frequency (humans notice high-frequency noise at
equal amplitude much less than low-frequency shifts).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from PIL import Image

from ..config import ProtectConfig, resolve_device
from ..data import resize_long_side, round_to_multiple, to_pil, to_tensor
from .base import ProtectResult, Protector

if TYPE_CHECKING:
    from ..models.vae import SDVAE


class PGDProtector(Protector):
    """Encoder attack. `loss_fn` is injectable so tests need no model weights."""

    name = "pgd-vae-encoder"

    def __init__(self, cfg: ProtectConfig | None = None, loss_fn=None, verbose: bool = False) -> None:
        super().__init__(cfg)
        self.device = resolve_device(self.cfg.device)
        self.loss_fn = loss_fn
        self.verbose = verbose
        self._model: SDVAE | None = None

    # ------------------------------------------------------------ setup
    def _build_loss(self, img: Image.Image):
        """Assemble -w_latent*latent + w_identity*cos_id + w_lpips*LPIPS.

        Built from the *working-size* image because the identity term needs the
        landmark-aligned sampling matrix in those exact pixel coordinates (the
        same theta is later applied to the VAE round-tripped image).
        """
        if self.loss_fn is not None:  # injected (tests, ablations)
            return self.loss_fn

        from .losses import CombinedLoss, LatentLoss, LPIPSTerm, target_image

        if self._model is None:
            from ..models.vae import load_vae

            self._model = load_vae(device=self.device)

        target = None
        if self.cfg.latent_target != "none":
            w, h = img.size
            target = target_image(self.cfg.latent_target, h, w)
        terms: dict = {"latent": (LatentLoss(self._model.encode, target=target), self.cfg.w_latent)}
        if self.cfg.w_lpips > 0:
            terms["lpips"] = (LPIPSTerm(device=self.device), self.cfg.w_lpips)

        if self.cfg.w_identity > 0:
            from ..models.arcface import IdentityModel

            from .losses import IdentityLoss

            ident = IdentityModel(device=self.device, torch_embedder=True)
            theta = ident.alignment_theta(img)
            if theta is None:
                raise ValueError(
                    "w_identity > 0 but no face was detected in the working image. "
                    "Either upload a photo with a visible face, or set w_identity=0."
                )
            terms["identity"] = (
                IdentityLoss(
                    embed=ident.embedder,
                    theta=theta,
                    encode=self._model.encode,
                    decode=self._model.decode,
                ),
                self.cfg.w_identity,
            )
            if self.verbose:
                print(f"identity theta {theta.shape}, embedder {type(ident.embedder).__name__}")

        self.loss_fn = CombinedLoss(terms)
        if self.cfg.eot_k > 0:
            from .transforms import EOTWrapper

            self.loss_fn = EOTWrapper(
                self.loss_fn,
                k=self.cfg.eot_k,
                seed=self.cfg.seed,
            )
            if self.verbose:
                print(f"EOT enabled: {self.loss_fn!r}")
        if self.verbose:
            print(f"loaded {self._model!r}")
            print(f"objective {self.loss_fn!r}")
        return self.loss_fn

    def working_image(self, image: Image.Image) -> Image.Image:
        """Long side -> cfg.size, then crop to a multiple of 8 (VAE requirement).

        We crop instead of padding: padding invents pixels that were never in the
        photo, cropping the last <=7 columns is invisible.
        """
        img = resize_long_side(image, self.cfg.size)
        w, h = img.size
        w2, h2 = round_to_multiple(w), round_to_multiple(h)
        if (w2, h2) != (w, h):
            img = img.crop((0, 0, w2, h2))
        return img

    # ------------------------------------------------------------ the attack
    def protect(self, image: Image.Image) -> ProtectResult:
        torch.manual_seed(self.cfg.seed)  # reproducible: same photo + cfg -> same delta
        started = time.perf_counter()

        img = self.working_image(image)
        x = to_tensor(img).to(self.device)
        if x.shape[-2] % 8 or x.shape[-1] % 8:  # defensive: catches a bad resize path
            raise ValueError(f"VAE needs sides divisible by 8, got {tuple(x.shape[-2:])}")

        eps, alpha = self.cfg.eps, self.cfg.alpha
        loss_fn = self._build_loss(img)
        if hasattr(loss_fn, "prepare"):
            loss_fn.prepare(x)

        delta = torch.zeros_like(x)
        history: list[dict] = []
        log_every = max(1, self.cfg.steps // 5)

        for step in range(self.cfg.steps):
            d = delta.clone().requires_grad_(True)  # fresh leaf each step
            x_adv = (x + d).clamp(0.0, 1.0)
            loss, parts = loss_fn(x, x_adv)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}; lower alpha/eps")
            grad = torch.autograd.grad(loss, d)[0]

            with torch.no_grad():  # in-place on a leaf is only legal without grad mode
                delta = (delta - alpha * grad.sign()).clamp(-eps, eps)

            history.append({"step": step, **{k: float(v) for k, v in parts.items()}})
            if self.verbose and (step % log_every == 0 or step == self.cfg.steps - 1):
                extra = "".join(
                    f"  {k}={v:.4f}" for k, v in parts.items() if k != "latent_rms"
                )
                print(
                    f"  step {step + 1:3d}/{self.cfg.steps}  "
                    f"latent_rms={parts.get('latent_rms', float('nan')):.4f}{extra}"
                )

        x_adv = (x + delta).clamp(0.0, 1.0).detach()
        eff_delta = x_adv - x  # the perturbation that ACTUALLY happened
        linf = float(eff_delta.abs().max())

        # Hard promise check: clamping delta to [-eps,eps] is not enough proof,
        # because the [0,1] clamp on x_adv can shrink it. Assert on reality.
        if linf > eps + 1e-6:
            raise RuntimeError(f"budget violated: linf={linf:.6f} > eps={eps:.6f}")

        info = {
            "protector": self.name,
            "device": self.device,
            "config": self.cfg.as_dict(),
            "input_size": f"{image.size[0]}x{image.size[1]}",
            "working_size": f"{img.size[0]}x{img.size[1]}",
            "linf": linf,
            "linf_255": linf * 255.0,
            "loss_start": history[0].get("latent_rms", 0.0) if history else 0.0,
            "loss_end": history[-1].get("latent_rms", 0.0) if history else 0.0,
            "seconds": round(time.perf_counter() - started, 2),
            "history": history,
        }
        return ProtectResult(image=to_pil(x_adv), delta=eff_delta.cpu(), info=info)
