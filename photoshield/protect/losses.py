"""Loss terms for the attack.

Contract: every loss object is callable as `loss_fn(x, x_adv) -> (loss, parts)`
where `loss` is what PGD *minimises* and `parts` is a dict of python floats for
logging. Terms that should be *maximised* are therefore negated here, once, and
never in the optimiser - that way the sign of a term is decided in one place and
can be unit-tested.

Phase 2 ships `LatentLoss` (encoder attack). Phase 3 adds the identity term.
"""

from __future__ import annotations

import torch


def latent_rms(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Root-mean-square distance between two latents (differentiable).

    Shaped latents are (B,4,H/8,W/8) = ~786k numbers at 512 px. A raw L2 norm
    would grow with resolution and make `alpha` resolution-dependent; RMS keeps
    the scale comparable across image sizes. Same direction, stable step size.
    """
    return (z - z0).pow(2).mean().sqrt()


LATENT_TARGETS = ("none", "gray", "pattern")


def target_image(kind: str, height: int, width: int) -> torch.Tensor:
    """(1,3,H,W) target in [0,1] for the targeted encoder attack.

    "gray"    - flat mid-grey, the PhotoGuard encoder-attack target.
    "pattern" - deterministic high-contrast stripes/checker, in the spirit of
                Mist's textural target: the editor then paints that texture
                instead of the photo.
    """
    if kind == "gray":
        return torch.full((1, 3, height, width), 0.5)
    if kind == "pattern":
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        stripes = 0.5 + 0.5 * torch.sin((xx + yy) * (2 * torch.pi / 24))
        checker = ((xx // 32 + yy // 32) % 2).float()
        rgb = torch.stack([stripes, checker, 1.0 - stripes * checker], dim=0)
        return rgb.unsqueeze(0).clamp(0, 1)
    raise ValueError(f"unknown latent target {kind!r}; choose from {LATENT_TARGETS[1:]}")


class LatentLoss:
    """PhotoGuard-style encoder attack.

    Untargeted (`target=None`): maximise ||E(x_adv) - E(x)||.
    Targeted (`target` = (1,3,H,W) image): minimise ||E(x_adv) - E(target)||.
    PhotoGuard and Mist both use the targeted form: "be this other image" is a
    consistent direction for the editor, "be anything else" is not.

    `encode` is any callable [0,1] pixels -> latent. In production it is
    `SDVAE.encode`; in tests it is a tiny differentiable stub, so the optimiser
    can be verified on a laptop in milliseconds without downloading 330 MB.
    """

    def __init__(self, encode, target: torch.Tensor | None = None) -> None:
        self.encode = encode
        self.target = target
        self._z0: torch.Tensor | None = None
        self._zt: torch.Tensor | None = None

    def prepare(self, x: torch.Tensor) -> None:
        """Cache E(x) (and E(target)) once. Halves the forward cost and fixes the target."""
        with torch.no_grad():
            self._z0 = self.encode(x).detach()
            if self.target is not None:
                self._zt = self.encode(self.target.to(x.device, x.dtype)).detach()

    def __call__(self, x: torch.Tensor, x_adv: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self._z0 is None:
            self.prepare(x)
        z = self.encode(x_adv)
        d = latent_rms(z, self._z0)
        if self._zt is None:
            # negated: PGD minimises, we want the distance to be large
            return -d, {"latent_rms": float(d.detach())}
        dt = latent_rms(z, self._zt)
        # positive: we want to be close to the target. latent_rms still logs the
        # distance from the source so both modes print comparable numbers.
        return dt, {"latent_rms": float(d.detach()), "to_target": float(dt.detach())}

    def __repr__(self) -> str:
        mode = "targeted" if self.target is not None else "untargeted"
        return f"LatentLoss({mode}, encode={type(self.encode).__name__})"


class LPIPSTerm:
    """Perceptual penalty: keep the perturbation invisible.

    Added after Phase 2 measurement showed the pure encoder attack reached
    LPIPS 0.275 at 4/255 (13x the random-noise control at the same eps): it
    spends the whole budget on structured high-frequency texture because
    nothing in the objective says "stay invisible". This term says it.

    It is a *penalty* (positive, minimised), unlike LatentLoss which is negated
    because it is maximised. Sign lives here, once, and is unit-tested.
    """

    def __init__(self, device: str = "cpu", net: str = "alex") -> None:
        self.device = device
        self.net = net
        self._model = None

    def _model_or_load(self):
        if self._model is None:
            from ..metrics import get_lpips  # lazy: torchvision + 233 MB trunk

            self._model = get_lpips(net=self.net, device=self.device)
        return self._model

    def __call__(self, x: torch.Tensor, x_adv: torch.Tensor) -> tuple[torch.Tensor, dict]:
        model = self._model_or_load()
        # LPIPS is defined on [-1,1]; our tensors are [0,1]
        value = model(x.to(self.device) * 2 - 1, x_adv.to(self.device) * 2 - 1).mean()
        return value, {"lpips": float(value.detach())}


class CombinedLoss:
    """Weighted sum of terms: L = sum_i w_i * term_i(x, x_adv).

    Every term returns a value that is *minimised* (already sign-corrected).
    Weights: `+1.0` for a penalty (LPIPS), `+1.0` for an already-negated
    maximisation target (LatentLoss). Phase 3 adds the identity term with its
    own weight `b`.

    `prepare(x)` is forwarded so expensive one-off work (E(x)) happens once.
    """

    def __init__(self, terms: dict[str, tuple[object, float]]) -> None:
        if not terms:
            raise ValueError("need at least one term")
        self.terms = terms

    def prepare(self, x: torch.Tensor) -> None:
        for term, _ in self.terms.values():
            if hasattr(term, "prepare"):
                term.prepare(x)

    def __call__(self, x: torch.Tensor, x_adv: torch.Tensor) -> tuple[torch.Tensor, dict]:
        total = None
        parts: dict[str, float] = {}
        for name, (term, weight) in self.terms.items():
            value, sub = term(x, x_adv)
            total = weight * value if total is None else total + weight * value
            if not sub:  # terms that log nothing get their raw value under their own name
                parts[name] = float(value.detach())
            parts.update(sub)
        return total, parts

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}*{w:g}" for k, (_, w) in self.terms.items())
        return f"CombinedLoss({inner})"


def crop_resize(x: torch.Tensor, box: tuple[int, int, int, int], size: int = 112) -> torch.Tensor:
    """Cut a box out of (B,3,H,W) and resize it to (B,3,size,size).

    Differentiable by construction (slice + bilinear interpolate), which is the
    whole reason the identity loss can be optimised: the crop of the protected
    image must carry gradients back to every pixel inside the box.
    """
    if len(box) != 4:
        raise ValueError(f"box must be (x0,y0,x1,y1), got {box!r}")
    x0, y0, x1, y1 = (int(v) for v in box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty box {box!r}")
    crop = x[..., y0:y1, x0:x1]
    return torch.nn.functional.interpolate(
        crop, size=(size, size), mode="bilinear", align_corners=False
    )


def align_crop(x: torch.Tensor, theta, size: int = 112) -> torch.Tensor:
    """Differentiable landmark-aligned crop: (B,3,H,W) -> (B,3,size,size).

    `theta` comes from `models.arcface.affine_theta` (the same affine as
    insightface.norm_crop), so the loss and the ArcFace metric finally look at
    the same face. `grid_sample` is differentiable; `cv2.warpAffine` is not.
    """
    if x.ndim != 4:
        raise ValueError(f"expected (B,3,H,W), got {tuple(x.shape)}")
    b = x.shape[0]
    t = torch.as_tensor(theta, dtype=x.dtype, device=x.device)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.shape[0] == 1 and b > 1:
        t = t.expand(b, -1, -1)
    grid = torch.nn.functional.affine_grid(
        t, size=(b, 3, size, size), align_corners=False
    )
    # border padding == cv2.BORDER_REPLICATE in align_face
    return torch.nn.functional.grid_sample(
        x, grid, mode="bilinear", padding_mode="border", align_corners=False
    )


class IdentityLoss:
    """Minimise cos( embed(aligned(roundtrip(x_adv))), embed(aligned(roundtrip(x))) ).

    `roundtrip` = decode(encode(.)) - the VAE decoder stands in for the editor
    (Phase 3a). `theta` is the landmark-aligned sampling matrix from
    `IdentityModel.alignment_theta`; `box` remains for diagnostics/tests.

    Why minimise cosine instead of "maximise distance": cosine is already a
    similarity in [-1, 1]; minimising it pushes the edited face toward "not this
    person". Negating a distance would be the same thing with an extra sign to
    get wrong.
    """

    def __init__(
        self,
        embed,
        box=None,
        size: int = 112,
        encode=None,
        decode=None,
        theta=None,
    ) -> None:
        if (box is None) == (theta is None):
            raise ValueError("pass exactly one of box=(x0,y0,x1,y1) or theta=(2,3)")
        self.embed = embed
        self.box = box
        self.theta = theta
        self.size = size
        self.encode = encode
        self.decode = decode
        self._src = None

    def _roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        if self.encode is None or self.decode is None:
            return x
        return self.decode(self.encode(x))

    def _view(self, x: torch.Tensor) -> torch.Tensor:
        y = self._roundtrip(x)
        if self.theta is not None:
            return align_crop(y, self.theta, self.size)
        return crop_resize(y, self.box, self.size)

    def prepare(self, x: torch.Tensor) -> None:
        """Reference identity is taken from the *round-tripped original*, not the
        raw original: VAE reconstruction already changes features slightly, and
        we want to measure the perturbation's effect, not the codec's."""
        with torch.no_grad():
            self._src = self.embed(self._view(x)).detach()

    def __call__(self, x: torch.Tensor, x_adv: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self._src is None:
            self.prepare(x)
        emb = self.embed(self._view(x_adv))
        cos = (emb * self._src).sum(dim=-1).mean()
        return cos, {"cos_id": float(cos.detach())}

    def __repr__(self) -> str:
        view = "aligned" if self.theta is not None else f"box={self.box}"
        return f"IdentityLoss({view}, size={self.size}, surrogate={'vae' if self.decode else 'pixel'})"
