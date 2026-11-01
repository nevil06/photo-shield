"""Expectation-over-Transformations (EOT) augmentations for Phase 4.

WHY THESE EXIST
---------------
Phase 2-3 build a perturbation that fools the VAE encoder on the exact pixel
array it was created for. Real photos go through lossy channels before editing:
social-media platforms recompress to JPEG q≈75-85, resize to their CDN
resolution, sometimes crop. If the perturbation breaks under JPEG q=80, it
offers no real-world protection.

EOT (Athalye et al., 2018) fixes this by averaging the gradient *over a batch
of random augmentations* at every PGD step:

    g_eot = (1/K) * Σ_k  d/d(delta)  L(aug_k(x + delta))

This forces the perturbation to survive the augmentations because the gradient
only reinforces patterns that survive ALL of them.

DIFFERENTIABLE JPEG
-------------------
JPEG is not differentiable as shipped. We approximate it by:
  1. Applying an 8x8 DCT (matrix multiply — fully differentiable)
  2. Dividing by the standard luminance quantisation matrix scaled by quality q
  3. Rounding with a straight-through estimator (STE): forward rounds, backward
     passes gradient straight through. This is the standard trick for
     non-differentiable discrete ops.
  4. De-quantising and applying the inverse DCT.

Quality q=90 → small quantisation → mild degradation (social-media CDN).
Quality q=50 → heavy quantisation → stress test.

WHICH AUGMENTATIONS
-------------------
Chosen to match the threat model (social-media upload pipeline):
  - JPEG q ∈ [50, 90]         (primary threat)
  - Resize long side ×[0.7, 1.0] then back to original (CDN resize)
  - Random crop 90% of area   (some platforms auto-crop)
  - Gaussian blur σ ∈ [0, 0.5] (optional; mild)

NOT included: rotation, horizontal flip — those change identity too drastically
and are not part of the typical upload pipeline.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- DCT basis
def _dct_basis(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build the 1-D DCT-II orthonormal basis matrix (n, n).

    C[k, x] = w(k) * cos(pi*(2x+1)*k / (2n))
    where w(0) = 1/sqrt(n), w(k>0) = sqrt(2/n).

    This is the forward DCT used by JPEG.
    """
    k = torch.arange(n, device=device, dtype=dtype)      # (n,)
    x = torch.arange(n, device=device, dtype=dtype)      # (n,)
    C = torch.cos(math.pi * k[:, None] * (2 * x[None, :] + 1) / (2 * n))
    # Orthonormal weights
    w = torch.full((n,), math.sqrt(2.0 / n), device=device, dtype=dtype)
    w[0] = 1.0 / math.sqrt(n)
    return w[:, None] * C  # (n, n)


# Cache bases so they are only built once per (n, device, dtype)
_DCT_CACHE: dict = {}


def _get_basis(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n, str(device), str(dtype))
    if key not in _DCT_CACHE:
        _DCT_CACHE[key] = _dct_basis(n, device, dtype)
    return _DCT_CACHE[key]


def dct2d(x: torch.Tensor, norm: bool = True) -> torch.Tensor:
    """2-D DCT-II of the last two dims of x, via separable matrix products.

    x: (..., H, W) — both H and W must be multiples of 8.
    Returns: (..., H, W) in DCT-coefficient space.
    """
    *batch, H, W = x.shape
    C_h = _get_basis(H, x.device, x.dtype)  # (H, H)
    C_w = _get_basis(W, x.device, x.dtype)  # (W, W)
    # Rows: X' = C @ X → (batch, H, W) with @ on last two dims
    y = torch.einsum("...ij,...jk->...ik", C_h.unsqueeze(0).expand(*batch, -1, -1), x)
    # Cols: Y = X' @ C^T
    y = torch.einsum("...ij,...kj->...ik", y, C_w.unsqueeze(0).expand(*batch, -1, -1))
    return y


def idct2d(y: torch.Tensor) -> torch.Tensor:
    """2-D inverse DCT-II (= DCT-III). C is orthonormal so C^{-1} = C^T."""
    *batch, H, W = y.shape
    C_h = _get_basis(H, y.device, y.dtype)
    C_w = _get_basis(W, y.device, y.dtype)
    x = torch.einsum("...ij,...jk->...ik", C_h.T.unsqueeze(0).expand(*batch, -1, -1), y)
    x = torch.einsum("...ij,...kj->...ik", x, C_w.T.unsqueeze(0).expand(*batch, -1, -1))
    return x


# ---------------------------------------------------------------- JPEG quantisation matrix
# Standard JPEG luminance table (ITU-T T.81 Annex K)
_JPEG_LUM_TABLE = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)

# Standard JPEG chrominance table
_JPEG_CHR_TABLE = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


def _quant_table(quality: float, luma: bool = True) -> torch.Tensor:
    """Scale the standard JPEG table for a given quality (1..100).

    IJG (libjpeg) formula:
        if quality < 50:  scale = 5000 / quality
        else:             scale = 200 - 2*quality
        table = floor((table * scale + 50) / 100)
        table = clip(table, 1, 255)
    """
    t = _JPEG_LUM_TABLE if luma else _JPEG_CHR_TABLE
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    q = torch.floor((t * scale + 50.0) / 100.0).clamp(1.0, 255.0)
    return q


class _STERound(torch.autograd.Function):
    """Round with a straight-through estimator (STE).

    Forward: round to integer.
    Backward: gradient passes straight through (identity).
    This lets the DCT quantisation step participate in backprop.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: N805
        return x.round()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:  # noqa: N805
        return grad


_ste_round = _STERound.apply


def _jpeg_compress_single_channel(
    x: torch.Tensor, Q: torch.Tensor, block: int = 8
) -> torch.Tensor:
    """JPEG-approximate a single-channel image x ∈ [0,1].

    x: (H, W)  — H,W must be multiples of `block` (8 for JPEG)
    Q: (block, block) quantisation table

    Pipeline:
        1. Scale to [0, 255] and level-shift to [-128, 127]
        2. Split into (block×block) patches
        3. Forward 2D-DCT of each patch
        4. Quantise (round(DCT / Q)) with STE
        5. De-quantise (× Q)
        6. Inverse DCT
        7. Unshift and scale back to [0, 1]
    """
    H, W = x.shape
    assert H % block == 0 and W % block == 0, f"image dims must be multiples of {block}"

    Q = Q.to(x.device).to(x.dtype)

    x255 = x * 255.0 - 128.0  # level shift

    # Reshape to (n_blocks_h, block, n_blocks_w, block) then permute
    # to get all patches: (n_blocks_h * n_blocks_w, block, block)
    nh, nw = H // block, W // block
    patches = x255.reshape(nh, block, nw, block).permute(0, 2, 1, 3)  # (nh, nw, block, block)

    dct_patches = dct2d(patches)                   # (nh, nw, block, block)
    quant = _ste_round(dct_patches / Q)            # STE round
    dequant = quant * Q                            # de-quantise
    idct_patches = idct2d(dequant)                 # (nh, nw, block, block)

    # Reassemble image
    out255 = idct_patches.permute(0, 2, 1, 3).reshape(H, W) + 128.0
    return (out255 / 255.0).clamp(0.0, 1.0)


def differentiable_jpeg(x: torch.Tensor, quality: float) -> torch.Tensor:
    """Differentiable JPEG approximation for a (B, 3, H, W) [0,1] tensor.

    Uses luma quantisation table for all three channels (RGB, not YCbCr).
    This is a slight simplification but acceptable for the adversarial objective:
    the goal is robustness, not colour accuracy. Real JPEG uses YCbCr; using a
    single table avoids a colour-space transform chain that adds noise to the
    gradient without helping.

    H and W must be multiples of 8.
    """
    if quality <= 0 or quality > 100:
        raise ValueError(f"quality must be in (0, 100], got {quality}")
    Q = _quant_table(quality, luma=True)
    B, C, H, W = x.shape
    out = torch.stack(
        [
            torch.stack(
                [_jpeg_compress_single_channel(x[b, c], Q) for c in range(C)], dim=0
            )
            for b in range(B)
        ],
        dim=0,
    )
    return out


# ---------------------------------------------------------------- random augmentations
def random_jpeg(x: torch.Tensor, q_min: float = 50.0, q_max: float = 90.0, rng=None) -> torch.Tensor:
    """Apply differentiable JPEG with quality sampled uniformly from [q_min, q_max]."""
    gen = rng or torch.default_generator
    q = float(torch.rand(1, generator=gen)) * (q_max - q_min) + q_min
    return differentiable_jpeg(x, float(q))


def random_resize_back(x: torch.Tensor, scale_min: float = 0.7, scale_max: float = 1.0, rng=None) -> torch.Tensor:
    """Down-scale then up-scale back to original size (CDN resize simulation).

    scale ∈ [0.7, 1.0] → image is downsampled to 70-100% then bicubic back.
    At scale=1.0 this is an identity (within bilinear precision).
    """
    B, C, H, W = x.shape
    gen = rng or torch.default_generator
    scale = float(torch.rand(1, generator=gen)) * (scale_max - scale_min) + scale_min
    if abs(scale - 1.0) < 1e-4:
        return x  # skip the interpolation entirely
    small_h = max(8, int(H * scale))
    small_w = max(8, int(W * scale))
    small = F.interpolate(x, size=(small_h, small_w), mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)


def random_crop_pad(x: torch.Tensor, min_area: float = 0.90, rng=None) -> torch.Tensor:
    """Random crop to `min_area` fraction of area, then pad back to original size.

    Padding with reflected values so we don't introduce a solid border
    (which the encoder would notice differently from a natural image).

    Maintains spatial size so the PGD loop's shape assumptions hold.
    """
    B, C, H, W = x.shape
    gen = rng or torch.default_generator
    # Sample crop size
    crop_ratio = (float(torch.rand(1, generator=gen)) * (1.0 - min_area) + min_area) ** 0.5
    ch = max(8, int(H * crop_ratio))
    cw = max(8, int(W * crop_ratio))
    # Sample top-left corner
    top_max = H - ch
    left_max = W - cw
    if top_max <= 0 or left_max <= 0:
        return x
    # Use torch.randint with the generator for reproducibility
    top  = int(torch.randint(0, top_max + 1,  (1,), generator=gen))
    left = int(torch.randint(0, left_max + 1, (1,), generator=gen))
    crop = x[:, :, top : top + ch, left : left + cw]
    return F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)


def random_blur(x: torch.Tensor, sigma_max: float = 0.5, rng=None) -> torch.Tensor:
    """Gaussian blur with σ ∈ [0, sigma_max]. At σ=0 this is an identity."""
    gen = rng or torch.default_generator
    sigma = float(torch.rand(1, generator=gen)) * sigma_max
    if sigma < 0.05:
        return x
    # Kernel radius = 3σ, minimum size 3
    r = max(1, int(math.ceil(3 * sigma)))
    k = 2 * r + 1
    device, dtype = x.device, x.dtype
    # 1-D Gaussian kernel
    ax = torch.arange(-r, r + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-ax.pow(2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]  # (k, k)
    # Expand to (C, 1, k, k) for depthwise convolution
    B, C, H, W = x.shape
    kernel = kernel_2d.expand(C, 1, k, k)
    pad = r
    return F.conv2d(x, kernel, groups=C, padding=pad)


# ---------------------------------------------------------------- EOT wrapper
DEFAULT_TRANSFORMS: list[Callable] = [
    random_jpeg,
    random_resize_back,
    random_crop_pad,
]


class EOTWrapper:
    """Wraps any loss_fn to average its gradient over K random augmentations.

    At each PGD step, instead of:
        g = d/d_delta  L(x_adv)

    we compute:
        g_eot = (1/K) * Σ_{k=1}^{K}  d/d_delta  L(aug_k(x_adv))

    where each aug_k is an independent random draw.

    The trick: we call loss_fn K times and sum the *losses* before calling
    .backward() once. PyTorch accumulates the gradients correctly.

    `loss_fn(x, x_adv) -> (loss, parts)` contract is preserved.
    The reported `parts` are averaged over the K samples.
    """

    def __init__(
        self,
        loss_fn,
        transforms: list[Callable] | None = None,
        k: int = 4,
        seed: int = 0,
    ) -> None:
        self.loss_fn = loss_fn
        self.transforms = transforms if transforms is not None else DEFAULT_TRANSFORMS
        self.k = k
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def prepare(self, x: torch.Tensor) -> None:
        if hasattr(self.loss_fn, "prepare"):
            self.loss_fn.prepare(x)

    def __call__(self, x: torch.Tensor, x_adv: torch.Tensor):
        total_loss = None
        total_parts: dict[str, float] = {}

        for _ in range(self.k):
            # Apply the same random augmentation to both x and x_adv so the
            # latent reference (E(x)) and the perturbed image see the same crop.
            x_aug     = self._augment(x.detach())       # ref image: no grad needed
            x_adv_aug = self._augment(x_adv)            # perturbed: gradients flow

            loss, parts = self.loss_fn(x_aug, x_adv_aug)
            total_loss = loss if total_loss is None else total_loss + loss

            for k, v in parts.items():
                total_parts[k] = total_parts.get(k, 0.0) + v / self.k

        return total_loss / self.k, total_parts  # type: ignore[operator]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence, each with an independent random draw."""
        for t in self.transforms:
            x = t(x, rng=self._rng)
        return x

    def __repr__(self) -> str:
        names = [t.__name__ for t in self.transforms]
        return f"EOTWrapper(k={self.k}, transforms={names})"
