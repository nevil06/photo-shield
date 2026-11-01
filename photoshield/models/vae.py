"""Frozen Stable Diffusion VAE: the "eye" we attack in Phase 2.

Why this model: SD 1.x/2.x pipelines all start by encoding the photo with this
VAE (8x downsampling, 4 latent channels). If the latent is pushed off the
manifold of natural images, every downstream step (noising, denoising, decoding)
receives something the UNet considers junk. Because the VAE is shared across
many checkpoints, an attack on it transfers better than an attack on one UNet.

Only the VAE is loaded - no UNet, no scheduler, no text encoder (~330 MB vs ~4 GB).
"""

from __future__ import annotations

import torch

DEFAULT_VAE_ID = "stabilityai/sd-vae-ft-mse"


class SDVAE(torch.nn.Module):
    """encode: [0,1] pixels -> scaled latent.  decode: latent -> [0,1] pixels.

    The public API deliberately speaks [0, 1] like the rest of PhotoShield; the
    [-1, 1] conversion diffusers expects happens inside, in one place.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_VAE_ID,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        from diffusers import AutoencoderKL  # local import: diffusers is heavy

        self.model_id = model_id
        self.device = torch.device(device)
        self.vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=dtype)
        self.vae.requires_grad_(False)  # we never train the VAE, only differentiate through it
        # Recompute activations in backward instead of storing them. Measured at
        # 512 px (encode+decode, fwd+bwd, CPU): peak RAM 9.8 GB -> 6.0 GB, same
        # speed, bit-identical gradients. Without it the 512 px run ran out of RAM.
        self.vae.enable_gradient_checkpointing()
        self.vae.eval()
        self.vae.to(self.device)
        self.scaling_factor = float(self.vae.config.scaling_factor)
        # sd-vae-ft-mse has no shift_factor; SDXL-style VAEs do. Default to 0.
        self.shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)

    # ------------------------------------------------------------ core
    def encode(self, x01: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) in [0,1] -> (B,4,H/8,W/8) scaled latent.

        `.mode()` instead of `.sample()`: sampling injects noise into the
        gradient every step, which makes PGD jitter and results irreproducible.
        """
        x = x01.to(self.device) * 2.0 - 1.0
        z = self.vae.encode(x).latent_dist.mode()
        return (z - self.shift_factor) * self.scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Scaled latent -> (B,3,H,W) in [0,1]. Used as the Phase 3 surrogate."""
        x = self.vae.decode(z / self.scaling_factor + self.shift_factor).sample
        return ((x + 1.0) / 2.0).clamp(0, 1)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.encode(x01)

    def __repr__(self) -> str:
        return f"SDVAE({self.model_id}, scaling={self.scaling_factor:g}, device={self.device})"


_CACHE: dict[tuple[str, str], SDVAE] = {}


def load_vae(model_id: str = DEFAULT_VAE_ID, device: str = "cpu") -> SDVAE:
    """Load once per (model, device); the 330 MB belongs in RAM, not in a loop."""
    key = (model_id, str(device))
    if key not in _CACHE:
        _CACHE[key] = SDVAE(model_id, device=device)
    return _CACHE[key]
