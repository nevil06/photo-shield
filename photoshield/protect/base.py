"""The contract every protector must satisfy.

Why an abstract base class at all: Phase 2 (VAE-encoder PGD), Phase 3
(+ identity loss) and Phase 7 (DCT) must be swappable in the evaluator and the
Gradio app without editing them. One interface, three implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from PIL import Image

from ..config import ProtectConfig


@dataclass
class ProtectResult:
    """What a protector returns.

    image : PIL, same size as the input (never upscaled/downscaled back
            differently - that would itself be a visible change)
    delta : the additive perturbation actually applied, (1,3,H,W) in [0,1]
            units. Kept so we can prove `delta.abs().max() <= eps`.
    info  : free-form run metadata (loss history, timings, config) - goes into
            the CSV and into docs_private/05_results_and_limits.md
    """

    image: Image.Image
    delta: torch.Tensor
    info: dict = field(default_factory=dict)

    @property
    def linf(self) -> float:
        return float(self.delta.abs().max())


class Protector(ABC):
    """Turns a photo into a protected photo."""

    name: str = "protector"

    def __init__(self, cfg: ProtectConfig | None = None) -> None:
        self.cfg = cfg or ProtectConfig()

    @abstractmethod
    def protect(self, image: Image.Image) -> ProtectResult:
        """Return a protected copy of `image`. Must not mutate the input."""

    def __call__(self, image: Image.Image) -> ProtectResult:
        return self.protect(image)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(eps={self.cfg.eps_grey_levels}/255, steps={self.cfg.steps})"
