"""Single source of truth for paths, defaults and the attack budget.

Why a module instead of constants scattered around: every phase prints numbers
(eps, steps, size). If they live in one frozen dataclass, an experiment log is
just `repr(cfg)` and a viva answer is "here is the exact config I ran".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SAMPLES_DIR = ROOT / "assets" / "samples"
WEIGHTS_DIR = ROOT / "weights"

# ---------------------------------------------------------------- budgets
# Perturbation is measured in [0, 1] pixel units (we rescale uint8 by 255).
# eps = 4/255 means: no pixel moves by more than 4 grey levels.
EPS_4_255 = 4 / 255
EPS_8_255 = 8 / 255


def resolve_device(pref: str = "auto") -> str:
    """'auto' -> cuda if present, else mps, else cpu. Never raises."""
    if pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class ProtectConfig:
    """Knobs for one protection run (Phase 2+ actually uses them)."""

    eps: float = EPS_4_255  # L-inf budget, [0,1] units; 4/255 .. 8/255 is the sweet spot
    steps: int = 50  # PGD iterations; 50 is a good CPU/GPU compromise
    step_size: float | None = None  # None -> 2.5 * eps / steps
    size: int = 512  # working resolution: long side is resized to this
    seed: int = 0  # torch/numpy seed for reproducible noise + EOT sampling
    device: str = "auto"
    # --- objective weights (the a and c of the corrected loss) ---
    # Measured in Phase 2 at 256 px: w_lpips=0 gives LPIPS 0.143, w_lpips=2
    # gives 0.0062 while keeping 96% of the latent shift. Do not set to 0
    # without re-reading docs_private/05_results_and_limits.md.
    w_latent: float = 1.0
    w_lpips: float = 2.0
    # Phase 3: weight of the identity term. 0.0 = off (default until measured).
    # Needs a detected face + the ArcFace weights; the engine raises clearly if
    # it is non-zero and either is missing.
    w_identity: float = 0.0
    # Phase 4: EOT sample count. 0 = off (no augmentations). 4 is the practical
    # default: enough to smooth the gradient without quadrupling runtime.
    # Recommended range: 4-8. Above 8 the marginal gain is diminishing.
    eot_k: int = 0
    # Latent objective: "none" = untargeted (push away from E(x)); "gray" /
    # "pattern" = targeted (pull toward E(target)), as in PhotoGuard / Mist.
    latent_target: str = "none"

    def __post_init__(self) -> None:
        if self.latent_target not in ("none", "gray", "pattern"):
            raise ValueError("latent_target must be one of none, gray, pattern")
        if not 0 < self.eps <= 32 / 255:
            raise ValueError("eps must be in (0, 32/255]; larger just means visible")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.size < 64:
            raise ValueError("size must be >= 64 (VAE needs /8, LPIPS needs >=64)")
        if self.w_latent < 0 or self.w_lpips < 0 or self.w_identity < 0:
            raise ValueError("loss weights must be >= 0")
        if self.eot_k < 0:
            raise ValueError("eot_k must be >= 0")

    @property
    def alpha(self) -> float:
        """Gradient step length. Heuristic 2.5*eps/steps, clamped by eps."""
        return self.step_size if self.step_size is not None else 2.5 * self.eps / self.steps

    @property
    def eps_grey_levels(self) -> int:
        """eps expressed in uint8 grey levels - the number a human understands."""
        return round(self.eps * 255)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["alpha"] = self.alpha
        return d
