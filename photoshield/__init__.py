"""PhotoShield - imperceptible adversarial protection for *your own* photos.

Phase 1 scope: image I/O (EXIF-safe) + reference-free quality metrics.
Later phases add the PGD protector (protect/), editors, and evaluation.

Nothing here is imported eagerly on purpose: `import photoshield` must stay
cheap so the CLI starts fast and tests do not pull torch/torchvision in.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
