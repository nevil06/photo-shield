"""Protection strategies.

Phase 1 ships the interface (`base.py`); Phase 2 adds `pgd_engine.py` +
`losses.py`; Phases 3-4 add the identity term and EOT transforms.
"""

from .base import ProtectResult, Protector
from .pgd_engine import PGDProtector

__all__ = ["Protector", "ProtectResult", "PGDProtector"]
