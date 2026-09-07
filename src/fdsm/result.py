"""Result containers for FDSM transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FDSMResult:
    """Numerical FDSM output and the coordinates needed to interpret it.

    ``map`` has shape ``(H, W)`` for a single signal and ``(N, H, W)`` for a
    batch. ``orders`` has length ``H`` and ``x`` has length ``W``.
    """

    map: NDArray[np.floating]
    x: NDArray[np.float64]
    orders: NDArray[np.float64]
    metadata: dict[str, Any]

