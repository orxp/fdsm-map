"""Input and parameter validation for FDSM transforms."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

NonuniformPolicy = Literal["raise", "warn", "ignore"]


def make_orders(
    min_order: float = 0.0,
    max_order: float = 2.0,
    order_step: float = 0.25,
) -> NDArray[np.float64]:
    """Construct an inclusive, uniformly spaced derivative-order ladder."""
    values = np.asarray([min_order, max_order, order_step], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("min_order, max_order, and order_step must be finite")
    if min_order < 0:
        raise ValueError("min_order must be nonnegative")
    if max_order < min_order:
        raise ValueError("max_order must be greater than or equal to min_order")
    if order_step <= 0:
        raise ValueError("order_step must be positive")

    intervals_float = (max_order - min_order) / order_step
    intervals = int(round(intervals_float))
    if not np.isclose(intervals_float, intervals, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "order_step must divide the requested order range so both endpoints are included; "
            "pass an explicit orders array for a nonuniform ladder"
        )
    return np.linspace(min_order, max_order, intervals + 1, dtype=np.float64)


def validate_orders(orders: ArrayLike) -> NDArray[np.float64]:
    """Validate and return a one-dimensional derivative-order array."""
    array = np.asarray(orders, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("orders must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("orders must contain only finite values")
    if np.any(array < 0):
        raise ValueError("orders must be nonnegative")
    if np.any(np.diff(array) <= 0):
        raise ValueError("orders must be strictly increasing")
    return array


def validate_savgol(length: int, window: int | None, polyorder: int) -> None:
    """Validate Savitzky-Golay settings before SciPy is called."""
    if window is None:
        return
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise TypeError("sg_window must be an odd integer or None")
    if window < 3 or window % 2 == 0:
        raise ValueError("sg_window must be an odd integer of at least 3")
    if window > length:
        raise ValueError("sg_window cannot exceed the signal length")
    if isinstance(polyorder, bool) or not isinstance(polyorder, (int, np.integer)):
        raise TypeError("sg_polyorder must be an integer")
    if polyorder < 0 or polyorder >= window:
        raise ValueError("sg_polyorder must satisfy 0 <= sg_polyorder < sg_window")


def prepare_axis_and_signal(
    y: ArrayLike,
    x: ArrayLike | None,
    *,
    nonuniform: NonuniformPolicy,
    uniformity_rtol: float,
    uniformity_atol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, bool, float]:
    """Validate a single signal and return an ascending uniform coordinate."""
    raw_signal = np.asarray(y)
    if np.iscomplexobj(raw_signal):
        raise ValueError("y must be real-valued")
    signal = np.asarray(raw_signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("y must be a one-dimensional signal")
    if signal.size < 3:
        raise ValueError("y must contain at least three samples")
    if not np.all(np.isfinite(signal)):
        raise ValueError("y must contain only finite values")

    if x is None:
        axis = np.arange(signal.size, dtype=np.float64)
    else:
        raw_axis = np.asarray(x)
        if np.iscomplexobj(raw_axis):
            raise ValueError("x must be real-valued")
        axis = np.asarray(raw_axis, dtype=np.float64)
        if axis.ndim != 1 or axis.size != signal.size:
            raise ValueError("x must be one-dimensional and have the same length as y")
        if not np.all(np.isfinite(axis)):
            raise ValueError("x must contain only finite values")

    spacing = np.diff(axis)
    if np.all(spacing > 0):
        reversed_axis = False
    elif np.all(spacing < 0):
        axis = axis[::-1].copy()
        signal = signal[::-1].copy()
        spacing = np.diff(axis)
        reversed_axis = True
    else:
        raise ValueError("x must be strictly monotonic")

    delta = float(np.median(spacing))
    max_deviation = float(np.max(np.abs(spacing - delta)))
    allowed = float(uniformity_atol + uniformity_rtol * abs(delta))
    relative_irregularity = max_deviation / max(abs(delta), np.finfo(float).tiny)

    if max_deviation > allowed:
        message = (
            "x must be uniformly spaced for FFT-based differentiation "
            f"(maximum relative spacing deviation={relative_irregularity:.3g}); "
            "resample explicitly or set nonuniform='warn'/'ignore' deliberately"
        )
        if nonuniform == "raise":
            raise ValueError(message)
        if nonuniform == "warn":
            warnings.warn(
                message + "; proceeding with the median spacing",
                UserWarning,
                stacklevel=2,
            )

    return axis, signal, delta, reversed_axis, relative_irregularity


def validate_transform_parameters(
    *,
    pad_ratio: float,
    crop_ratio: float,
    epsilon: float,
    uniformity_rtol: float,
    uniformity_atol: float,
    nonuniform: str,
    nyquist: str,
) -> None:
    """Validate scalar transform policies."""
    if not 0 <= pad_ratio:
        raise ValueError("pad_ratio must be nonnegative")
    if not 0 <= crop_ratio < 0.5:
        raise ValueError("crop_ratio must satisfy 0 <= crop_ratio < 0.5")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if uniformity_rtol < 0 or uniformity_atol < 0:
        raise ValueError("uniformity tolerances must be nonnegative")
    if nonuniform not in {"raise", "warn", "ignore"}:
        raise ValueError("nonuniform must be 'raise', 'warn', or 'ignore'")
    if nyquist not in {"retain", "zero_unpaired"}:
        raise ValueError("nyquist must be 'retain' or 'zero_unpaired'")
