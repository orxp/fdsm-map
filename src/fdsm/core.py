"""Domain-independent fractional derivative spectrum map construction."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.signal import savgol_filter

from .result import FDSMResult
from .validation import (
    NonuniformPolicy,
    make_orders,
    prepare_axis_and_signal,
    validate_orders,
    validate_savgol,
    validate_transform_parameters,
)

DEFAULT_MIN_ORDER = 0.0
DEFAULT_MAX_ORDER = 2.0
DEFAULT_ORDER_STEP = 0.25
DEFAULT_SG_WINDOW = 11
DEFAULT_SG_POLYORDER = 3
DEFAULT_PAD_RATIO = 0.25
DEFAULT_CROP_RATIO = 0.05
DEFAULT_EPSILON = 1e-8

Normalization = Literal["per_order"] | None
NyquistPolicy = Literal["retain", "zero_unpaired"]


def _crop_count(length: int, crop_ratio: float) -> int:
    count = int(np.floor(length * crop_ratio))
    if 2 * count >= length:
        raise ValueError("crop_ratio is too large; no signal samples remain")
    return count


def _crop(values: NDArray, count: int) -> NDArray:
    return values.copy() if count == 0 else values[count:-count]


def _zscore(values: NDArray[np.float64], epsilon: float) -> NDArray[np.float64]:
    return (values - np.mean(values)) / (np.std(values) + epsilon)


def fractional_derivative_fft(
    y: ArrayLike,
    *,
    delta: float,
    order: float,
    pad_ratio: float = DEFAULT_PAD_RATIO,
    nyquist: NyquistPolicy = "retain",
) -> NDArray[np.float64]:
    """Calculate a real fractional derivative on a uniform grid using the FFT.

    The principal-branch multiplier ``(i 2 pi f)**order`` is used. ``retain``
    keeps the unpaired Nyquist coefficient and returns the inverse transform's
    real component. ``zero_unpaired`` removes that coefficient when its
    multiplier is complex.
    """
    raw_signal = np.asarray(y)
    if np.iscomplexobj(raw_signal):
        raise ValueError("y must be real-valued")
    signal = np.asarray(raw_signal, dtype=np.float64)
    if signal.ndim != 1 or signal.size < 2:
        raise ValueError("y must be a one-dimensional signal with at least two samples")
    if not np.all(np.isfinite(signal)):
        raise ValueError("y must contain only finite values")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    if not np.isfinite(order) or order < 0:
        raise ValueError("order must be finite and nonnegative")
    if pad_ratio < 0:
        raise ValueError("pad_ratio must be nonnegative")
    if nyquist not in {"retain", "zero_unpaired"}:
        raise ValueError("nyquist must be 'retain' or 'zero_unpaired'")

    pad = int(np.ceil(signal.size * pad_ratio))
    padded = np.pad(signal, pad_width=pad, mode="reflect") if pad else signal
    frequencies = np.fft.fftfreq(padded.size, d=delta)
    spectrum = np.fft.fft(padded)
    multiplier = (1j * 2.0 * np.pi * frequencies) ** float(order)
    if order > 0:
        multiplier[0] = 0.0

    if nyquist == "zero_unpaired" and padded.size % 2 == 0:
        nyquist_index = padded.size // 2
        nyquist_multiplier = multiplier[nyquist_index]
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(nyquist_multiplier))
        if abs(nyquist_multiplier.imag) > tolerance:
            multiplier[nyquist_index] = 0.0

    derivative = np.fft.ifft(spectrum * multiplier)
    if pad:
        derivative = derivative[pad:-pad]
    return np.real(derivative)


def transform(
    y: ArrayLike,
    *,
    x: ArrayLike | None = None,
    min_order: float = DEFAULT_MIN_ORDER,
    max_order: float = DEFAULT_MAX_ORDER,
    order_step: float = DEFAULT_ORDER_STEP,
    orders: ArrayLike | None = None,
    sg_window: int | None = DEFAULT_SG_WINDOW,
    sg_polyorder: int = DEFAULT_SG_POLYORDER,
    pad_ratio: float = DEFAULT_PAD_RATIO,
    crop_ratio: float = DEFAULT_CROP_RATIO,
    normalize: Normalization = "per_order",
    scale_input: bool = True,
    epsilon: float = DEFAULT_EPSILON,
    nonuniform: NonuniformPolicy = "raise",
    uniformity_rtol: float = 1e-5,
    uniformity_atol: float = 1e-12,
    nyquist: NyquistPolicy = "retain",
    output_dtype: DTypeLike = np.float32,
) -> FDSMResult:
    """Transform one uniformly sampled 1D signal into a 2D FDSM.

    Pass either the inclusive ``min_order``/``max_order``/``order_step``
    specification or an explicit strictly increasing ``orders`` array.
    """
    validate_transform_parameters(
        pad_ratio=pad_ratio,
        crop_ratio=crop_ratio,
        epsilon=epsilon,
        uniformity_rtol=uniformity_rtol,
        uniformity_atol=uniformity_atol,
        nonuniform=nonuniform,
        nyquist=nyquist,
    )
    if normalize not in {"per_order", None}:
        raise ValueError("normalize must be 'per_order' or None")
    dtype = np.dtype(output_dtype)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError("output_dtype must be a floating-point dtype")

    order_values = (
        make_orders(min_order, max_order, order_step)
        if orders is None
        else validate_orders(orders)
    )
    axis, signal, delta, reversed_axis, irregularity = prepare_axis_and_signal(
        y,
        x,
        nonuniform=nonuniform,
        uniformity_rtol=uniformity_rtol,
        uniformity_atol=uniformity_atol,
    )
    validate_savgol(signal.size, sg_window, sg_polyorder)

    smoothed = (
        signal.copy()
        if sg_window is None
        else savgol_filter(signal, window_length=sg_window, polyorder=sg_polyorder, mode="interp")
    )
    crop_count = _crop_count(signal.size, crop_ratio)
    retained_smoothed = _crop(smoothed, crop_count)

    if scale_input:
        working = (smoothed - np.mean(retained_smoothed)) / (
            np.std(retained_smoothed) + epsilon
        )
    else:
        working = smoothed

    rows = []
    for order in order_values:
        row = fractional_derivative_fft(
            working,
            delta=delta,
            order=float(order),
            pad_ratio=pad_ratio,
            nyquist=nyquist,
        )
        row = _crop(row, crop_count)
        if normalize == "per_order":
            row = _zscore(row, epsilon)
        rows.append(row)

    result_map = np.stack(rows, axis=0).astype(dtype, copy=False)
    output_axis = _crop(axis, crop_count).astype(np.float64, copy=False)
    metadata = {
        "n_input_samples": int(signal.size),
        "n_output_samples": int(output_axis.size),
        "n_orders": int(order_values.size),
        "delta": delta,
        "axis_reversed": reversed_axis,
        "relative_spacing_deviation": irregularity,
        "nonuniform_policy": nonuniform,
        "sg_window": sg_window,
        "sg_polyorder": sg_polyorder if sg_window is not None else None,
        "pad_ratio": float(pad_ratio),
        "crop_ratio": float(crop_ratio),
        "crop_samples_per_edge": crop_count,
        "scale_input": bool(scale_input),
        "normalization": normalize,
        "epsilon": float(epsilon),
        "nyquist_policy": nyquist,
        "output_dtype": str(result_map.dtype),
    }
    return FDSMResult(map=result_map, x=output_axis, orders=order_values, metadata=metadata)


def transform_batch(
    y: ArrayLike,
    *,
    x: ArrayLike | None = None,
    **kwargs,
) -> FDSMResult:
    """Transform a batch with shape ``(N, L)`` using one shared coordinate axis."""
    signals = np.asarray(y, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[0] == 0:
        raise ValueError("batch y must have shape (n_signals, n_samples)")

    results = [transform(signal, x=x, **kwargs) for signal in signals]
    first = results[0]
    maps = np.stack([result.map for result in results], axis=0)
    metadata = dict(first.metadata)
    metadata["n_signals"] = int(signals.shape[0])
    metadata["axis_reversed"] = bool(first.metadata["axis_reversed"])
    return FDSMResult(map=maps, x=first.x, orders=first.orders, metadata=metadata)
