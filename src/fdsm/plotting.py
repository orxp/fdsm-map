"""Optional Matplotlib visualization for FDSM results."""

from __future__ import annotations

from .result import FDSMResult


def plot_map(result: FDSMResult, *, ax=None, cmap: str = "RdBu_r"):
    """Plot a single-signal FDSM and return ``(figure, axes)``."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError("plot_map requires the optional 'plot' dependency") from exc

    if result.map.ndim != 2:
        raise ValueError("plot_map expects a single-signal result with a two-dimensional map")
    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 4))
    else:
        figure = ax.figure

    image = ax.imshow(
        result.map,
        aspect="auto",
        origin="lower",
        extent=[result.x[0], result.x[-1], result.orders[0], result.orders[-1]],
        cmap=cmap,
    )
    ax.set_xlabel("signal coordinate")
    ax.set_ylabel("derivative order")
    figure.colorbar(image, ax=ax, label="normalized response")
    return figure, ax

