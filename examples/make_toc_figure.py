"""Generate the public-data 1D-to-FDSM overview image used in the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import Normalize
from public_data_workflow import (
    download_public_examples,
    load_mzml_tic,
    load_openspecy_asp,
    resample_uniform,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fdsm import transform  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _relative_intensity(values: np.ndarray) -> np.ndarray:
    shifted = values - np.nanmin(values)
    maximum = np.nanmax(shifted)
    return shifted / maximum if maximum > 0 else shifted


def main() -> None:
    paths = download_public_examples(ROOT / "examples" / "data" / "raw")
    tic_time, tic_intensity = load_mzml_tic(paths["gcms"])
    tic_time, tic_intensity = resample_uniform(tic_time, tic_intensity)
    ftir_wavenumber, ftir_intensity = load_openspecy_asp(paths["ftir"])

    settings = dict(min_order=0, max_order=2, order_step=0.125, sg_window=11)
    tic_map = transform(tic_intensity, x=tic_time, **settings)
    ftir_map = transform(ftir_intensity, x=ftir_wavenumber, **settings)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
        }
    )
    figure = plt.figure(figsize=(12, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, width_ratios=(1.0, 0.12, 1.55, 0.055))
    raw_axes = [figure.add_subplot(grid[row, 0]) for row in range(2)]
    map_axes = [figure.add_subplot(grid[row, 2]) for row in range(2)]
    arrow_axes = [figure.add_subplot(grid[row, 1]) for row in range(2)]
    color_axis = figure.add_subplot(grid[:, 3])

    blue, red = "#176B87", "#B33F62"
    raw_axes[0].plot(tic_time, _relative_intensity(tic_intensity), color=blue, linewidth=0.8)
    raw_axes[0].fill_between(
        tic_time, _relative_intensity(tic_intensity), color=blue, alpha=0.13, linewidth=0
    )
    raw_axes[0].set(title="GC–MS total ion chromatogram", xlabel="Time (min)")

    raw_axes[1].plot(
        ftir_wavenumber, _relative_intensity(ftir_intensity), color=red, linewidth=0.8
    )
    raw_axes[1].set(title="FT–IR spectrum", xlabel="Wavenumber (cm⁻¹)")
    raw_axes[1].invert_xaxis()
    for axis in raw_axes:
        axis.set_ylabel("Relative intensity")
        axis.set_ylim(-0.03, 1.04)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D8DEE6", linewidth=0.5, alpha=0.8)

    norm = Normalize(vmin=-3, vmax=3)
    images = []
    for axis, result, x_label, descending in (
        (map_axes[0], tic_map, "Time (min)", False),
        (map_axes[1], ftir_map, "Wavenumber (cm⁻¹)", True),
    ):
        image = axis.imshow(
            result.map,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="RdBu_r",
            norm=norm,
            extent=(result.x[0], result.x[-1], result.orders[0], result.orders[-1]),
        )
        images.append(image)
        axis.set(xlabel=x_label, ylabel="Derivative order", yticks=[0, 0.5, 1, 1.5, 2])
        if descending:
            axis.invert_xaxis()
    map_axes[0].set_title("Fractional derivative spectrum map (FDSM)")
    map_axes[1].set_title("Fractional derivative spectrum map (FDSM)")

    for axis in arrow_axes:
        axis.set_axis_off()
        axis.annotate(
            "",
            xy=(0.96, 0.5),
            xytext=(0.04, 0.5),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": "#243447", "lw": 2.2},
        )

    colorbar = figure.colorbar(images[0], cax=color_axis)
    colorbar.set_label("Row-wise z-score")
    figure.suptitle(
        "Continuous unfolding of 1D analytical signals into 2D FDSM representations",
        fontsize=15,
        fontweight="bold",
    )
    output_dir = ROOT / "docs" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "fdsm_1d_to_2d.png"
    figure.savefig(output, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output}")
    print(
        "GC-MS:",
        tic_intensity.size,
        "samples ->",
        tic_map.map.shape,
        f"(uniform Δ={tic_map.metadata['delta']:.8g} min)",
    )
    print(
        "FT-IR:",
        ftir_intensity.size,
        "samples ->",
        ftir_map.map.shape,
        f"(Δ={ftir_map.metadata['delta']:.8g} cm^-1)",
    )


if __name__ == "__main__":
    main()
