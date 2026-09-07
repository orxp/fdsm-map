"""Download and parse the public signals used by the FDSM gallery example.

The raw files are intentionally excluded from Git. Downloads are pinned by
SHA-256 so that a changed upstream file fails visibly instead of silently
altering the example figure.
"""

from __future__ import annotations

import base64
import hashlib
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PublicFile:
    filename: str
    url: str
    sha256: str


GCMS_FILE = PublicFile(
    filename="EFWS-1.mzML",
    url=(
        "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/"
        "MTBLS11550/FILES/DERIVED_FILES/GC/EFWS-1.mzML"
    ),
    sha256="f3ead5cdae2ed4be96e8b5b8514da3b451ac7787f54c3fcebd5aab28e3727b2d",
)
FTIR_FILE = PublicFile(
    filename="ftir_ldpe_soil.asp",
    url=(
        "https://raw.githubusercontent.com/wincowgerDEV/OpenSpecy-package/"
        "main/inst/extdata/ftir_ldpe_soil.asp"
    ),
    sha256="96d2eea61cdc3ba8f675fde1566388949c2f43689b4e3eb6d3d41457df015547",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_public_examples(raw_dir: Path) -> dict[str, Path]:
    """Download missing example files and verify every file against its hash."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for label, item in (("gcms", GCMS_FILE), ("ftir", FTIR_FILE)):
        destination = raw_dir / item.filename
        if not destination.exists():
            partial = destination.with_suffix(destination.suffix + ".part")
            urllib.request.urlretrieve(item.url, partial)  # noqa: S310
            partial.replace(destination)
        actual_hash = _sha256(destination)
        if actual_hash.lower() != item.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {destination}: expected {item.sha256}, "
                f"received {actual_hash}"
            )
        paths[label] = destination
    return paths


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _decode_binary_array(element) -> tuple[str | None, str | None, NDArray[np.float64]]:
    accessions: set[str] = set()
    unit_name: str | None = None
    encoded = ""
    for child in element.iter():
        name = _local_name(child.tag)
        if name == "cvParam":
            accessions.add(child.attrib.get("accession", ""))
            if child.attrib.get("unitName"):
                unit_name = child.attrib["unitName"]
        elif name == "binary":
            encoded = child.text or ""

    if "MS:1000595" in accessions:
        array_type = "time"
    elif "MS:1000515" in accessions:
        array_type = "intensity"
    else:
        array_type = None

    if "MS:1000523" in accessions:
        dtype = np.dtype("f8")
    elif "MS:1000521" in accessions:
        dtype = np.dtype("f4")
    else:
        raise ValueError("mzML binary array lacks a supported 32/64-bit float declaration")

    if "MS:1000140" in accessions:
        dtype = dtype.newbyteorder(">")
    else:
        dtype = dtype.newbyteorder("<")

    payload = base64.b64decode(encoded)
    if "MS:1000574" in accessions:
        payload = zlib.decompress(payload)
    values = np.frombuffer(payload, dtype=dtype).astype(np.float64, copy=False)
    return array_type, unit_name, values


def load_mzml_tic(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load the total ion chromatogram from an mzML file.

    Returns time in minutes and detector intensity. The example parser supports
    ordinary base64 arrays with optional zlib compression; it is deliberately
    small and is not intended to replace a complete mzML library.
    """
    import xml.etree.ElementTree as et

    for _, element in et.iterparse(path, events=("end",)):
        if _local_name(element.tag) != "chromatogram":
            continue
        is_tic = element.attrib.get("id") == "TIC" or any(
            child.attrib.get("accession") == "MS:1000235"
            for child in element.iter()
            if _local_name(child.tag) == "cvParam"
        )
        if not is_tic:
            element.clear()
            continue

        arrays: dict[str, NDArray[np.float64]] = {}
        time_unit: str | None = None
        for child in element.iter():
            if _local_name(child.tag) != "binaryDataArray":
                continue
            array_type, unit_name, values = _decode_binary_array(child)
            if array_type is not None:
                arrays[array_type] = values
            if array_type == "time":
                time_unit = unit_name
        if {"time", "intensity"} - arrays.keys():
            raise ValueError("TIC chromatogram is missing its time or intensity array")
        time = arrays["time"]
        intensity = arrays["intensity"]
        if time.size != intensity.size or time.size < 2:
            raise ValueError("TIC arrays must have equal lengths of at least two")
        if time_unit and "second" in time_unit.lower():
            time = time / 60.0
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(intensity)):
            raise ValueError("TIC contains non-finite values")
        return time, intensity
    raise ValueError("No total ion chromatogram was found in the mzML file")


def load_openspecy_asp(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load the simple Galactic ASP spectrum bundled with OpenSpecy."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(lines) < 7:
        raise ValueError("ASP file is too short")
    count = int(lines[0])
    start, stop = float(lines[1]), float(lines[2])
    intensity = np.asarray([float(value) for value in lines[6 : 6 + count]], dtype=np.float64)
    if intensity.size != count:
        raise ValueError(f"ASP declares {count} points but contains {intensity.size}")
    axis = np.linspace(start, stop, count, dtype=np.float64)
    if not np.all(np.isfinite(intensity)):
        raise ValueError("ASP spectrum contains non-finite values")
    return axis, intensity


def resample_uniform(
    x: NDArray[np.float64], y: NDArray[np.float64], *, n_points: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Linearly resample a strictly monotonic signal onto a uniform grid."""
    axis = np.asarray(x, dtype=np.float64)
    signal = np.asarray(y, dtype=np.float64)
    if axis.ndim != 1 or signal.ndim != 1 or axis.size != signal.size or axis.size < 2:
        raise ValueError("x and y must be equal-length 1D arrays with at least two points")
    differences = np.diff(axis)
    if np.all(differences < 0):
        axis, signal = axis[::-1], signal[::-1]
    elif not np.all(differences > 0):
        raise ValueError("x must be strictly monotonic")
    count = axis.size if n_points is None else int(n_points)
    if count < 2:
        raise ValueError("n_points must be at least two")
    uniform_axis = np.linspace(axis[0], axis[-1], count, dtype=np.float64)
    return uniform_axis, np.interp(uniform_axis, axis, signal)
