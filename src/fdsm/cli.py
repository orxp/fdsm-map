"""Command-line interface for transforming text and CSV signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import transform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fdsm",
        description="Transform a uniformly sampled 1D signal into an FDSM map.",
    )
    parser.add_argument("input", type=Path, help="Input text or CSV file")
    parser.add_argument("output", type=Path, help="Output .npz archive")
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Input delimiter; use 'whitespace' for any whitespace",
    )
    parser.add_argument(
        "--x-column", type=int, default=None, help="Optional zero-based axis column"
    )
    parser.add_argument(
        "--y-column", type=int, default=0, help="Zero-based signal column (default: 0)"
    )
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--min-order", type=float, default=0.0)
    parser.add_argument("--max-order", type=float, default=2.0)
    parser.add_argument("--order-step", type=float, default=0.25)
    parser.add_argument("--sg-window", type=int, default=11)
    parser.add_argument("--sg-polyorder", type=int, default=3)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--pad-ratio", type=float, default=0.25)
    parser.add_argument("--crop-ratio", type=float, default=0.05)
    parser.add_argument("--nonuniform", choices=["raise", "warn", "ignore"], default="raise")
    parser.add_argument("--nyquist", choices=["retain", "zero_unpaired"], default="retain")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    delimiter = None if args.delimiter == "whitespace" else args.delimiter
    data = np.loadtxt(args.input, delimiter=delimiter, skiprows=args.skiprows, ndmin=2)
    try:
        signal = data[:, args.y_column]
        axis = None if args.x_column is None else data[:, args.x_column]
    except IndexError as exc:
        raise SystemExit(f"requested column is not present in {args.input}") from exc

    result = transform(
        signal,
        x=axis,
        min_order=args.min_order,
        max_order=args.max_order,
        order_step=args.order_step,
        sg_window=None if args.no_smoothing else args.sg_window,
        sg_polyorder=args.sg_polyorder,
        pad_ratio=args.pad_ratio,
        crop_ratio=args.crop_ratio,
        nonuniform=args.nonuniform,
        nyquist=args.nyquist,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        map=result.map,
        x=result.x,
        orders=result.orders,
        metadata=np.asarray(json.dumps(result.metadata, sort_keys=True)),
    )
    print(
        f"saved {result.map.shape} {result.map.dtype} FDSM to {args.output} "
        f"(delta={result.metadata['delta']:.8g})"
    )
    return 0
