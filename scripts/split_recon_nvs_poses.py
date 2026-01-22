#!/usr/bin/env python3
"""
Split dumped camera poses NPZ into two chunks: first m frames (recon) and the rest (nvs).

Given an input NPZ produced by `dump_camera_poses_npz(...)` (expected keys:
- extrinsics_w2c: (N,4,4)
- intrinsics: (N,3,3)
- frame_ids: (N,)
and possibly other keys),

this script writes:
- recon_poses.npz: first m views (default m=80)
- nvs_poses.npz: remaining (N-m) views

It will:
- Split all arrays whose first dimension == N using the same index split
  (e.g. extrinsics_w2c_aligned if present).
- Keep arrays that don't match (N, ...) as-is (e.g. image_size_hw, hf_alignment_world_to_gltf).

Usage:
  python3 scripts/split_recon_nvs_poses.py \
    --input-npz /path/to/poses.npz \
    --split-m 80 \
    --output-dir /path/to/out_dir

This writes:
  /path/to/out_dir/recon_poses.npz
  /path/to/out_dir/nvs_poses.npz

Optional:
  --output-npz /path/to/nvs_poses.npz   # override only the NVS output path

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split poses.npz into recon (first m) and nvs (rest).")
    p.add_argument("--input-npz", type=str, required=True, help="Input poses.npz path")
    p.add_argument(
        "--split-m",
        type=int,
        default=80,
        help="Number of first frames to put into recon_poses.npz (default: 80).",
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write recon_poses.npz and nvs_poses.npz. If omitted, uses input-npz parent dir.",
    )

    # Backward-compatible: --output-npz is treated as the NVS output file path.
    p.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help="(Deprecated) Output NPZ path for the NVS subset. Use --output-nvs-npz instead.",
    )
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    in_path = Path(args.input_npz).expanduser().resolve()

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else in_path.parent

    recon_path = out_dir / "recon_poses.npz"

    nvs_path = Path(args.output_npz).expanduser().resolve() if args.output_npz else out_dir / "nvs_poses.npz"

    return in_path, recon_path, nvs_path


def _split_npz_dict(
    data: np.lib.npyio.NpzFile, keys: list[str], N: int, m: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    recon: Dict[str, Any] = {}
    nvs: Dict[str, Any] = {}

    for k in keys:
        v = data[k]
        # Per-view arrays: (N, ...)
        if isinstance(v, np.ndarray) and v.shape[:1] == (N,):
            recon[k] = v[:m]
            nvs[k] = v[m:]
        else:
            # Meta / global arrays preserved in both outputs
            recon[k] = v
            nvs[k] = v

    return recon, nvs


def main() -> None:
    args = parse_args()
    in_path, recon_path, nvs_path = _resolve_paths(args)

    if not in_path.exists():
        raise FileNotFoundError(f"input npz not found: {in_path}")

    m = int(args.split_m)
    if m < 0:
        raise ValueError(f"--split-m must be >= 0, got {m}")

    data = np.load(in_path, allow_pickle=False)
    keys = list(data.keys())

    if "frame_ids" not in data:
        raise KeyError(f"'frame_ids' not found in {in_path}. Keys: {keys}")

    frame_ids = np.asarray(data["frame_ids"])
    if frame_ids.ndim != 1:
        raise ValueError(f"frame_ids must be 1D, got shape {frame_ids.shape}")

    N = int(frame_ids.shape[0])
    m = min(m, N)

    recon_out, nvs_out = _split_npz_dict(data, keys, N, m)

    recon_path.parent.mkdir(parents=True, exist_ok=True)
    nvs_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(recon_path, **recon_out)
    np.savez_compressed(nvs_path, **nvs_out)

    print(
        f"Saved recon (first {m}): {recon_path}\n"
        f"Saved nvs (rest {N - m}): {nvs_path}\n"
        f"Input views: {N}, recon: {m}, nvs: {N - m}\n"
        f"Keys: {keys}"
    )


if __name__ == "__main__":
    main()
