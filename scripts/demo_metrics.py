from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import pyiqa


# ----------------------------
# IO
# ----------------------------


def _read_rgb_tensor(path: Path, *, device: torch.device) -> torch.Tensor:
    """
    Read an RGB image and return a float32 tensor in [0, 1] with shape [1, 3, H, W].
    """
    # Use numpy as the bridge from PIL -> Tensor. (pyiqa/torch stack typically already depends on numpy.)
    import numpy as np  # noqa: F401

    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im)  # (H, W, 3), uint8
        x = torch.from_numpy(arr).to(torch.float32)

    # (H, W, 3) -> (1, 3, H, W)
    x = x.permute(2, 0, 1).unsqueeze(0) / 255.0
    return x.to(device)


# ----------------------------
# Pairing (GT/pred order alignment)
# ----------------------------


def _require_dir(p: Path, desc: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{desc} does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"{desc} is not a directory: {p}")


def _extract_frame_id(path: Path) -> int:
    """
    Extract frame index from filename (use the last digit group).

    Examples:
      - frame.0081.color.jpg -> 81
      - view_0000.png       -> 0
      - 000123.jpg          -> 123
    """
    m = re.findall(r"(\d+)", path.stem)
    if not m:
        raise ValueError(f"Cannot extract frame id from filename: {path.name}")
    return int(m[-1])


def _list_images(dir_path: Path) -> list[Path]:
    exts = ("png", "jpg", "jpeg")
    files: list[Path] = []
    for e in exts:
        files.extend(dir_path.glob(f"*.{e}"))
    return files


def _list_images_with_ids(dir_path: Path) -> list[tuple[int, Path]]:
    files = _list_images(dir_path)
    if not files:
        raise FileNotFoundError(
            f"No RGB images found under: {dir_path} (searched extensions: png/jpg/jpeg)"
        )
    items = [(_extract_frame_id(p), p) for p in files]
    items.sort(key=lambda t: t[0])
    return items


# ----------------------------
# Metrics (IQA-PyTorch / pyiqa)
# ----------------------------


@torch.no_grad()
def evaluate_rgb_dirs(gt_rgb_dir: Path, pred_rgb_dir: Path) -> dict:
    """
    Evaluate average PSNR / SSIM / LPIPS over two directories using IQA-PyTorch (pyiqa).

    Pairing rule:
      - extract last integer from filenames
      - sort GT and pred separately by that integer
      - zip by order
    This supports GT starting at e.g. 81 while pred starts at 0, as long as counts match and order is consistent.
    """
    _require_dir(gt_rgb_dir, "GT RGB dir")
    _require_dir(pred_rgb_dir, "Pred RGB dir")

    gt_items = _list_images_with_ids(gt_rgb_dir)
    pred_items = _list_images_with_ids(pred_rgb_dir)

    if len(gt_items) != len(pred_items):
        raise FileNotFoundError(
            "GT/pred frame count mismatch: "
            f"gt={len(gt_items)} pred={len(pred_items)}. "
            "If GT and pred are aligned by order, they must contain the same number of images."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    psnr_metric = pyiqa.create_metric("psnr", device=device).eval()
    ssim_metric = pyiqa.create_metric("ssim", device=device).eval()
    lpips_metric = pyiqa.create_metric("lpips", device=device).eval()

    psnr_list: list[float] = []
    ssim_list: list[float] = []
    lpips_list: list[float] = []

    def _resize_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # x: [1,3,H,W] in [0,1]
        if x.shape[-2:] == (h, w):
            return x
        return F.interpolate(x, size=(h, w), mode="area")

    for (gt_id, gt_path), (pred_id, pred_path) in zip(gt_items, pred_items):
        gt = _read_rgb_tensor(gt_path, device=device)
        pred = _read_rgb_tensor(pred_path, device=device)

        if gt.shape != pred.shape:
            # unify to smaller spatial size to avoid upsampling artifacts
            tgt_h = min(gt.shape[-2], pred.shape[-2])
            tgt_w = min(gt.shape[-1], pred.shape[-1])
            gt = _resize_to(gt, tgt_h, tgt_w)
            pred = _resize_to(pred, tgt_h, tgt_w)

        psnr_list.append(float(psnr_metric(pred, gt).item()))
        ssim_list.append(float(ssim_metric(pred, gt).item()))
        lpips_list.append(float(lpips_metric(pred, gt).item()))

    # Use torch for mean (avoid numpy dependency here)
    psnr_mean = torch.tensor(psnr_list).mean().item() if psnr_list else float("nan")
    ssim_mean = torch.tensor(ssim_list).mean().item() if ssim_list else float("nan")
    lpips_mean = torch.tensor(lpips_list).mean().item() if lpips_list else float("nan")

    return {
        "PSNR": float(psnr_mean),
        "SSIM": float(ssim_mean),
        "LPIPS": float(lpips_mean),
        "num_frames": int(len(psnr_list)),
    }


# ----------------------------
# CLI
# ----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Demo RGB metrics script (IQA-PyTorch/pyiqa: PSNR/SSIM/LPIPS).")
    p.add_argument("--gt-rgb-dir", type=Path, required=True, help="GT RGB directory (png/jpg/jpeg)")
    p.add_argument("--pred-rgb-dir", type=Path, required=True, help="Pred RGB directory (png/jpg/jpeg)")
    p.add_argument("--out", type=Path, default=None, help="Optional output json path")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    scores = evaluate_rgb_dirs(args.gt_rgb_dir, args.pred_rgb_dir)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(f"Wrote: {args.out}")

    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
