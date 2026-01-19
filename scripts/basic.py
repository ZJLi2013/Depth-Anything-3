import argparse
import glob
import os
from typing import List, Optional

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.camera_trj_helpers import (
    cam_trace_visualization,
    render_novel_view_orbit_path,
)


def gather_images(input_dir: str) -> List[str]:
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    images: List[str] = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(input_dir, ext)))
    return sorted(images)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 basic inference script (no intrinsics/extrinsics inputs)."
    )
    parser.add_argument(
        "--input-images",
        type=str,
        required=True,
        help="Path to input images directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to write outputs (default: ./output)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/dataset/DA3NESTED-GIANT-LARGE",
        help="Path to DA3 model directory (default: /dataset/DA3NESTED-GIANT-LARGE)",
    )
    parser.add_argument(
        "--gs-head",
        action="store_true",
        help="Enable Gaussian Splatting branch (infer_gs=True)",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default=None,
        help=(
            "Export format string passed to DA3 (e.g. glb, gs_views, glb-gs_views). "
            "If omitted: glb when --gs-head is off, otherwise gs_views."
        ),
    )
    parser.add_argument(
        "--cam-trace",
        action="store_true",
        help="Export a camera-only GLB (camera_trace.glb) to visualize camera poses.",
    )
    parser.add_argument(
        "--novel-orbit",
        action="store_true",
        help="Generate novel-view poses and export to --output-dir (novel_orbit_poses.npz + *_trace.glb).",
    )
    parser.add_argument(
        "--novel-orbit-frames",
        "--novel-orbit-num-frames",
        dest="novel_orbit_frames",
        type=int,
        default=120,
        help="Number of novel-view poses to generate when --novel-orbit is set (default: 120)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: cuda/cpu. Default: cuda if available else cpu.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    images = gather_images(args.input_images)
    print(f"Found {len(images)} images in {args.input_images}")
    if len(images) == 0:
        raise ValueError("No images found. Supported formats: png, jpg, jpeg (case-insensitive)")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    export_format = args.export_format
    if export_format is None:
        export_format = "gs_views" if args.gs_head else "glb"

    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device=device)

    prediction = model.inference(
        images,
        extrinsics=None,
        intrinsics=None,
        infer_gs=args.gs_head,
        export_dir=args.output_dir,
        export_format=export_format,
    )

    # Basic prints
    if prediction.processed_images is not None:
        print("Processed images shape:", prediction.processed_images.shape)
    if prediction.depth is not None:
        print("Depth shape:", prediction.depth.shape)
    if prediction.conf is not None:
        print("Confidence shape:", prediction.conf.shape)
    if prediction.extrinsics is not None:
        print("Extrinsics shape:", prediction.extrinsics.shape)
    if prediction.intrinsics is not None:
        print("Intrinsics shape:", prediction.intrinsics.shape)

    # generate novel view poses (export novel-view camera poses)
    if args.novel_orbit:
        if prediction.extrinsics is None or prediction.intrinsics is None:
            print("[WARN] Skip novel orbit: prediction.extrinsics/intrinsics not available.")
        else:
            # Use processed image size if available; fallback to depth size
            if prediction.processed_images is not None:
                H, W = prediction.processed_images.shape[1:3]
            else:
                H, W = prediction.depth.shape[-2:]

            # render_novel_view_orbit_path expects torch tensors
            ex_w2c = torch.from_numpy(prediction.extrinsics).to(device)
            in_k = torch.from_numpy(prediction.intrinsics).to(device)

            novel_w2c_b, novel_k_b = render_novel_view_orbit_path(
                extrinsics_w2c=ex_w2c,
                intrinsics=in_k,
                num_frames=int(args.novel_orbit_frames),
            )

            # Save as npz on CPU
            novel_w2c = novel_w2c_b.squeeze(0).detach().cpu().numpy()  # (T,4,4)
            novel_k = novel_k_b.squeeze(0).detach().cpu().numpy()  # (T,3,3)

            out_npz = os.path.join(args.output_dir, "novel_orbit_poses.npz")
            np.savez_compressed(out_npz, extrinsics_w2c=novel_w2c, intrinsics=novel_k)
            print(f"[INFO] Saved novel orbit poses to: {out_npz}")

            out_glb = os.path.join(args.output_dir, "novel_orbit_poses_trace.glb")
            cam_trace_visualization(
                export_dir=args.output_dir,
                extrinsics_w2c=novel_w2c,
                intrinsics=novel_k,
                image_sizes=(H, W),
                output_name=os.path.basename(out_glb),
            )
            print(f"[INFO] Saved novel orbit trace glb to: {out_glb}")

    # Optional camera trace visualization
    if args.cam_trace:
        if prediction.extrinsics is None or prediction.intrinsics is None:
            print("[WARN] Skip camera trace: prediction.extrinsics/intrinsics not available.")
            return

        if prediction.processed_images is not None:
            H, W = prediction.processed_images.shape[1:3]
        else:
            H, W = prediction.depth.shape[-2:]

        cam_trace_visualization(
            export_dir=args.output_dir,
            extrinsics_w2c=prediction.extrinsics,  # (N,3,4) or (N,4,4)
            intrinsics=prediction.intrinsics,  # (N,3,3) or (3,3)
            image_sizes=(H, W),
            output_name="camera_trace.glb",
        )
        print(
            f"[INFO] Saved camera trajectory visualization to: {os.path.join(args.output_dir, 'camera_trace.glb')}"
        )

    # generate novel view orbit cam path


if __name__ == "__main__":
    main()
