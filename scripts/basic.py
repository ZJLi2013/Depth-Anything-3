import argparse
import glob
import os
import re
from typing import List

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.camera_trj_helpers import cam_trace_visualization
from depth_anything_3.utils.export.glb import dump_camera_poses_npz

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
        "--export-format",
        type=str,
        default=None,
        help=("Export format string passed to DA3 (e.g. glb, gs_views). " "If omitted: glb."),
    )
    parser.add_argument(
        "--render-trace",
        type=str,
        default=None,
        help=("Absolute path to a trace NPZ."),
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
        export_format = "glb"
    infer_gs = "gs" in export_format

    render_exts = None
    render_ixts = None
    render_hw = None
    if args.render_trace is not None:
        trace_path = args.render_trace
        if not os.path.exists(trace_path):
            raise FileNotFoundError(f"--render-trace file not found: {trace_path}")

        trace_npz = np.load(trace_path)
        if "extrinsics_w2c" not in trace_npz or "intrinsics" not in trace_npz:
            raise KeyError(
                f"Trace NPZ must contain keys 'extrinsics_w2c' and 'intrinsics', got: {list(trace_npz.keys())}"
            )
        render_exts_np = trace_npz["extrinsics_w2c"]
        render_ixts_np = trace_npz["intrinsics"]

        # Convert to torch tensors for renderer (affine_inverse is torchscript -> Tensor only)
        render_exts = torch.as_tensor(render_exts_np, dtype=torch.float32, device=device)
        render_ixts = torch.as_tensor(render_ixts_np, dtype=torch.float32, device=device)

        # Normalize shapes to (B, V, ...)
        # extrinsics: (V,4,4)/(V,3,4) or (B,V,*,*)
        if render_exts.ndim == 3:
            render_exts = render_exts.unsqueeze(0)
        elif render_exts.ndim != 4:
            raise ValueError(
                f"extrinsics_w2c must be (V,4,4)/(V,3,4) or (B,V,*,*), got {tuple(render_exts.shape)}"
            )

        # intrinsics: (3,3)/(V,3,3) or (B,V,3,3)
        if render_ixts.ndim == 2:
            render_ixts = render_ixts.unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
            render_ixts = render_ixts.expand(render_exts.shape[0], render_exts.shape[1], 3, 3)
        elif render_ixts.ndim == 3:
            render_ixts = render_ixts.unsqueeze(0)
        elif render_ixts.ndim != 4:
            raise ValueError(
                f"intrinsics must be (3,3)/(V,3,3) or (B,V,3,3), got {tuple(render_ixts.shape)}"
            )

    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device=device)

    prediction = model.inference(
        images,
        extrinsics=None,
        intrinsics=None,
        infer_gs=infer_gs,
        render_exts=render_exts,
        render_ixts=render_ixts,
        render_hw=render_hw,
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

    # Determine image size (needed for camera trace visualization)
    H = W = None
    if prediction.processed_images is not None:
        H, W = prediction.processed_images.shape[1:3]
    elif prediction.depth is not None:
        H, W = prediction.depth.shape[-2:]

    # For GLB exports: dump camera poses (for later gs_views render_trace) and visualize them.
    # IMPORTANT: poses.npz must stay in the same coordinate system as prediction.extrinsics/intrinsics
    # because gs_views render_trace expects DA3's original w2c/intrinsics.
    if "glb" in export_format:
        if prediction.extrinsics is None or prediction.intrinsics is None:
            print("[WARN] Skip pose dump: prediction.extrinsics/intrinsics not available.")
        else:
            # Parse frame_id from input image file names in the SAME order as `images`
            # (so frame_ids[i] aligns with extrinsics_w2c[i]).
            frame_ids_list: list[int] = []
            for img_path in images:
                base = os.path.basename(img_path)
                m = re.search(r"frame\.(\d+)\.color\.", base)
                frame_ids_list.append(int(m.group(1)) if m else -1)
            frame_ids = np.asarray(frame_ids_list, dtype=np.int32)

            poses_path = dump_camera_poses_npz(
                export_dir=args.output_dir,
                extrinsics_w2c=prediction.extrinsics,
                intrinsics=prediction.intrinsics,
                output_name="poses.npz",
                image_size_hw=(int(H), int(W)) if (H is not None and W is not None) else None,
                frame_ids=frame_ids,
            )
            print(f"[INFO] Saved camera poses to: {poses_path}")

            if H is None or W is None:
                print("[WARN] Cannot determine (H,W); skip cam_trace_visualization.")
            else:
                poses_npz = np.load(poses_path)
                trace_path = cam_trace_visualization(
                    export_dir=args.output_dir,
                    extrinsics_w2c=poses_npz["extrinsics_w2c"],
                    intrinsics=poses_npz["intrinsics"],
                    image_sizes=(int(H), int(W)),
                    output_name="poses_trace.glb",
                    camera_size=0.03,
                    align_to_gltf=True,
                )
                print(f"[INFO] Saved camera trace visualization to: {trace_path}")


if __name__ == "__main__":
    main()
