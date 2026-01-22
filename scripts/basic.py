import argparse
import glob
import os
import re
from typing import List

import numpy as np
import torch
from PIL import Image

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
        "--input-poses",
        type=str,
        default=None,
        help=(
            "Absolute path to an input poses NPZ. If provided, will use its "
            "extrinsics_w2c/intrinsics (matched by frame_ids) as input extrinsics/intrinsics "
            "to lock reconstruction in the same coordinate system."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: cuda/cpu. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--process-res-to-input",
        action="store_true",
        help=(
            "If set, infer process_res from the first input image longest side (may use more VRAM). "
            "Default: disabled, use process_res=504."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    images = gather_images(args.input_images)
    print(f"Found {len(images)} images in {args.input_images}")
    if len(images) == 0:
        raise ValueError("No images found. Supported formats: png, jpg, jpeg (case-insensitive)")

    # process_res controls InputProcessor resize (target LONGEST side, then made divisible by patch size).
    # Default is 504 to save VRAM. Optionally align to input image size.
    if args.process_res_to_input:
        with Image.open(images[0]) as im:
            w0, h0 = im.size
        process_res = max(h0, w0)
        print(f"[INFO] Using process_res={process_res} inferred from first input image size (H,W)=({h0},{w0})")
    else:
        process_res = 504
        print("[INFO] Using default process_res=504 (enable --process-res-to-input to align to input image size)")

    # Parse frame_ids from input image file names in the SAME order as `images`.
    # frame_ids[i] aligns with any prediction outputs in view dimension i.
    frame_ids_list: list[int] = []
    for img_path in images:
        base = os.path.basename(img_path)
        m = re.search(r"frame\.(\d+)\.color\.", base)
        frame_ids_list.append(int(m.group(1)) if m else -1)
    frame_ids = np.asarray(frame_ids_list, dtype=np.int32)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    export_format = args.export_format
    if export_format is None:
        export_format = "glb"
    infer_gs = "gs" in export_format

    # When rendering with a trace (gs_views), the output render resolution (render_hw)
    # must be consistent with the intrinsics stored in the poses/trace NPZ.
    # We try to infer this from `image_size_hw` saved in NPZs.
    input_hw = None  # (H, W)

    render_exts = None
    render_ixts = None
    render_hw = None
    if args.render_trace is not None:
        trace_path = args.render_trace
        if not os.path.exists(trace_path):
            raise FileNotFoundError(f"--render-trace file not found: {trace_path}")

        trace_npz = np.load(trace_path)
        if "image_size_hw" in trace_npz:
            hw = np.asarray(trace_npz["image_size_hw"]).reshape(-1).tolist()
            if len(hw) == 2:
                input_hw = (int(hw[0]), int(hw[1]))
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

    # Optional: lock reconstruction to a provided poses.npz coordinate system
    input_exts = None
    input_ixts = None
    if args.input_poses is not None:
        poses_path = args.input_poses
        if not os.path.exists(poses_path):
            raise FileNotFoundError(f"--input-poses file not found: {poses_path}")

        poses_npz = np.load(poses_path)
        if "image_size_hw" in poses_npz:
            hw = np.asarray(poses_npz["image_size_hw"]).reshape(-1).tolist()
            if len(hw) == 2:
                poses_hw = (int(hw[0]), int(hw[1]))
                if input_hw is None:
                    input_hw = poses_hw
                elif input_hw != poses_hw:
                    print(
                        f"[WARN] image_size_hw mismatch: render/trace HW={input_hw}, input_poses HW={poses_hw}. "
                        "Rendering may look blurry/misaligned."
                    )
        for k in ("extrinsics_w2c", "intrinsics", "frame_ids"):
            if k not in poses_npz:
                raise KeyError(
                    f"--input-poses NPZ must contain key '{k}', got keys: {list(poses_npz.keys())}"
                )

        poses_frame_ids = np.asarray(poses_npz["frame_ids"]).astype(np.int32)
        if poses_frame_ids.ndim != 1:
            raise ValueError(f"--input-poses frame_ids must be 1D, got {poses_frame_ids.shape}")

        # In our dataset, poses_npz is assumed to be index-aligned with input images:
        # extrinsics_w2c[i] / intrinsics[i] correspond to images[i].
        input_exts = np.asarray(poses_npz["extrinsics_w2c"])
        input_ixts = np.asarray(poses_npz["intrinsics"])

        assert input_exts.shape[0] == len(images), (
            f"--input-poses extrinsics_w2c views ({input_exts.shape[0]}) must match "
            f"number of input images ({len(images)})."
        )
        assert poses_frame_ids.shape[0] == len(images), (
            f"--input-poses frame_ids views ({poses_frame_ids.shape[0]}) must match "
            f"number of input images ({len(images)})."
        )

        if input_ixts.ndim == 2:
            input_ixts = np.broadcast_to(input_ixts[None, ...], (len(images), 3, 3)).copy()
        else:
            assert input_ixts.shape[0] == len(images), (
                f"--input-poses intrinsics views ({input_ixts.shape[0]}) must match "
                f"number of input images ({len(images)})."
            )

    # If we inferred an input/render resolution from poses/trace, use it for rendering.
    render_hw = input_hw

    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device=device)

    prediction = model.inference(
        images,
        extrinsics=input_exts,
        intrinsics=input_ixts,
        infer_gs=infer_gs,
        render_exts=render_exts,
        render_ixts=render_ixts,
        render_hw=render_hw,
        process_res=process_res,
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
