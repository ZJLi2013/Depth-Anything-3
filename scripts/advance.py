import glob
import json
import os
import re
import argparse
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.camera_trj_helpers import cam_trace_visualization


def extract_frame_id_from_filename(path: str) -> str:
    """
    Extract frame id using Hypersim naming 'frame.<digits>.' pattern first.
    Fallback to last numeric token. Normalize by stripping leading zeros (e.g. '0007' -> '7').
    """
    name = os.path.basename(path)
    # Prefer explicit 'frame.<digits>' pattern like 'frame.0037.diffuse_reflectance_linear.png'
    m = re.search(r"frame\.(\d+)(?:[^\d]|$)", name)
    if m:
        fid = m.group(1).lstrip("0")
        return fid if fid != "" else "0"
    # Fallback: last numeric token in filename
    nums = re.findall(r"\d+", name)
    if nums:
        fid = nums[-1].lstrip("0")
        return fid if fid != "" else "0"
    raise ValueError(f"No numeric frame id found in filename: {name}")


def normalize_frame_id(fid: str) -> str:
    """
    Normalize a frame id string by stripping leading zeros.
    """
    fid_norm = fid.strip()
    fid_norm = fid_norm.lstrip("0")
    return fid_norm if fid_norm != "" else "0"


def load_intrinsics_hypersim(path: str) -> np.ndarray:
    """
    Load Hypersim-style intrinsics JSON/JSONL and convert to pixel-space pinhole K (3x3).
    Expected fields (shared across frames):
      - image_size: {width, height}
      - M_proj_4x4: 4x4 projection matrix (OpenGL-style) [optional]
      - M_cam_from_uv_3x3: 3x3 mapping from UV to camera [optional]

    Derivation:
      - Prefer M_proj_4x4 diagonal: fx_ndc = M_proj[0,0], fy_ndc = M_proj[1,1]
      - Fallback to M_cam_from_uv_3x3: fx_ndc = 1.0 / M_cam_from_uv[0,0], fy_ndc = 1.0 / M_cam_from_uv[1,1]
      - Convert to pixels: fx = fx_ndc * (W / 2), fy = fy_ndc * (H / 2)
      - Principal point: cx = W/2, cy = H/2
      - K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    # Read JSON or the first non-empty JSON line if JSONL
    if path.lower().endswith(".jsonl"):
        # Robust handling:
        # 1) Try parsing the entire file content as a single JSON object (multi-line JSON)
        # 2) Fallback to line-by-line JSONL: first non-empty, non-comment line via json.loads
        #    and as a last resort, ast.literal_eval to tolerate single quotes (Python-literal style)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        text_stripped = text.strip()
        data = None
        try:
            # Supports multi-line JSON object files
            data = json.loads(text_stripped)
        except Exception:
            # Fallback: treat as JSONL (one JSON object per line)
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#") or ln.startswith("//"):
                        continue
                    try:
                        data = json.loads(ln)
                        break
                    except Exception:
                        try:
                            import ast

                            data = ast.literal_eval(ln)
                            break
                        except Exception:
                            continue
        if data is None:
            raise ValueError(
                f"Failed to parse intrinsics from {path}: ensure valid JSON or JSONL format."
            )
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if (
        "image_size" not in data
        or "width" not in data["image_size"]
        or "height" not in data["image_size"]
    ):
        raise ValueError("intrinsics JSON must contain image_size.width and image_size.height")
    W = float(data["image_size"]["width"])
    H = float(data["image_size"]["height"])

    fx_ndc = None
    fy_ndc = None

    if "M_proj_4x4" in data and isinstance(data["M_proj_4x4"], list):
        M = np.array(data["M_proj_4x4"], dtype=np.float64)
        if M.shape == (4, 4):
            fx_ndc = float(M[0, 0])
            fy_ndc = float(M[1, 1])

    if (
        (fx_ndc is None or fy_ndc is None)
        and "M_cam_from_uv_3x3" in data
        and isinstance(data["M_cam_from_uv_3x3"], list)
    ):
        C = np.array(data["M_cam_from_uv_3x3"], dtype=np.float64)
        if C.shape == (3, 3):
            # Hypersim example shows C[0,0]=0.577..., reciprocal ~1.732... aligns with proj diag
            fx_ndc = float(1.0 / C[0, 0])
            fy_ndc = float(1.0 / C[1, 1])

    if fx_ndc is None or fy_ndc is None:
        raise ValueError(
            "Unable to derive focal scales: need either M_proj_4x4 (diag) or M_cam_from_uv_3x3 (reciprocal of diag)"
        )

    fx = fx_ndc * (W / 2.0)
    fy = fy_ndc * (H / 2.0)
    cx = W / 2.0
    cy = H / 2.0

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def c2w_to_w2c(
    R_c2w: np.ndarray, t_c2w: np.ndarray | None = None, C_w: np.ndarray | None = None
) -> np.ndarray:
    """
    Convert camera-to-world (c2w) pose to world-to-camera (w2c) pose.

    Args:
        R_c2w: (3,3) rotation from camera to world
        t_c2w: (3,) translation from camera to world (optional if C_w provided)
        C_w:   (3,) camera center in world coordinates (optional; used if t_c2w is None)

    Returns:
        (4,4) w2c homogeneous transform [[R_wc, t_wc],[0,0,0,1]]
        where R_wc = R_c2w^T and t_wc = -R_c2w^T * t_c2w (or -R_c2w^T * C_w)
    """
    R_c2w = np.asarray(R_c2w, dtype=np.float32)
    assert R_c2w.shape == (3, 3), "R_c2w must be (3,3)"
    R_wc = R_c2w.T

    if t_c2w is not None:
        t_c2w = np.asarray(t_c2w, dtype=np.float32).reshape(3)
        t_wc = -R_wc @ t_c2w
    elif C_w is not None:
        C_w = np.asarray(C_w, dtype=np.float32).reshape(3)
        t_wc = -R_wc @ C_w
    else:
        raise ValueError("c2w_to_w2c requires either t_c2w or C_w")

    E4 = np.eye(4, dtype=np.float32)
    E4[:3, :3] = R_wc
    E4[:3, 3] = t_wc
    return E4


def load_extrinsics_hypersim_jsonl(
    path: str, cam_filter: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Load per-frame extrinsics from a JSONL file with lines like:
      {"scene_path": "...", "cam": "cam_00", "frame": "0007",
       "R_cw": [[...3x3...]], "C_w": [x, y, z],
       "E_3x4": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz]]}

    Returns a mapping from normalized frame id -> 4x4 world-to-camera extrinsic.
    If cam_filter is provided, only entries with matching 'cam' are used.
    """
    mapping: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                item = json.loads(ln)
            except json.JSONDecodeError:
                # Allow trailing commas or non-JSONL content? Skip invalid lines.
                continue

            if cam_filter is not None and str(item.get("cam", "")) != cam_filter:
                continue

            frame = str(item.get("frame", "")).strip()
            if frame == "":
                continue
            fid = normalize_frame_id(frame)

            E_3x4 = item.get("E_3x4", None)
            if E_3x4 is not None:
                # Treat Hypersim E_3x4 as c2w and convert to w2c
                E_c2w = np.array(E_3x4, dtype=np.float32)
                if E_c2w.shape == (3, 4):
                    R_c2w = E_c2w[:, :3]
                    t_c2w = E_c2w[:, 3]
                    E4 = c2w_to_w2c(R_c2w, t_c2w=t_c2w)
                    mapping[fid] = E4
                    continue

            # Fallback: treat provided rotation/center as c2w and convert to w2c
            R_c2w = np.array(item.get("R_cw", []), dtype=np.float32)
            C_w = np.array(item.get("C_w", []), dtype=np.float32)
            if R_c2w.shape != (3, 3) or C_w.shape != (3,):
                # Skip if insufficient data
                continue

            E4 = c2w_to_w2c(R_c2w, C_w=C_w)
            mapping[fid] = E4

    if len(mapping) == 0:
        raise ValueError(f"No extrinsics parsed from {path} (cam_filter={cam_filter})")
    return mapping


def build_extrinsics_array_for_images(
    images: List[str], ext_mapping: Dict[str, np.ndarray], strict: bool = True
) -> np.ndarray:
    """
    For a sorted list of image paths, build an (N, 4, 4) extrinsics array using the mapping by frame id.
    If strict=True, raise an error if any image is missing extrinsics.
    """
    exts: List[np.ndarray] = []
    missing: List[str] = []
    for img in images:
        fid = extract_frame_id_from_filename(img)
        E4 = ext_mapping.get(fid, None)
        if E4 is None:
            missing.append(img)
            continue
        exts.append(E4.astype(np.float32))

    if strict and missing:
        raise ValueError(
            f"Missing extrinsics for {len(missing)} images: "
            + ", ".join(os.path.basename(m) for m in missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )

    if len(exts) == 0:
        raise ValueError("No extrinsics matched to images.")
    return np.stack(exts, axis=0)


def broadcast_intrinsics(K: np.ndarray, N: int) -> np.ndarray:
    """
    Broadcast a single 3x3 K to shape (N, 3, 3).
    """
    if K.shape != (3, 3):
        raise ValueError("broadcast_intrinsics: K must be 3x3")
    return np.repeat(K[None, :, :].astype(np.float32), repeats=N, axis=0)


def debug_check_frame_mapping(
    images: List[str], ext_mapping: Dict[str, np.ndarray], sample_n: int = 10
) -> None:
    """
    Debug helper: only print unmatched items between image-derived frame_ids and extrinsics keys.
    """
    # Collect image frame ids
    image_fids: List[str] = []
    for img in images:
        try:
            fid = extract_frame_id_from_filename(img)
        except Exception as e:
            print(f"[WARN] Failed to extract frame id from {os.path.basename(img)}: {e}")
            continue
        image_fids.append(fid)

    ext_fids = set(ext_mapping.keys())
    img_fids_set = set(image_fids)

    # Images with no matching extrinsics
    unmatched_images = [
        img for img in images if extract_frame_id_from_filename(img) not in ext_fids
    ]
    if unmatched_images:
        print(f"[WARN] {len(unmatched_images)} images have no matching extrinsics frame_id:")
        for name in [os.path.basename(p) for p in unmatched_images[:sample_n]]:
            print(f"  - {name}")
        if len(unmatched_images) > sample_n:
            print(f"  ... and {len(unmatched_images) - sample_n} more")
    else:
        print("[INFO] All images matched extrinsics frame_ids.")

    # Extrinsics that are not used by any image
    unused_ext_fids = [fid for fid in ext_fids if fid not in img_fids_set]
    if unused_ext_fids:
        print(f"[WARN] {len(unused_ext_fids)} extrinsics frame_ids are not present in images:")
        for fid in unused_ext_fids[:sample_n]:
            print(f"  - {fid}")
        if len(unused_ext_fids) > sample_n:
            print(f"  ... and {len(unused_ext_fids) - sample_n} more")
    else:
        print("[INFO] No unused extrinsics frame_ids.")


def filter_images_and_extrinsics(
    images: List[str], ext_map: Dict[str, np.ndarray], sample_n: int = 10
) -> Tuple[List[str], Dict[str, np.ndarray], List[str], List[str]]:
    """
    Filter to the strict intersection of image frames and extrinsics frames.
    Returns:
        matched_images: images whose frame_id exists in ext_map
        filtered_ext_map: ext_map filtered to only used frame_ids
        dropped_images: images without a matching extrinsics frame_id
        dropped_ext_fids: extrinsics frame_ids not used by any image
    """
    matched_images: List[str] = []
    matched_fids: List[str] = []
    dropped_images: List[str] = []

    for img in images:
        try:
            fid = extract_frame_id_from_filename(img)
        except Exception:
            dropped_images.append(img)
            continue
        if fid in ext_map:
            matched_images.append(img)
            matched_fids.append(fid)
        else:
            dropped_images.append(img)

    # Sort matched results by numeric frame id (ascending) and print first few
    try:
        pairs = [(int(fid), fid, img) for fid, img in zip(matched_fids, matched_images)]
    except ValueError:
        # Fallback: robust int conversion using regex when fid contains non-numeric parts
        def to_int_safe(fid_str: str) -> int:
            nums = re.findall(r"\d+", fid_str)
            return int(nums[-1]) if nums else 0

        pairs = [(to_int_safe(fid), fid, img) for fid, img in zip(matched_fids, matched_images)]
    pairs.sort(key=lambda x: x[0])

    matched_fids = [fid for _, fid, _ in pairs]
    matched_images = [img for _, _, img in pairs]

    print("[INFO] Matched images sorted by frame id (numeric ascending):")
    for i, (_, fid, img) in enumerate(pairs[:sample_n]):
        print(f"  [{i:02d}] fid={fid} name={os.path.basename(img)}")

    matched_set = set(matched_fids)
    filtered_ext_map: Dict[str, np.ndarray] = {fid: ext_map[fid] for fid in matched_set}
    dropped_ext_fids: List[str] = [fid for fid in ext_map.keys() if fid not in matched_set]

    # Summary
    print(
        f"[INFO] Images total={len(images)}, matched={len(matched_images)}, dropped={len(dropped_images)}"
    )
    if dropped_images:
        print(f"[WARN] Dropped {len(dropped_images)} unmatched images (showing up to {sample_n}):")
        for name in [os.path.basename(p) for p in dropped_images[:sample_n]]:
            print(f"  - {name}")
        if len(dropped_images) > sample_n:
            print(f"  ... and {len(dropped_images) - sample_n} more")

    print(
        f"[INFO] Extrinsics total={len(ext_map)}, used={len(filtered_ext_map)}, dropped={len(dropped_ext_fids)}"
    )
    if dropped_ext_fids:
        print(
            f"[WARN] Dropped {len(dropped_ext_fids)} unused extrinsics frame_ids (up to {sample_n}):"
        )
        for fid in dropped_ext_fids[:sample_n]:
            print(f"  - {fid}")
        if len(dropped_ext_fids) > sample_n:
            print(f"  ... and {len(dropped_ext_fids) - sample_n} more")

    return matched_images, filtered_ext_map, dropped_images, dropped_ext_fids


def main():
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 with Gaussian Splatting + camera inputs"
    )
    parser.add_argument(
        "--input-images",
        type=str,
        default="assets/examples/SOH",
        help="Path to input images directory (default: assets/examples/SOH)",
    )
    parser.add_argument(
        "--intrinsics-json",
        type=str,
        default=None,
        help="Path to shared intrinsics JSON or JSONL (Hypersim-style)",
    )
    parser.add_argument(
        "--extrinsics-jsonl",
        type=str,
        default=None,
        help="Path to per-frame extrinsics JSONL (Hypersim-style)",
    )
    parser.add_argument(
        "--cam",
        type=str,
        default=None,
        help="Camera stream filter (e.g., 'cam_00'); if provided, only use matching extrinsics entries",
    )
    parser.add_argument(
        "--strict-match",
        action="store_true",
        help="Require extrinsics for every image; otherwise missing frames are skipped",
    )
    parser.add_argument(
        "--align-to-input-ext-scale",
        action="store_true",
        help="Align prediction to input extrinsics scale (default: enabled)",
    )
    parser.add_argument(
        "--no-align-to-input-ext-scale",
        action="store_true",
        help="Disable aligning prediction scale to input extrinsics",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save GS outputs (default: ./output)",
    )
    parser.add_argument(
        "--gs-head",
        action="store_true",
        help="enable GS head",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda")

    # Load model (da3nested-giant-large as specified)
    model = DepthAnything3.from_pretrained("/dataset/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)

    # Gather images (support multiple extensions)
    example_path = args.input_images
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    images: List[str] = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(example_path, ext)))
    images = sorted(images)

    print(f"Found {len(images)} images in {example_path}")
    if len(images) == 0:
        print("Warning: No images found. Supported formats: png, jpg, jpeg (case-insensitive)")
        return

    # Load intrinsics (shared)
    intrinsics_arr: Optional[np.ndarray] = None
    if args.intrinsics_json is not None:
        K = load_intrinsics_hypersim(args.intrinsics_json)
        intrinsics_arr = broadcast_intrinsics(K, len(images))
        print("Loaded intrinsics (shared) K:\n", K)
    else:
        print("No intrinsics JSON provided; proceeding without K (model will estimate or handle).")

    # Load extrinsics (per-frame)
    extrinsics_arr: Optional[np.ndarray] = None
    if args.extrinsics_jsonl is not None:
        ext_map = load_extrinsics_hypersim_jsonl(args.extrinsics_jsonl, cam_filter=args.cam)
        # Only print unmatches
        debug_check_frame_mapping(images, ext_map)
        # Strict intersection: keep only frames that appear in both images and extrinsics
        matched_images, filtered_ext_map, dropped_images, dropped_ext_fids = (
            filter_images_and_extrinsics(images, ext_map)
        )
        images = matched_images
        # Re-broadcast intrinsics after filtering (if provided)
        if args.intrinsics_json is not None:
            intrinsics_arr = broadcast_intrinsics(K, len(images))
        # Build extrinsics for filtered images (strict)
        extrinsics_arr = build_extrinsics_array_for_images(images, filtered_ext_map, strict=True)
        print(f"Loaded {extrinsics_arr.shape[0]} per-frame extrinsics from JSONL.")
    else:
        print("No extrinsics JSONL provided; proceeding without E (model will estimate poses).")

    # Safety: DA3 api currently assumes intrinsics is not None when extrinsics is provided in alignment
    if extrinsics_arr is not None and intrinsics_arr is None:
        print(
            "Error: extrinsics provided but intrinsics missing. "
            "Please provide --intrinsics-json so K can be aligned and resized correctly."
        )
        return

    if args.gs_head:
        # Run inference with GS head enabled
        prediction = model.inference(
            images,
            extrinsics=extrinsics_arr,
            intrinsics=intrinsics_arr,
            infer_gs=True,  # Enable Gaussian Splatting branch
            export_dir=args.output_dir,  # Directory to save GS outputs
            export_format="gs_views",
        )
    else:
        prediction = model.inference(
            images,
            extrinsics=extrinsics_arr,
            intrinsics=intrinsics_arr,
            infer_gs=False,
            export_dir=args.output_dir,
            export_format="glb",
        )

    # Standard outputs
    # prediction.processed_images : [N, H, W, 3] uint8 array
    print("Processed images shape:", prediction.processed_images.shape)
    # prediction.depth            : [N, H, W]    float32 array
    print("Depth shape:", prediction.depth.shape)
    # prediction.conf             : [N, H, W]    float32 array
    print("Confidence shape:", prediction.conf.shape)
    # prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
    print("Extrinsics shape:", prediction.extrinsics.shape)
    # prediction.intrinsics       : [N, 3, 3]    float32 array
    print("Intrinsics shape:", prediction.intrinsics.shape)

    # Add cam pose visualization (camera-only GLB, for trajectory debugging)
    if prediction.extrinsics is not None and prediction.intrinsics is not None:
        # Use processed image size if available; fallback to depth size
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
    else:
        print(
            "[WARN] Skip camera trajectory visualization: prediction.extrinsics/intrinsics not available."
        )


if __name__ == "__main__":
    main()
