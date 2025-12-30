import os
import json
import argparse
import numpy as np
from typing import Optional, Tuple
from PIL import Image

from depth_anything_3.api import DepthAnything3


def extract_frame_id_from_filename(path: str) -> str:
    """
    Extract frame id using Hypersim naming 'frame.<digits>.' pattern first.
    Fallback to last numeric token. Normalize by stripping leading zeros (e.g. '0007' -> '7').
    """
    import re

    name = os.path.basename(path)
    m = re.search(r"frame\.(\d+)(?:[^\d]|$)", name)
    if m:
        fid = m.group(1).lstrip("0")
        return fid if fid != "" else "0"
    nums = re.findall(r"\d+", name)
    if nums:
        fid = nums[-1].lstrip("0")
        return fid if fid != "" else "0"
    raise ValueError(f"No numeric frame id found in filename: {name}")


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
    # Read JSON or whole-file if JSONL with a single object
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        try:
            data = json.loads(text)
        except Exception:
            # Fallback: first valid JSON line
            data = None
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
                raise ValueError(f"Failed to parse intrinsics from {path}")
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
    R_c2w: np.ndarray, t_c2w: Optional[np.ndarray] = None, C_w: Optional[np.ndarray] = None
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


def get_w2c_for_frame(
    extrinsics_jsonl: str, frame_id: str, cam_filter: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Lookup c2w pose for a given frame in extrinsics.jsonl and convert to w2c.
    Supports either E_3x4 (assumed c2w) or R_cw (assumed c2w) + C_w.
    """
    with open(extrinsics_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                item = json.loads(ln)
            except json.JSONDecodeError:
                continue

            if cam_filter is not None and str(item.get("cam", "")) != cam_filter:
                continue

            fstr = str(item.get("frame", "")).strip()
            if fstr == "":
                continue
            fid_norm = fstr.lstrip("0") or "0"
            if fid_norm != frame_id:
                continue

            # Prefer E_3x4 if present
            E_3x4 = item.get("E_3x4", None)
            if E_3x4 is not None:
                E_c2w = np.array(E_3x4, dtype=np.float32)
                if E_c2w.shape == (3, 4):
                    R_c2w = E_c2w[:, :3]
                    t_c2w = E_c2w[:, 3]
                    E_w2c = c2w_to_w2c(R_c2w, t_c2w=t_c2w)
                    return E_w2c

            # Fallback: R_cw (assumed c2w) + C_w
            R_c2w = np.array(item.get("R_cw", []), dtype=np.float32)
            C_w = np.array(item.get("C_w", []), dtype=np.float32)
            if R_c2w.shape == (3, 3) and C_w.shape == (3,):
                E_w2c = c2w_to_w2c(R_c2w, C_w=C_w)
                return E_w2c
    return None


def save_depth_png(
    depth: np.ndarray, out_path: str, min_percentile: float = 5.0, max_percentile: float = 95.0
) -> None:
    """
    Save a single-channel depth map as PNG after percentile normalization.
    """
    d = np.asarray(depth).astype(np.float32)
    # Handle non-finite
    mask = np.isfinite(d)
    if not np.any(mask):
        raise ValueError("Depth contains no finite values")
    d_valid = d[mask]
    lo = np.percentile(d_valid, min_percentile)
    hi = np.percentile(d_valid, max_percentile)
    if hi <= lo:
        hi = float(d_valid.max())
        lo = float(d_valid.min())
    d_clip = np.clip((d - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    img8 = (d_clip * 255.0).astype(np.uint8)
    Image.fromarray(img8).save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Validate DA3 depth with Hypersim intrinsics/extrinsics (single image)"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to a single Hypersim image")
    parser.add_argument(
        "--intrinsics-json", type=str, required=True, help="Path to intrinsics.json(l)"
    )
    parser.add_argument(
        "--extrinsics-jsonl", type=str, default=None, help="Path to extrinsics.jsonl (optional)"
    )
    parser.add_argument(
        "--cam", type=str, default=None, help="Camera stream filter (e.g., 'cam_00')"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/dataset/DA3NESTED-GIANT-LARGE",
        help="DA3 model checkpoint path",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./validate_pose_out", help="Output directory"
    )
    parser.add_argument(
        "--disable-align",
        action="store_true",
        help="Disable aligning prediction scale to input extrinsics (default: disabled for this validation)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract frame id
    frame_id = extract_frame_id_from_filename(args.image)
    print(f"[INFO] Using image: {args.image} | frame_id={frame_id}")

    # Load intrinsics (pixel-space K)
    K = load_intrinsics_hypersim(args.intrinsics_json)
    print(f"[INFO] Loaded intrinsics K:\n{K}")

    # Load extrinsics (w2c) for this frame if provided
    E_w2c = None
    if args.extrinsics_jsonl:
        E_w2c = get_w2c_for_frame(args.extrinsics_jsonl, frame_id, cam_filter=args.cam)
        if E_w2c is None:
            print(
                f"[WARN] No matching extrinsic found for frame_id={frame_id}; proceeding without extrinsics."
            )
        else:
            print(f"[INFO] Loaded w2c extrinsic (4x4):\n{E_w2c}")
            # Consistency check if C_w available is not here; skip

    # Load model
    model = DepthAnything3.from_pretrained(args.model_path)
    model = model.to(device="cuda")

    # Inference: only intrinsics
    print("[INFO] Running inference: intrinsics only")
    pred_onlyK = model.inference(
        [args.image],
        intrinsics=K[None],
        extrinsics=None,
        align_to_input_ext_scale=not args.disable_align,
        infer_gs=False,
        export_dir=None,
    )
    save_depth_png(pred_onlyK.depth[0], os.path.join(args.output_dir, "depth_intrinsics_only.png"))
    print(
        f"[STATS] onlyK depth: min={pred_onlyK.depth[0].min():.4f} max={pred_onlyK.depth[0].max():.4f} "
        f"p5={np.percentile(pred_onlyK.depth[0],5):.4f} p95={np.percentile(pred_onlyK.depth[0],95):.4f}"
    )

    # Inference: intrinsics + extrinsics (if available)
    if E_w2c is not None:
        print("[INFO] Running inference: intrinsics + extrinsics (w2c)")
        pred_withE = model.inference(
            [args.image],
            intrinsics=K[None],
            extrinsics=E_w2c[None],
            align_to_input_ext_scale=not args.disable_align,
            infer_gs=False,
            export_dir=None,
        )
        save_depth_png(
            pred_withE.depth[0], os.path.join(args.output_dir, "depth_with_extrinsics.png")
        )
        print(
            f"[STATS] withE depth: min={pred_withE.depth[0].min():.4f} max={pred_withE.depth[0].max():.4f} "
            f"p5={np.percentile(pred_withE.depth[0],5):.4f} p95={np.percentile(pred_withE.depth[0],95):.4f}"
        )
    else:
        print("[INFO] Skipped extrinsics-based inference (no E provided).")

    print(f"[DONE] Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
