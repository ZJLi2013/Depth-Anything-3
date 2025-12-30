import os
import json
import argparse
import numpy as np
import random
import glob
from typing import Optional, Tuple, List, Dict
from PIL import Image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.pose_align import align_poses_umeyama


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


def parse_extrinsics_all(extrinsics_jsonl: str, cam_filter: Optional[str] = None) -> List[Dict]:
    """
    Parse extrinsics.jsonl and unify to c2w 4x4 per frame.

    Returns a list of dicts:
      { "frame_id": str, "E_c2w": np.ndarray (4,4), "has_Cw": bool }
    """
    entries: List[Dict] = []
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

            E_3x4 = item.get("E_3x4", None)
            R_cw = item.get("R_cw", None)
            C_w = item.get("C_w", None)

            E_c2w = None
            has_Cw = False
            if E_3x4 is not None:
                M = np.array(E_3x4, dtype=np.float32)
                if M.shape == (3, 4):
                    E_c2w = np.eye(4, dtype=np.float32)
                    E_c2w[:3, :3] = M[:, :3]
                    E_c2w[:3, 3] = M[:, 3]
            elif R_cw is not None and C_w is not None:
                R = np.array(R_cw, dtype=np.float32)
                C = np.array(C_w, dtype=np.float32).reshape(3)
                if R.shape == (3, 3) and C.shape == (3,):
                    E_c2w = np.eye(4, dtype=np.float32)
                    E_c2w[:3, :3] = R
                    E_c2w[:3, 3] = C
                    has_Cw = True

            if E_c2w is not None:
                entries.append({"frame_id": fid_norm, "E_c2w": E_c2w, "has_Cw": has_Cw})
    return entries


def sample_random_frames(entries: List[Dict], num_frames: int, seed: int) -> List[Dict]:
    rnd = random.Random(seed)
    if num_frames >= len(entries):
        return entries
    return rnd.sample(entries, num_frames)


def run_extrinsics_checks(
    extrinsics_jsonl: str, cam_filter: Optional[str], num_frames: int, seed: int
) -> None:
    """
    Algebraic consistency checks for c2w_to_w2c conversion:
    - Rotation orthogonality & det=+1
    - Inverse consistency: inv(w2c) ≈ c2w
    - Camera center roundtrip: origin -> C_w -> origin
    """
    assert extrinsics_jsonl is not None, "--extrinsics-jsonl is required for --check-extrinsics"
    entries = parse_extrinsics_all(extrinsics_jsonl, cam_filter)
    if len(entries) == 0:
        print("[ERROR] No valid extrinsics entries found.")
        return
    picked = sample_random_frames(entries, max(1, num_frames), seed)
    print(f"[INFO] Algebra checks on {len(picked)}/{len(entries)} sampled frames")

    def mat_inv(E: np.ndarray) -> np.ndarray:
        return np.linalg.inv(E)

    total = 0
    passed = 0
    for it in picked:
        fid = it["frame_id"]
        E_c2w = it["E_c2w"]
        R_c2w = E_c2w[:3, :3]
        t_c2w = E_c2w[:3, 3]

        # Convert to w2c
        R_wc = R_c2w.T
        t_wc = -R_wc @ t_c2w
        E_w2c = np.eye(4, dtype=np.float32)
        E_w2c[:3, :3] = R_wc
        E_w2c[:3, 3] = t_wc

        # Checks
        orth_err = float(np.linalg.norm(R_c2w.T @ R_c2w - np.eye(3)))
        det_val = float(np.linalg.det(R_c2w))
        inv_err = float(np.linalg.norm(mat_inv(E_w2c) - E_c2w))

        # Origin roundtrip
        origin_cam = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        world_from_cam = (E_c2w @ origin_cam)[:3]
        cam_from_world = (E_w2c @ np.append(world_from_cam, 1.0))[:3]
        origin_roundtrip_err = float(np.linalg.norm(cam_from_world - origin_cam[:3]))

        ok = (
            (orth_err < 1e-3)
            and (abs(det_val - 1.0) < 1e-3)
            and (inv_err < 1e-5)
            and (origin_roundtrip_err < 1e-5)
        )
        print(
            f"[CHECK] frame={fid} | orth_err={orth_err:.2e} det={det_val:+.6f} inv_err={inv_err:.2e} origin_rt={origin_roundtrip_err:.2e} | {'PASS' if ok else 'FAIL'}"
        )
        total += 1
        passed += int(ok)

    print(f"[SUMMARY] Passed {passed}/{total} frames ({(passed/total)*100:.1f}%).")


def get_images_map(images_dir: str, image_ext: str) -> Dict[str, str]:
    """
    Build map of frame_id -> image_path using extract_frame_id_from_filename.
    """
    paths = glob.glob(os.path.join(images_dir, f"*{image_ext}"))
    fmap: Dict[str, str] = {}
    for p in paths:
        try:
            fid = extract_frame_id_from_filename(p)
            fmap[fid] = p
        except Exception:
            continue
    return fmap


def to_4x4(mats: np.ndarray) -> np.ndarray:
    """
    Ensure extrinsics are (N,4,4) by padding if (N,3,4).
    """
    mats = np.asarray(mats)
    if mats.ndim != 3:
        raise ValueError("Extrinsics must be (N,3,4) or (N,4,4)")
    N = mats.shape[0]
    if mats.shape[1:] == (4, 4):
        return mats
    if mats.shape[1:] == (3, 4):
        out = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)
        out[:, :3, :4] = mats
        return out
    raise ValueError("Unsupported extrinsics shape")


def w2c_to_center(E: np.ndarray) -> np.ndarray:
    """
    Compute camera center in world coords from w2c 4x4.
    """
    c2w = np.linalg.inv(E)
    return c2w[:3, 3]


def run_scale_validation(
    images_dir: str,
    image_ext: str,
    extrinsics_jsonl: str,
    cam_filter: Optional[str],
    num_frames: int,
    seed: int,
    K: np.ndarray,
    model_path: str,
) -> None:
    """
    Scale/alignment validation using random sampled frames (requires >=3).
    - Sample frames with both extrinsics and images
    - Run model to get predicted trajectory (intrinsics-only)
    - Align predicted to input extrinsics via Umeyama and report scale & ATE
    """
    assert extrinsics_jsonl is not None, "--extrinsics-jsonl is required for --check-scale"
    assert images_dir is not None, "--images-dir is required for --check-scale"
    entries = parse_extrinsics_all(extrinsics_jsonl, cam_filter)
    fmap = get_images_map(images_dir, image_ext)
    usable = [it for it in entries if it["frame_id"] in fmap]

    if len(usable) < 3:
        print(f"[ERROR] Not enough usable frames with images. Found={len(usable)} (need >=3).")
        return

    picked = sample_random_frames(usable, max(3, num_frames), seed)
    images = [fmap[it["frame_id"]] for it in picked]
    ex_w2c = np.stack([it["E_c2w"] for it in picked], axis=0)
    # Convert c2w to w2c
    ex_w2c = np.array([np.linalg.inv(E) for E in ex_w2c], dtype=np.float32)

    # Stack intrinsics
    K_stack = np.repeat(K[None, ...], len(images), axis=0)

    # Load model and run prediction (intrinsics-only)
    model = DepthAnything3.from_pretrained(model_path)
    model = model.to(device="cuda")
    print(f"[INFO] Running prediction on {len(images)} frames (intrinsics-only)")
    pred = model.inference(
        images,
        intrinsics=K_stack,
        extrinsics=None,
        align_to_input_ext_scale=False,
        infer_gs=False,
        export_dir=None,
    )

    pred_ext = to_4x4(pred.extrinsics)
    # Align predicted to input via Umeyama
    print("[INFO] Performing Umeyama Sim(3) alignment (ransac if N>=10)")
    r, t, s, aligned_pred = align_poses_umeyama(
        pred_ext,
        ex_w2c,
        ransac=len(pred_ext) >= 10,
        return_aligned=True,
        random_state=seed,
    )

    # Compute ATE RMSE on camera centers
    aligned_4x4 = to_4x4(aligned_pred)
    centers_aligned = np.stack([w2c_to_center(E) for E in aligned_4x4], axis=0)
    centers_input = np.stack([w2c_to_center(E) for E in ex_w2c], axis=0)
    diffs = centers_aligned - centers_input
    rmse = float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))

    print(f"[RESULT] scale={s:.6f} | ATE_RMSE={rmse:.6f} | frames={len(images)}")
    print(
        "[NOTE] Large ATE or extreme scale indicates possible coordinate/ordering/intrinsics mismatch."
    )


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
    # MVP: sampling and validation flags
    parser.add_argument(
        "--num-frames", type=int, default=3, help="Number of frames to sample for checks (>=1)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--check-extrinsics",
        action="store_true",
        help="Run algebraic extrinsics checks only (no model)",
    )
    parser.add_argument(
        "--check-scale",
        action="store_true",
        help="Run scale/alignment validation (requires >=3 frames)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory with images (matched by frame id) for scale check",
    )
    parser.add_argument(
        "--image-ext", type=str, default=".png", help="Image file extension filter for images-dir"
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

    # Optional validation modes (MVP)
    if args.check_extrinsics:
        run_extrinsics_checks(args.extrinsics_jsonl, args.cam, args.num_frames, args.seed)
        print("[DONE] Algebraic extrinsics checks completed.")
        return

    if args.check_scale:
        if args.num_frames < 3:
            raise ValueError("Scale check requires --num-frames >= 3")
        if args.images_dir is None:
            raise ValueError("Scale check requires --images-dir to locate images")
        run_scale_validation(
            args.images_dir,
            args.image_ext,
            args.extrinsics_jsonl,
            args.cam,
            args.num_frames,
            args.seed,
            K,
            args.model_path,
        )
        print("[DONE] Scale/alignment check completed.")
        return

    # Load model
    model = DepthAnything3.from_pretrained(args.model_path)
    model = model.to(device="cuda")

    # Inference: only intrinsics
    print("[INFO] Running inference: intrinsics only")
    pred_onlyK = model.inference(
        [args.image],
        intrinsics=K[None],
        extrinsics=None,
        align_to_input_ext_scale=False,
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
        print(
            "[INFO] Running inference: extrinsics provided but single-frame; skipping alignment (intrinsics-only)"
        )
        pred_withE = model.inference(
            [args.image],
            intrinsics=K[None],
            extrinsics=None,
            align_to_input_ext_scale=False,
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
