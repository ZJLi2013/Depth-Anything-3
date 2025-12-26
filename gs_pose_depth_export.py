import glob, os, torch, argparse, numpy as np
from PIL import Image
import json
from depth_anything_3.api import DepthAnything3

"""
Depth Anything 3: Gaussian Splatting with per-image camera pose and relative depth export.
This script supersedes 'basic_gs.py' with a more descriptive name.
"""

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Depth Anything 3: GS with per-image camera pose and relative depth export"
)
parser.add_argument(
    "--input-images",
    type=str,
    default="assets/examples/SOH",
    help="Path to input images directory (default: assets/examples/SOH)",
)
args = parser.parse_args()

device = torch.device("cuda")

# Load model with GS support (da3-giant or da3nested-giant-large)
# Using da3nested-giant-large as specified in the original basic.py
model = DepthAnything3.from_pretrained("/dataset/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

example_path = args.input_images
# Support multiple image formats (png, jpg, jpeg)
image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
images = []
for ext in image_extensions:
    images.extend(glob.glob(os.path.join(example_path, ext)))
images = sorted(images)

# Print debug info
print(f"Found {len(images)} images in {example_path}")
if len(images) == 0:
    print(f"Warning: No images found in {example_path}")
    print(f"Supported formats: png, jpg, jpeg (case-insensitive)")

# Run inference with GS head enabled
prediction = model.inference(
    images,
    infer_gs=True,  # Enable Gaussian Splatting branch
    export_dir="./output_gs",  # Directory to save GS outputs
    export_format="gs_ply",  # Export GS format (can also use "gs_ply-gs_video" for both)
)

# Standard outputs
# prediction.processed_images : [N, H, W, 3] uint8   array
print("Processed images shape:", prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print("Depth shape:", prediction.depth.shape)
# prediction.conf             : [N, H, W]    float32 array
print("Confidence shape:", prediction.conf.shape)
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print("Extrinsics shape:", prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print("Intrinsics shape:", prediction.intrinsics.shape)

# GS-specific output
# prediction.gaussians        : Gaussian Splatting data
if hasattr(prediction, "gaussians"):
    print("\nGaussian Splatting data available in prediction.gaussians")
    print("GS means shape:", prediction.gaussians.means.shape)
    print("GS PLY file exported to: ./output_gs/gs_ply/0000.ply")
else:
    print("\nWarning: No Gaussian data found. Make sure infer_gs=True is set.")

# Additional outputs in aux dictionary
if hasattr(prediction, "aux"):
    print("\nAuxiliary outputs available:")
    for key in prediction.aux.keys():
        print(f"  - {key}")

# ------------------------------------------------------------------------------
# Per-image outputs: estimated camera pose, relative depth .npy/.png, and rel_depth .ply
# Saved into the same directory as the 3D GS PLY outputs (./output_gs/gs_ply)
# ------------------------------------------------------------------------------

# Directory where GS PLYs are written by export_format="gs_ply"
gs_dir = os.path.join("./output_gs", "gs_ply")
os.makedirs(gs_dir, exist_ok=True)


def save_depth_png16(path: str, depth: np.ndarray) -> None:
    """
    Save depth map as 16-bit PNG, min-max normalized per image.
    Values are mapped to [0, 65535]. This is a visualization format (relative depth).
    """
    d = depth.astype(np.float32)
    finite_mask = np.isfinite(d)
    if not np.any(finite_mask):
        d16 = np.zeros_like(d, dtype=np.uint16)
    else:
        vmin = float(d[finite_mask].min())
        vmax = float(d[finite_mask].max())
        if vmax <= vmin:
            d16 = np.zeros_like(d, dtype=np.uint16)
        else:
            scaled = (d - vmin) / (vmax - vmin)
            d16 = (scaled.clip(0.0, 1.0) * 65535.0).astype(np.uint16)
    img = Image.fromarray(d16, mode="I;16")
    img.save(path)


def save_cam_pose_json(
    path: str, image_path: str, intr: np.ndarray, ext_3x4_w2c: np.ndarray
) -> None:
    """
    Save estimated camera intrinsics and extrinsics (w2c, 3x4) to JSON.
    """
    data = {
        "image_path": image_path,
        "intrinsics_3x3": intr.tolist(),
        "extrinsics_3x4_w2c": ext_3x4_w2c.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_rel_depth_ply(
    path: str,
    depth: np.ndarray,
    intr: np.ndarray,
    ext_3x4_w2c: np.ndarray,
    rgb: np.ndarray | None = None,
    stride: int = 4,
) -> None:
    """
    Generate a per-image point cloud (relative depth) and save as ASCII PLY.
    - Unprojects pixels using intrinsics (fx, fy, cx, cy)
    - Transforms camera points to world using inverse of w2c extrinsics
    - Optionally colors points using RGB from processed image
    """
    H, W = depth.shape
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]

    # Build 4x4 w2c and invert to get c2w
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = ext_3x4_w2c[:, :3]
    E[:3, 3] = ext_3x4_w2c[:, 3]
    C2W = np.linalg.inv(E)

    vertices = []
    colors = []

    for v in range(0, H, stride):
        for u in range(0, W, stride):
            z = float(depth[v, u])
            if not np.isfinite(z) or z <= 0.0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            cam_p = np.array([x, y, z, 1.0], dtype=np.float32)
            wp = C2W @ cam_p
            vertices.append((wp[0], wp[1], wp[2]))
            if rgb is not None:
                c = rgb[v, u]
                colors.append((int(c[0]), int(c[1]), int(c[2])))
            else:
                colors.append((255, 255, 255))

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(vertices, colors):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


# Perform saves for each input image
N = prediction.depth.shape[0]
for idx, img_path in enumerate(images):
    depth = prediction.depth[idx]  # [H, W], float32, relative
    intr = prediction.intrinsics[idx]  # [3, 3]
    ext = prediction.extrinsics[idx]  # [3, 4] w2c
    rgb = prediction.processed_images[idx] if hasattr(prediction, "processed_images") else None

    # Save depth .npy (float32, relative depth)
    depth_npy_path = os.path.join(gs_dir, f"depth_{idx:04d}.npy")
    np.save(depth_npy_path, depth)

    # Save depth.png (16-bit visualization, min-max per image)
    depth_png_path = os.path.join(gs_dir, f"depth_{idx:04d}.png")
    save_depth_png16(depth_png_path, depth)

    # Save camera pose (intrinsics + w2c extrinsics) as JSON
    cam_json_path = os.path.join(gs_dir, f"cam_pose_{idx:04d}.json")
    save_cam_pose_json(cam_json_path, img_path, intr, ext)

    # Save relative-depth-derived point cloud as PLY
    rel_ply_path = os.path.join(gs_dir, f"rel_depth_{idx:04d}.ply")
    save_rel_depth_ply(rel_ply_path, depth, intr, ext, rgb=rgb, stride=4)

    print(f"\nSaved per-image outputs for index {idx:04d}:")
    print(f"  - depth (npy)  : {depth_npy_path}")
    print(f"  - depth (png16): {depth_png_path}")
    print(f"  - cam pose json: {cam_json_path}")
    print(f"  - rel_depth ply: {rel_ply_path}")
