import glob, os, torch, argparse
from depth_anything_3.api import DepthAnything3

# Parse command line arguments
parser = argparse.ArgumentParser(description="Depth Anything 3 with Gaussian Splatting")
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
