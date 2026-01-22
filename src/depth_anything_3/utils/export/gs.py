# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Literal, Optional
import moviepy.editor as mpy
import numpy as np
import torch
from PIL import Image

from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.layout_helpers import hcat, vcat
from depth_anything_3.utils.pose_align import align_poses_umeyama
from depth_anything_3.utils.visualize import vis_depth_map_tensor

VIDEO_QUALITY_MAP = {
    "low": {"crf": "28", "preset": "veryfast"},
    "medium": {"crf": "23", "preset": "medium"},
    "high": {"crf": "18", "preset": "slow"},
}


def export_to_gs_ply(
    prediction: Prediction,
    export_dir: str,
    gs_views_interval: Optional[
        int
    ] = 1,  # export GS every N views, useful for extremely dense inputs
):
    gs_world = prediction.gaussians
    pred_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(gs_world.means)  # v h w 1
    idx = 0
    os.makedirs(os.path.join(export_dir, "gs_ply"), exist_ok=True)
    save_path = os.path.join(export_dir, f"gs_ply/{idx:04d}.ply")
    if gs_views_interval is None:  # select around 12 views in total
        gs_views_interval = max(pred_depth.shape[0] // 12, 1)
    save_gaussian_ply(
        gaussians=gs_world,
        save_path=save_path,
        ctx_depth=pred_depth,
        shift_and_scale=False,
        save_sh_dc_only=True,
        gs_views_interval=gs_views_interval,
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )


def export_to_gs_video(
    prediction: Prediction,
    export_dir: str,
    extrinsics: Optional[torch.Tensor] = None,  # render views' world2cam, "b v 4 4"
    intrinsics: Optional[torch.Tensor] = None,  # render views' unnormed intrinsics, "b v 3 3"
    out_image_hw: Optional[tuple[int, int]] = None,  # render views' resolution, (h, w)
    chunk_size: Optional[int] = 4,
    trj_mode: Literal[
        "original",
        "smooth",
        "interpolate",
        "interpolate_smooth",
        "wander",
        "dolly_zoom",
        "extend",
        "wobble_inter",
    ] = "extend",
    color_mode: Literal["RGB+D", "RGB+ED"] = "RGB+ED",
    vis_depth: Optional[Literal["hcat", "vcat"]] = None,
    enable_tqdm: Optional[bool] = True,
    output_name: Optional[str] = None,
    video_quality: Literal["low", "medium", "high"] = "high",
) -> None:
    gs_world = prediction.gaussians
    # if target poses are not provided, render the (smooth/interpolate) input poses
    if extrinsics is not None:
        tgt_extrs = extrinsics
    else:
        tgt_extrs = torch.from_numpy(prediction.extrinsics).unsqueeze(0).to(gs_world.means)
        if prediction.is_metric:
            scale_factor = prediction.scale_factor
            if scale_factor is not None:
                tgt_extrs[:, :, :3, 3] /= scale_factor
    tgt_intrs = (
        intrinsics
        if intrinsics is not None
        else torch.from_numpy(prediction.intrinsics).unsqueeze(0).to(gs_world.means)
    )
    # if render resolution is not provided, render the input ones
    if out_image_hw is not None:
        H, W = out_image_hw
    else:
        H, W = prediction.depth.shape[-2:]
    # if single views, render wander trj
    if tgt_extrs.shape[1] <= 1:
        trj_mode = "wander"
        # trj_mode = "dolly_zoom"

    color, depth = run_renderer_in_chunk_w_trj_mode(
        gaussians=gs_world,
        extrinsics=tgt_extrs,
        intrinsics=tgt_intrs,
        image_shape=(H, W),
        chunk_size=chunk_size,
        trj_mode=trj_mode,
        use_sh=True,
        color_mode=color_mode,
        enable_tqdm=enable_tqdm,
    )

    # save as video
    ffmpeg_params = [
        "-crf",
        VIDEO_QUALITY_MAP[video_quality]["crf"],
        "-preset",
        VIDEO_QUALITY_MAP[video_quality]["preset"],
        "-pix_fmt",
        "yuv420p",
    ]  # best compatibility

    os.makedirs(os.path.join(export_dir, "gs_video"), exist_ok=True)
    for idx in range(color.shape[0]):
        video_i = color[idx]
        if vis_depth is not None:
            depth_i = vis_depth_map_tensor(depth[0])
            cat_fn = hcat if vis_depth == "hcat" else vcat
            video_i = torch.stack([cat_fn(c, d) for c, d in zip(video_i, depth_i)])
        frames = list(
            (video_i.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        )  # T x H x W x C, uint8, numpy()

        fps = 24
        clip = mpy.ImageSequenceClip(frames, fps=fps)
        output_name = f"{idx:04d}_{trj_mode}" if output_name is None else output_name
        save_path = os.path.join(export_dir, f"gs_video/{output_name}.mp4")
        # clip.write_videofile(save_path, codec="libx264", audio=False, bitrate="4000k")
        clip.write_videofile(
            save_path,
            codec="libx264",
            audio=False,
            fps=fps,
            ffmpeg_params=ffmpeg_params,
        )
    return


def export_to_gs_views(
    prediction: Prediction,
    export_dir: str,
    extrinsics: Optional[torch.Tensor] = None,  # render views' world2cam, "b v 4 4"
    intrinsics: Optional[torch.Tensor] = None,  # render views' unnormed intrinsics, "b v 3 3"
    out_image_hw: Optional[tuple[int, int]] = None,  # render views' resolution, (h, w)
    chunk_size: Optional[int] = 4,
    trj_mode: Literal[
        "original",
        "smooth",
        "interpolate",
        "interpolate_smooth",
        "wander",
        "dolly_zoom",
        "extend",
        "wobble_inter",
    ] = "original",
    color_mode: Literal["RGB+D", "RGB+ED"] = "RGB+ED",
    enable_tqdm: Optional[bool] = True,
    output_name: Optional[str] = None,
    image_format: Literal["png", "jpg", "jpeg"] = "png",
    align_extrinsics_to_prediction: bool = True,
    align_ransac: bool = False,
) -> list[list[str]]:
    """Render a set of novel-view images from 3DGS and save them to disk.

    This is similar to :func:`export_to_gs_video`, but instead of encoding an mp4,
    it writes individual frames (novel views) as images. Depth visualization is
    intentionally omitted (color only), which is typically what metrics such as
    PSNR/SSIM/LPIPS operate on.

    Returns:
        A nested list of saved image paths with shape [B][T] where B is the batch
        dimension of the renderer output and T is the number of rendered views.
    """
    gs_world = prediction.gaussians

    # Reference extrinsics: the same coordinate system used to render well by default.
    ref_extrs = torch.from_numpy(prediction.extrinsics).unsqueeze(0).to(gs_world.means)
    if prediction.is_metric:
        scale_factor = prediction.scale_factor
        if scale_factor is not None:
            ref_extrs[:, :, :3, 3] /= scale_factor

    # If target poses are not provided, render the input poses (in the same coord as gaussians).
    if extrinsics is not None:
        tgt_extrs = extrinsics.to(gs_world.means)
        # If prediction.is_metric, make the provided extrinsics consistent with gaussians coord too.
        if prediction.is_metric:
            scale_factor = prediction.scale_factor
            if scale_factor is not None:
                tgt_extrs = tgt_extrs.clone()
                tgt_extrs[:, :, :3, 3] /= scale_factor

        # Optional: align provided extrinsics to the prediction coordinate system (gaussians coord).
        # This avoids severe blur/ghosting when `extrinsics` come from a different run / coordinate system.
        if align_extrinsics_to_prediction:
            # Ensure (B,V,4,4)
            if tgt_extrs.shape[-2:] == (3, 4):
                pad = torch.zeros((*tgt_extrs.shape[:-2], 4, 4), dtype=tgt_extrs.dtype, device=tgt_extrs.device)
                pad[..., :3, :4] = tgt_extrs
                pad[..., 3, 3] = 1.0
                tgt_extrs_44 = pad
            else:
                tgt_extrs_44 = tgt_extrs

            if ref_extrs.shape[-2:] == (3, 4):
                pad = torch.zeros((*ref_extrs.shape[:-2], 4, 4), dtype=ref_extrs.dtype, device=ref_extrs.device)
                pad[..., :3, :4] = ref_extrs
                pad[..., 3, 3] = 1.0
                ref_extrs_44 = pad
            else:
                ref_extrs_44 = ref_extrs

            # Align each batch element to the same reference (batch=0) trajectory.
            # If ref/est have the same length, we can directly estimate a Sim(3) with Umeyama.
            # If lengths differ (e.g., recon vs NVS subset with no 1-1 correspondence), we fall back to a
            # simple "gauge mapping" using only camera-center mean/std (scale+translation, no rotation).
            ref_np_full = ref_extrs_44[0].detach().cpu().numpy()

            v_ref = ref_extrs_44.shape[1]
            v_est = tgt_extrs_44.shape[1]

            # Convert w2c -> c2w to access camera centers in world.
            with torch.no_grad():
                ref_c2w_full = torch.linalg.inv(ref_extrs_44[0])  # (Vref,4,4)
                ref_centers_full = ref_c2w_full[:, :3, 3]  # (Vref,3)

            aligned_batches = []
            for b in range(tgt_extrs_44.shape[0]):
                est_np_full = tgt_extrs_44[b].detach().cpu().numpy()

                if v_ref == v_est and v_ref >= 3:
                    # True Sim(3) alignment (requires same number of poses / 1-1 correspondence).
                    r, t, s, est_aligned_full = align_poses_umeyama(
                        ext_ref=ref_np_full,
                        ext_est=est_np_full,
                        return_aligned=True,
                        ransac=align_ransac,
                        random_state=42,
                    )
                    try:
                        angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))))
                        print(
                            "[gs_views][align] sim3(full): "
                            f"ref={v_ref}, est={v_est}, s={float(s):.6f}, |t|={float(np.linalg.norm(t)):.6f}, rot_deg={angle_deg:.6f}"
                        )
                    except Exception:
                        pass

                    aligned_batches.append(torch.from_numpy(est_aligned_full).to(tgt_extrs_44))
                    continue

                # Fallback: gauge mapping based on camera centers (no rotation).
                with torch.no_grad():
                    est_c2w_full = torch.linalg.inv(tgt_extrs_44[b])  # (Vest,4,4)
                    est_centers_full = est_c2w_full[:, :3, 3]  # (Vest,3)

                    mu_ref = ref_centers_full.mean(dim=0)
                    mu_est = est_centers_full.mean(dim=0)

                    # Use RMS radius as "sigma" (more stable than per-axis std for trajectories).
                    sigma_ref = (ref_centers_full - mu_ref).norm(dim=1).mean().clamp(min=1e-6)
                    sigma_est = (est_centers_full - mu_est).norm(dim=1).mean().clamp(min=1e-6)

                    s_map = (sigma_ref / sigma_est).item()
                    t_map = (mu_ref - s_map * mu_est).cpu().numpy()

                    pre_center_diff = (est_centers_full - mu_est).norm(dim=1).mean().item()
                    print(
                        "[gs_views][align] gauge_map: "
                        f"ref={v_ref}, est={v_est}, s={s_map:.6f}, |t|={float(np.linalg.norm(t_map)):.6f}, "
                        f"sigma_ref={sigma_ref.item():.6f}, sigma_est={sigma_est.item():.6f}, pre_mean_radius={pre_center_diff:.6f}"
                    )

                    # Apply mapping: C' = s*C + t
                    est_centers_aligned = est_centers_full * s_map + torch.as_tensor(
                        t_map, device=est_centers_full.device, dtype=est_centers_full.dtype
                    )

                    # Update c2w translations, keep rotations unchanged.
                    est_c2w_aligned = est_c2w_full.clone()
                    est_c2w_aligned[:, :3, 3] = est_centers_aligned

                    # Back to w2c
                    est_w2c_aligned = torch.linalg.inv(est_c2w_aligned)

                    # Print post mapping center diff stats vs ref distribution (not 1-1).
                    mu_aligned = est_centers_aligned.mean(dim=0)
                    sigma_aligned = (est_centers_aligned - mu_aligned).norm(dim=1).mean().item()
                    print(
                        "[gs_views][align] gauge_map(post): "
                        f"mu_ref={mu_ref.cpu().numpy()}, mu_est_aligned={mu_aligned.cpu().numpy()}, sigma_est_aligned={sigma_aligned:.6f}"
                    )

                aligned_batches.append(est_w2c_aligned.cpu())

            tgt_extrs = torch.stack(aligned_batches, dim=0).to(tgt_extrs_44)
    else:
        tgt_extrs = ref_extrs

    tgt_intrs = (
        intrinsics
        if intrinsics is not None
        else torch.from_numpy(prediction.intrinsics).unsqueeze(0).to(gs_world.means)
    )

    # if render resolution is not provided, render the input ones
    if out_image_hw is not None:
        H, W = out_image_hw
    else:
        H, W = prediction.depth.shape[-2:]

    # if single views, render wander trj
    if tgt_extrs.shape[1] <= 1:
        trj_mode = "wander"

    color, _depth = run_renderer_in_chunk_w_trj_mode(
        gaussians=gs_world,
        extrinsics=tgt_extrs,
        intrinsics=tgt_intrs,
        image_shape=(H, W),
        chunk_size=chunk_size,
        trj_mode=trj_mode,
        use_sh=True,
        color_mode=color_mode,
        enable_tqdm=enable_tqdm,
    )

    name = trj_mode if output_name is None else output_name
    base_dir = os.path.join(export_dir, "gs_views", name)
    os.makedirs(base_dir, exist_ok=True)

    saved: list[list[str]] = []

    # Expected: color = (B, T, C, H, W), float in [0,1]
    for b in range(color.shape[0]):
        view_dir = os.path.join(base_dir, f"view_{b:04d}")
        os.makedirs(view_dir, exist_ok=True)

        imgs_b = color[b].clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
        paths_b: list[str] = []
        for t, frame in enumerate(imgs_b):
            out_path = os.path.join(view_dir, f"{t:04d}.{image_format}")
            Image.fromarray(frame).save(out_path)
            paths_b.append(out_path)

        saved.append(paths_b)

    return saved
