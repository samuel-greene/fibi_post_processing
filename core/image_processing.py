#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk

import openslide
import pyvips

from skimage.registration import phase_cross_correlation
from skimage.transform import rotate
from skimage.filters import sobel
from scipy.ndimage import shift as nd_shift

# =========================
#  REGISTRATION HELPERS
# =========================

def load_reg_image(
    path: str,
    max_dim: int = 6000,
    target_downsample: Optional[float] = None,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        reg_rgb   : np.ndarray (H,W,3) uint8
        scale_yx  : np.ndarray([sy, sx]) mapping reg->level0 pixels
    """
    if path.lower().endswith(".svs"):
        slide = openslide.OpenSlide(path)

        if target_downsample is None:
            # highest-res level that fits max_dim
            level = 0
            for i, (wL, hL) in enumerate(slide.level_dimensions):
                if max(wL, hL) <= max_dim:
                    level = i
                    break
        else:
            # pick level closest to requested downsample, then ensure it fits max_dim
            level = slide.get_best_level_for_downsample(float(target_downsample))
            while level < slide.level_count - 1:
                wL, hL = slide.level_dimensions[level]
                if max(wL, hL) <= max_dim:
                    break
                level += 1

        wL, hL = slide.level_dimensions[level]
        region = slide.read_region((0, 0), level, (wL, hL)).convert("RGB")
        reg_rgb = np.array(region)

        w0, h0 = slide.level_dimensions[0]

        def _safe_scale(n0: int, nL: int) -> float:
            if n0 <= 1 or nL <= 1:
                return 1.0
            return (n0 - 1) / (nL - 1)

        sy = _safe_scale(h0, hL)
        sx = _safe_scale(w0, wL)
        scale_yx = np.array([sy, sx], dtype=np.float32)


        if debug:
            ds = slide.level_downsamples[level]
            print(f"{os.path.basename(path)} level {level} dims ({wL}, {hL}) downsample {ds} scale_yx {scale_yx}")

        return reg_rgb, scale_yx

    # ----- Non-SVS: downsample via Pillow -----
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    max0 = max(w0, h0)

    if max0 <= max_dim:
        reg_img = img
        reg_w, reg_h = w0, h0
    else:
        s = max0 / max_dim
        reg_w = int(round(w0 / s))
        reg_h = int(round(h0 / s))
        reg_img = img.resize((reg_w, reg_h), resample=Image.BILINEAR)

    reg_rgb = np.array(reg_img)
    def _safe_scale(n0: int, nL: int) -> float:
        if n0 <= 1 or nL <= 1:
            return 1.0
        return (n0 - 1) / (nL - 1)

    sy = _safe_scale(h0, reg_h)
    sx = _safe_scale(w0, reg_w)
    scale_yx = np.array([sy, sx], dtype=np.float32)

    return reg_rgb, scale_yx



def center_patch(img: np.ndarray, patch_size: Optional[int]) -> np.ndarray:
    """
    Take a central square patch of size patch_size x patch_size.
    If patch_size is None or larger than the image, return the full image.
    """
    if patch_size is None:
        return img
    h, w = img.shape
    ps = min(patch_size, h, w)
    ys = h // 2 - ps // 2
    xs = w // 2 - ps // 2
    return img[ys:ys + ps, xs:xs + ps]


def estimate_transform(
    backlit_filepath: str,
    fibi_filepath: str,
    max_dim: int = 6000,
    downsample_factor: int = 1,
    upsample_factor: int = 10,
    patch_size: Optional[int] = 2048,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Estimate rotation (deg) and translation (full-res pixels) that
    aligns FIBI (auto) to backlit.

    Strategy:
      - Load both images at a registration resolution (max_dim).
      - Compute Sobel features.
      - For each candidate angle:
          * Rotate FIBI features.
          * Use phase_cross_correlation to find best translation for that angle.
          * Compute mutual information on a central patch.
      - Return angle / translation that maximize MI.

    Returns:
        angle_deg  : float
        shift_full : np.ndarray([dy_full, dx_full]) in full-res pixels
                     (in the backlit frame).
    """

    # --- Load downsampled registration images ---
    back_reg_rgb, back_scale_yx = load_reg_image(backlit_filepath, max_dim=max_dim, debug=True)

    # Match FIBI registration downsample to backlit's (use mean sy/sx as the requested ds)
    target_ds = float(np.mean(back_scale_yx))
    auto_reg_rgb, auto_scale_yx = load_reg_image(fibi_filepath, max_dim=max_dim, target_downsample=target_ds, debug=True)


    if progress_callback:
        progress_callback(5.0)

    # --- Convert to grayscale ---
    def to_gray(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            return arr.mean(axis=2).astype(np.float32)
        return arr.astype(np.float32)

    auto_gray = to_gray(auto_reg_rgb)
    back_gray = to_gray(back_reg_rgb)

    # --- Crop to common size (defensive) ---
    h = min(auto_gray.shape[0], back_gray.shape[0])
    w = min(auto_gray.shape[1], back_gray.shape[1])
    auto_gray = auto_gray[:h, :w]
    back_gray = back_gray[:h, :w]

    # --- Sobel features ---
    auto_feat = sobel(auto_gray)
    back_feat = sobel(back_gray)

    ds = downsample_factor
    auto_feat_ds = auto_feat[::ds, ::ds]
    back_feat_ds = back_feat[::ds, ::ds]

    back_c = center_patch(back_feat_ds, patch_size)

    # --- Mutual information helper ---
    def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
        a_flat = a.ravel()
        b_flat = b.ravel()
        hgram, _, _ = np.histogram2d(a_flat, b_flat, bins=bins)
        pxy = hgram / np.sum(hgram)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        nz = pxy > 0
        return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px_py[nz] + 1e-12))))

    # --- Search over small rotations; use PCC for translation at each angle ---
    angle_range = (-0.5, 0.5)   # degrees
    angle_step = 0.1
    angles = np.arange(angle_range[0], angle_range[1] + 1e-9, angle_step)

    best: Optional[Tuple[float, float, float, float]] = None  # (mi, angle, dy_ds, dx_ds)
    total_steps = len(angles)
    step_count = 0

    for ang in angles:
        # Rotate downsampled auto features
        auto_rot = rotate(
            auto_feat_ds,
            angle=ang,
            resize=False,
            order=1,
            mode="constant",
            cval=0,
            preserve_range=True,
        )

        # PCC gives translation for this angle
        shift_ds, _, _ = phase_cross_correlation(
            back_feat_ds, auto_rot, upsample_factor=upsample_factor
        )
        dy_ds, dx_ds = float(shift_ds[0]), float(shift_ds[1])

        # Shift once
        auto_shift = nd_shift(
            auto_rot,
            shift=(dy_ds, dx_ds),
            order=1,
            mode="constant",
            cval=0,
        ) 

        # Compute MI on central patch
        auto_c = center_patch(auto_shift, patch_size)
        if auto_c.shape != back_c.shape:
            # Shouldn't usually happen, but be defensive
            continue

        mi = mutual_information(back_c, auto_c, bins=64)

        if best is None or mi > best[0]:
            best = (mi, float(ang), dy_ds, dx_ds)

        step_count += 1
        if progress_callback:
            progress_callback(10.0 + 70.0 * step_count / total_steps)

    if best is None:
        raise RuntimeError("MI search failed to find a valid transform.")

    best_mi, best_ang, best_dy_ds, best_dx_ds = best

    # from ds-space to registration pixels
    shift_reg = np.array([best_dy_ds, best_dx_ds], dtype=np.float32) * ds
    # from registration pixels to full-res pixels (backlit frame)
    shift_full = shift_reg * back_scale_yx  # dy*sy, dx*sx

    if progress_callback:
        progress_callback(85.0)

    return float(best_ang), shift_full


# =========================
#  LOW-RES DEBUG OVERLAY
# =========================

def save_lowres_debug_overlay(
    backlit_path: str,
    fibi_path: str,
    angle_deg: float,
    shift_full: np.ndarray,   # [dy_full, dx_full]
    max_dim: int,
    debug_output_path: str,
    backlit_opacity: float = 0.5,
) -> None:
    """
    Save a small overlay at the registration resolution to debug alignment.

    - Reloads downsampled RGB images via load_reg_image()
    - Converts full-res shift back into registration pixels
    - Applies rotate + shift at low resolution
    - Writes a simple overlay image
    """
    back_reg_rgb, back_scale_yx = load_reg_image(backlit_path, max_dim=max_dim)
    auto_reg_rgb, _ = load_reg_image(fibi_path, max_dim=max_dim, target_downsample=float(np.mean(back_scale_yx)))

    shift_reg = shift_full / back_scale_yx
    # 3) Crop to common size (defensive)
    h = min(back_reg_rgb.shape[0], auto_reg_rgb.shape[0])
    w = min(back_reg_rgb.shape[1], auto_reg_rgb.shape[1])
    back_reg_rgb = back_reg_rgb[:h, :w]
    auto_reg_rgb = auto_reg_rgb[:h, :w]

    # 4) Rotate + shift auto at registration resolution
    auto_rot = np.zeros_like(auto_reg_rgb, dtype=np.float32)
    for c in range(auto_reg_rgb.shape[2]):
        auto_rot[..., c] = rotate(
            auto_reg_rgb[..., c],
            angle=angle_deg,
            resize=False,
            order=1,
            mode="constant",
            cval=0,
            preserve_range=True,
        )

    auto_shifted = np.zeros_like(auto_rot, dtype=np.float32)
    dy_reg, dx_reg = float(shift_reg[0]), float(shift_reg[1])
    for c in range(auto_rot.shape[2]):
        auto_shifted[..., c] = nd_shift(
            auto_rot[..., c],
            shift=(dy_reg, dx_reg),
            order=1,
            mode="constant",
            cval=0,
        )

    # 5) Blend + save
    back_f = back_reg_rgb.astype(np.float32)
    auto_f = auto_shifted.astype(np.float32)

    alpha = float(backlit_opacity)
    overlay = back_f * alpha + auto_f * (1.0 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    Image.fromarray(overlay).save(debug_output_path)


# =========================
#  FULL-RES OVERLAY (pyvips)
# =========================

import math
import os
import pyvips
import numpy as np
import openslide

def _vips_load_rgb(path: str) -> pyvips.Image:
    """
    Robust SVS loader across libvips/pyvips versions.

    - Tries openslideload(level=0, autocrop=False) if supported
    - Falls back to openslideload(level=0)
    - Falls back to new_from_file()
    - Normalizes output to 3-band RGB (drops alpha)
    """
    is_svs = path.lower().endswith(".svs")

    v = None

    if is_svs and hasattr(pyvips.Image, "openslideload"):
        loader = pyvips.Image.openslideload

        # Try the most explicit signature first, then back off
        for kwargs in (
            {"level": 0, "autocrop": False},
            {"level": 0},
            {},
        ):
            try:
                v = loader(path, **kwargs)
                break
            except TypeError:
                # this build doesn't support one of these kwargs
                v = None
            except Exception:
                # loader exists but failed for some other reason; we'll fallback
                v = None

    if v is None:
        # Generic fallback (works for SVS too, depending on your build)
        v = pyvips.Image.new_from_file(path, access="sequential")

    # Normalize to RGB
    if v.bands >= 4:
        v = v.extract_band(0, n=3)       # drop alpha/extra
    elif v.bands == 1:
        v = v.bandjoin(v).bandjoin(v)    # gray -> RGB

    return v


def write_pyramidal_overlay(
    backlit_path: str,
    fibi_path: str,
    angle_deg: float,
    shift_full: np.ndarray,   # [dy_full, dx_full]
    backlit_opacity: float,   # 0..1
    output_path: str,
    jpeg_q: int = 90,
) -> None:
    """
    Apply the estimated transform at full resolution using pyvips
    and write a pyramidal tiled BigTIFF (JPEG compression).
    """

    dy_full = float(shift_full[0])
    dx_full = float(shift_full[1])

    # Load in a consistent coordinate frame
    back_v = _vips_load_rgb(backlit_path)
    auto_v = _vips_load_rgb(fibi_path)

    # Crop both to a common canvas (top-left aligned) like your low-res path
    width = min(back_v.width, auto_v.width)
    height = min(back_v.height, auto_v.height)
    back_v = back_v.crop(0, 0, width, height)
    auto_v = auto_v.crop(0, 0, width, height)

    backf = back_v.cast("float")
    autof = auto_v.cast("float")

    theta = math.radians(-angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    # forward_like (matches your debug/skimage convention in your latest tester)
    matrix = [cos_t, -sin_t,
              sin_t,  cos_t]

    odx = dx_full + cx
    ody = dy_full + cy

    interp = pyvips.Interpolate.new("bicubic")

    # CRITICAL: force a fixed output canvas so we don't "accidentally translate"
    try:
        auto_aligned = autof.affine(
            matrix,
            odx=odx,
            ody=ody,
            idx=-cx,
            idy=-cy,
            interpolate=interp,
            extend="black",
            oarea=[0, 0, width, height],
        )
    except TypeError:
        # Older libvips: no oarea
        auto_aligned = autof.affine(
            matrix,
            odx=odx,
            ody=ody,
            interpolate=interp,
            extend="black",
        )

    # Ensure exact output size no matter what
    if auto_aligned.width != width or auto_aligned.height != height:
        if auto_aligned.width >= width and auto_aligned.height >= height:
            auto_aligned = auto_aligned.crop(0, 0, width, height)
        else:
            auto_aligned = auto_aligned.embed(0, 0, width, height, extend="black")

    alpha = float(backlit_opacity)
    overlay = backf * alpha + auto_aligned * (1.0 - alpha)
    overlay = overlay.cast("uchar")

    overlay.write_to_file(
        output_path,
        tile=True,
        tile_width=512,
        tile_height=512,
        pyramid=True,
        bigtiff=True,
        compression="jpeg",
        Q=jpeg_q,
    )

def create_overlay(fibi_path: str, backlit_path: str, output_path: str, opacity: float) -> str:
    """
    Simple wrapper to create an overlay image using default parameters.
    Returns the output path on success.
    """
    max_dim = 6000  # registration image max dimension

    angle_deg, shift_full = estimate_transform(
        backlit_path,
        fibi_path,
        max_dim=max_dim,
        downsample_factor=1,
        upsample_factor=10,
        patch_size=2048,
    )

    # Low-res debug overlay
    base, ext = os.path.splitext(output_path)
    debug_path = base + "_reg_debug.png"
    save_lowres_debug_overlay(
        backlit_path,
        fibi_path,
        angle_deg,
        shift_full,
        max_dim=max_dim,
        debug_output_path=debug_path,
        backlit_opacity=opacity,
    )

    # Full-res overlay (pyvips)
    write_pyramidal_overlay(
        backlit_path,
        fibi_path,
        angle_deg,
        shift_full,
        opacity,
        output_path,
        jpeg_q=90,
    )

    return output_path