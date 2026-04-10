#!/usr/bin/env python3
"""
registration_qc.py — Quality check for MRI image registration.

Loads a fixed (reference) and a moving (registered) NIfTI image, computes
several voxel-based similarity metrics, and produces visualization figures
(checkerboard, edge-overlay, difference map) for each anatomical plane.
An optional brain mask restricts all metric calculations to the region of
interest.

Outputs
-------
output/metrics.json     — JSON file with all computed metrics
output/qc_<view>.png    — one PNG per view (axial / sagittal / coronal)
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy.ndimage import sobel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_nifti(path: str) -> nib.Nifti1Image:
    """Load a NIfTI file and return a Nifti1Image."""
    img = nib.load(path)
    return nib.as_closest_canonical(img)


def extract_3d(img: nib.Nifti1Image) -> np.ndarray:
    """Return the 3-D data array.  For 4-D images the mean volume is used."""
    data = np.asarray(img.dataobj, dtype=np.float64)
    if data.ndim == 4:
        warnings.warn(
            "4-D image detected; using mean across volumes for QC metrics."
        )
        data = data.mean(axis=-1)
    elif data.ndim != 3:
        raise ValueError(f"Expected 3-D or 4-D data, got shape {data.shape}")
    return data


def resample_to_reference(
    moving_img: nib.Nifti1Image, fixed_img: nib.Nifti1Image
) -> np.ndarray:
    """
    Resample the moving image voxel grid into the fixed image voxel grid using
    affine-based trilinear interpolation (scipy.ndimage.map_coordinates).
    Returns the resampled 3-D array.
    """
    from scipy.ndimage import map_coordinates

    fixed_shape = fixed_img.shape[:3]
    fixed_affine = fixed_img.affine
    moving_affine = moving_img.affine

    # World coordinates of every voxel in fixed space
    i, j, k = np.meshgrid(
        np.arange(fixed_shape[0]),
        np.arange(fixed_shape[1]),
        np.arange(fixed_shape[2]),
        indexing="ij",
    )
    ijk_fixed = np.stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])

    # Map to world, then to moving voxel space
    world = fixed_affine @ ijk_fixed                     # 4 × N
    ijk_moving = np.linalg.inv(moving_affine) @ world    # 4 × N

    moving_data = extract_3d(moving_img)
    resampled = map_coordinates(
        moving_data,
        ijk_moving[:3],
        order=1,
        mode="constant",
        cval=0.0,
    )
    return resampled.reshape(fixed_shape)


def normalize(arr: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Linearly scale array values to [0, 1] inside the mask."""
    if mask is not None:
        vmin = arr[mask].min()
        vmax = arr[mask].max()
    else:
        vmin = arr.min()
        vmax = arr.max()
    denom = vmax - vmin
    if denom == 0:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - vmin) / denom


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def compute_mse(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray = None) -> float:
    """Mean Squared Error (lower is better)."""
    if mask is not None:
        diff = fixed[mask] - moving[mask]
    else:
        diff = fixed.ravel() - moving.ravel()
    return float(np.mean(diff ** 2))


def compute_ncc(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Normalised Cross-Correlation in [-1, 1].
    Values close to 1 indicate good alignment.
    """
    if mask is not None:
        f = fixed[mask].astype(np.float64)
        m = moving[mask].astype(np.float64)
    else:
        f = fixed.ravel().astype(np.float64)
        m = moving.ravel().astype(np.float64)

    f -= f.mean()
    m -= m.mean()
    denom = np.sqrt((f ** 2).sum() * (m ** 2).sum())
    if denom == 0:
        return 0.0
    return float((f * m).sum() / denom)


def compute_nmi(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray = None, bins: int = 64) -> float:
    """
    Normalised Mutual Information = (H(F) + H(M)) / H(F, M).
    Values > 1 indicate shared information; higher means better alignment.
    """
    if mask is not None:
        f = fixed[mask].ravel()
        m = moving[mask].ravel()
    else:
        f = fixed.ravel()
        m = moving.ravel()

    hist2d, _, _ = np.histogram2d(f, m, bins=bins)
    hist2d = hist2d / hist2d.sum()

    # Marginals
    pf = hist2d.sum(axis=1)
    pm = hist2d.sum(axis=0)

    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    hf = entropy(pf)
    hm = entropy(pm)
    hfm = entropy(hist2d.ravel())

    if hfm == 0:
        return 0.0
    return float((hf + hm) / hfm)


def compute_ssim(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Structural Similarity Index (SSIM) following Wang et al. (2004).
    Returns the mean SSIM over the entire volume (or mask).
    Values in [-1, 1]; closer to 1 is better.
    """
    f = fixed.astype(np.float64)
    m = moving.astype(np.float64)

    # Normalise to [0, 1]
    f = normalize(f, mask)
    m = normalize(m, mask)

    C1, C2 = 0.01 ** 2, 0.03 ** 2

    mu_f = f.mean() if mask is None else f[mask].mean()
    mu_m = m.mean() if mask is None else m[mask].mean()
    sigma_f2 = f.var() if mask is None else f[mask].var()
    sigma_m2 = m.var() if mask is None else m[mask].var()
    sigma_fm = (
        ((f - mu_f) * (m - mu_m)).mean()
        if mask is None
        else ((f - mu_f) * (m - mu_m))[mask].mean()
    )

    numerator = (2 * mu_f * mu_m + C1) * (2 * sigma_fm + C2)
    denominator = (mu_f ** 2 + mu_m ** 2 + C1) * (sigma_f2 + sigma_m2 + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0


def compute_overlap(fixed_mask: np.ndarray, moving_mask: np.ndarray) -> dict:
    """
    Dice coefficient and Jaccard index between two binary masks.
    Only meaningful when explicit ROI masks are provided.
    """
    intersection = np.logical_and(fixed_mask, moving_mask).sum()
    union = np.logical_or(fixed_mask, moving_mask).sum()
    sum_sizes = fixed_mask.sum() + moving_mask.sum()

    dice = float(2 * intersection / sum_sizes) if sum_sizes > 0 else 0.0
    jaccard = float(intersection / union) if union > 0 else 0.0
    return {"dice": dice, "jaccard": jaccard}


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

VIEWS = {
    "axial":    2,
    "coronal":  1,
    "sagittal": 0,
}


def _mid_slice(arr: np.ndarray, axis: int, mask: np.ndarray = None) -> int:
    """
    Return the slice index along *axis* that contains the most foreground
    voxels (or the centre of the mask extent if a mask is given).
    """
    if mask is not None:
        coords = np.where(mask)
        lo = coords[axis].min()
        hi = coords[axis].max()
        return int((lo + hi) // 2)
    return arr.shape[axis] // 2


def _get_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    return np.take(arr, idx, axis=axis)


def _edge_map(arr2d: np.ndarray) -> np.ndarray:
    """Sobel edge magnitude of a 2-D slice."""
    sx = sobel(arr2d, axis=0)
    sy = sobel(arr2d, axis=1)
    return np.hypot(sx, sy)


def _checkerboard(a: np.ndarray, b: np.ndarray, n: int = 8) -> np.ndarray:
    """Interleave two 2-D arrays in a checkerboard pattern (n tiles per axis)."""
    rows, cols = a.shape
    tile_r = max(1, rows // n)
    tile_c = max(1, cols // n)
    mask = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if (r // tile_r + c // tile_c) % 2 == 0:
                mask[r, c] = True
    out = np.where(mask, a, b)
    return out


def plot_qc_figure(
    fixed_data: np.ndarray,
    moving_data: np.ndarray,
    axis: int,
    view_name: str,
    mask_data: np.ndarray = None,
    output_path: str = "qc.png",
):
    """
    Produce a 4-panel QC figure for one anatomical view.

    Panels
    ------
    1. Fixed image
    2. Moving (registered) image
    3. Checkerboard overlay
    4. Absolute difference map
    """
    idx = _mid_slice(fixed_data, axis, mask_data)

    f_sl = _get_slice(fixed_data, axis, idx)
    m_sl = _get_slice(moving_data, axis, idx)

    # Normalise each slice independently for display
    f_norm = normalize(f_sl)
    m_norm = normalize(m_sl)

    diff = np.abs(f_norm - m_norm)
    checker = _checkerboard(f_norm, m_norm)

    # Edge overlay (fixed edges in green on moving image)
    edges = _edge_map(f_norm)
    edges_norm = normalize(edges)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Registration QC — {view_name} view (slice {idx})", fontsize=13)

    cmap_gray = "gray"

    ax = axes[0]
    ax.imshow(f_norm.T, cmap=cmap_gray, origin="lower", aspect="auto")
    ax.set_title("Fixed")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(m_norm.T, cmap=cmap_gray, origin="lower", aspect="auto")
    ax.set_title("Moving (registered)")
    ax.axis("off")

    ax = axes[2]
    ax.imshow(checker.T, cmap=cmap_gray, origin="lower", aspect="auto")
    ax.imshow(
        np.ma.masked_where(edges_norm.T < 0.15, edges_norm.T),
        cmap="Greens",
        alpha=0.6,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    ax.set_title("Checkerboard + fixed edges")
    ax.axis("off")

    ax = axes[3]
    im = ax.imshow(diff.T, cmap="hot", origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Absolute difference")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_qc(
    fixed_path: str,
    moving_path: str,
    mask_path: str = None,
    output_dir: str = "output",
    n_checkerboard_tiles: int = 8,
):
    """
    Full QC pipeline: load images, resample, compute metrics, save figures.

    Parameters
    ----------
    fixed_path  : Path to the fixed (reference) NIfTI image.
    moving_path : Path to the moving (registered) NIfTI image.
    mask_path   : Optional path to a binary brain mask in fixed space.
    output_dir  : Directory where all outputs are written.
    n_checkerboard_tiles : Number of checkerboard tiles per axis.

    Returns
    -------
    dict with all computed metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[QC] Loading fixed image:  {fixed_path}")
    fixed_img = load_nifti(fixed_path)
    fixed_data = extract_3d(fixed_img)

    print(f"[QC] Loading moving image: {moving_path}")
    moving_img = load_nifti(moving_path)

    # Resample moving into fixed voxel grid if grids differ
    fixed_shape = fixed_img.shape[:3]
    moving_shape = moving_img.shape[:3]
    grids_match = (fixed_shape == moving_shape) and np.allclose(
        fixed_img.affine, moving_img.affine, atol=1e-3
    )

    if grids_match:
        print("[QC] Grids match — no resampling needed.")
        moving_data = extract_3d(moving_img)
    else:
        print("[QC] Resampling moving image into fixed voxel grid ...")
        moving_data = resample_to_reference(moving_img, fixed_img)

    # Load mask
    mask = None
    if mask_path and mask_path not in ("null", "None", ""):
        print(f"[QC] Loading mask:         {mask_path}")
        mask_img = load_nifti(mask_path)
        mask = extract_3d(mask_img).astype(bool)
        if mask.shape != fixed_shape:
            print("[QC] Resampling mask into fixed voxel grid ...")
            mask_img_rs = nib.Nifti1Image(
                mask.astype(np.float32), mask_img.affine
            )
            mask_rs = resample_to_reference(mask_img_rs, fixed_img)
            mask = mask_rs > 0.5

    # Compute metrics
    print("[QC] Computing similarity metrics ...")
    metrics = {}
    metrics["nmi"] = compute_nmi(fixed_data, moving_data, mask)
    metrics["ncc"] = compute_ncc(fixed_data, moving_data, mask)
    metrics["mse"] = compute_mse(fixed_data, moving_data, mask)
    metrics["ssim"] = compute_ssim(fixed_data, moving_data, mask)

    # Interpretation helpers (qualitative thresholds, rough guidelines)
    metrics["quality"] = _quality_label(metrics)

    print(f"[QC]   NMI  = {metrics['nmi']:.4f}")
    print(f"[QC]   NCC  = {metrics['ncc']:.4f}")
    print(f"[QC]   MSE  = {metrics['mse']:.4f}")
    print(f"[QC]   SSIM = {metrics['ssim']:.4f}")
    print(f"[QC]   Quality label: {metrics['quality']}")

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=4)
    print(f"[QC] Metrics saved to {metrics_path}")

    # Produce visualization figures
    print("[QC] Generating QC figures ...")
    generated_figures = []
    for view_name, axis in VIEWS.items():
        out_png = os.path.join(output_dir, f"qc_{view_name}.png")
        plot_qc_figure(
            fixed_data,
            moving_data,
            axis=axis,
            view_name=view_name,
            mask_data=mask,
            output_path=out_png,
        )
        generated_figures.append(out_png)
        print(f"[QC]   Saved {out_png}")

    print("[QC] Done.")
    return metrics


def _quality_label(metrics: dict) -> str:
    """
    Rough qualitative label based on combined metric thresholds.
    These thresholds are conservative guidelines, not clinical standards.
    """
    ncc = metrics.get("ncc", 0)
    nmi = metrics.get("nmi", 0)
    ssim = metrics.get("ssim", 0)

    score = 0
    if ncc > 0.90:
        score += 2
    elif ncc > 0.75:
        score += 1

    if nmi > 1.20:
        score += 2
    elif nmi > 1.10:
        score += 1

    if ssim > 0.90:
        score += 2
    elif ssim > 0.75:
        score += 1

    if score >= 5:
        return "excellent"
    elif score >= 3:
        return "good"
    elif score >= 1:
        return "fair"
    else:
        return "poor"


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Registration QC: compute metrics and generate QC figures."
    )
    parser.add_argument("--fixed",   required=True, help="Fixed (reference) NIfTI image")
    parser.add_argument("--moving",  required=True, help="Moving (registered) NIfTI image")
    parser.add_argument("--mask",    default=None,  help="Optional brain mask in fixed space")
    parser.add_argument("--outdir",  default="output", help="Output directory (default: output/)")
    parser.add_argument(
        "--checkerboard-tiles", type=int, default=8,
        dest="tiles",
        help="Number of checkerboard tiles per axis (default: 8)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_qc(
        fixed_path=args.fixed,
        moving_path=args.moving,
        mask_path=args.mask,
        output_dir=args.outdir,
        n_checkerboard_tiles=args.tiles,
    )
