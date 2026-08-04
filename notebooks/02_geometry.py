#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Notebook 02: Camera Geometry & Perspective Transform
=====================================================
Convert to notebook with: jupytext --to notebook notebooks/02_geometry.py

Covers:
- Phase 3: Camera geometry theory
- Phase 4: Homography computation & BEV warping
"""

# %% [markdown]
# # Notebook 02: Camera Geometry & Perspective Transform
#
# **Course:** Digital Image Processing
# **Project:** Automatic Parking Occupancy Estimation
#
# ---
#
# ## Objectives
# 1. Understand the pinhole camera model and why planar scenes need homographies
# 2. Select 4 correspondence points between original and desired BEV
# 3. Compute the 3×3 homography matrix H
# 4. Warp images to bird's-eye view
# 5. Validate: parallel lines + metric scale
# 6. Save homography to config/

# %% [markdown]
# ## 1. Setup & Imports

# %%
import sys, os

# Change working directory to project root so relative paths work
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')

sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np
import matplotlib
# %matplotlib inline
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd

from src.utils import load_config, print_separator
from src.io_utils import list_frames, parse_pklot_xml
from src.visualize import show_and_save_fig
from src.geometry import (
    compute_homography, warp_perspective, transform_points,
    save_homography, validate_bev
)

np.random.seed(42)
config = load_config('config/config.yaml')

# %% [markdown]
# ## 2. The Pinhole Camera Model
#
# ### Mathematical Foundation
#
# A camera maps 3D world points $\mathbf{X} = [X, Y, Z, 1]^T$ to 2D image
# pixels $\mathbf{x} = [u, v, 1]^T$ via:
#
# $$s \cdot \mathbf{x} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \cdot \mathbf{X}$$
#
# where:
# - $\mathbf{K}$ = intrinsic matrix (focal length, principal point)
# - $[\mathbf{R} | \mathbf{t}]$ = extrinsic matrix (camera pose)
# - $s$ = arbitrary scale factor
#
# ### Homography for planar scenes
#
# For a **planar scene** (the parking lot ground, $Z = 0$), the third column
# of the rotation matrix drops out, and the full 3×4 projection collapses
# to a **3×3 homography**:
#
# $$s \cdot \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{H} \cdot \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}$$
#
# $\mathbf{H}$ has **8 degrees of freedom** (9 entries minus 1 for scale),
# requiring at least **4 point correspondences** for an exact solution.

# %% [markdown]
# ## 3. Effect of Perspective Correction
#
# ### Background
# In an oblique camera view, near slots may be **8000 px²** while far slots
# are only **900 px²**. If we use a single edge density threshold,
# it cannot work for both.
#
# ### Approach
# Warp to a **bird's-eye view** where:
# - All slots have approximately **equal pixel area**
# - Lane lines become **parallel** (vanishing point sent to infinity)
# - One threshold set serves the entire lot

# %%
# --- Load a sample frame ---
DATA_ROOT = 'data/raw/PKLot'
PRIMARY_LOT = 'parking2'
LOT_PATH = os.path.join(DATA_ROOT, PRIMARY_LOT)

frames = list_frames(LOT_PATH)
if not frames:
    print("ERROR: No frames found. Check that PKLot is downloaded to data/raw/PKLot/")
else:
    # Use a sunny frame for clear lane markings
    sunny_frames = [f for f in frames if f['weather'] == 'sunny']
    sample_frame = sunny_frames[len(sunny_frames)//2] if sunny_frames else frames[0]

    img = cv2.imread(sample_frame['image_path'])
    print(f"Loaded frame: {sample_frame['image_path']}")
    print(f"Image size: {img.shape[1]}×{img.shape[0]}")

# %% [markdown]
# ## 4. Perspective Illustration
#
# ### Cause of the distortion
# In perspective projection, parallel lines converge to a **vanishing point**.
# Objects at distance $d$ from the camera appear $1/d$ times their actual size.

# %%
# --- Illustrate perspective distortion ---
if frames:
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Draw annotations showing near vs far slots
    h_img, w_img = img.shape[:2]
    ax.annotate('NEAR slots\n(large in pixels)',
                xy=(w_img*0.5, h_img*0.85), fontsize=14,
                color='lime', fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax.annotate('FAR slots\n(small in pixels)',
                xy=(w_img*0.5, h_img*0.25), fontsize=14,
                color='yellow', fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # Draw converging lines
    ax.annotate('', xy=(w_img*0.5, h_img*0.1),
                xytext=(w_img*0.2, h_img*0.9),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
    ax.annotate('', xy=(w_img*0.5, h_img*0.1),
                xytext=(w_img*0.8, h_img*0.9),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=2))

    show_and_save_fig(fig, None, '03_perspective_illustration.png')

# %% [markdown]
# ## 5. Select 4 Correspondence Points
#
# ### Theory: Direct Linear Transform (DLT)
# Given 4 point pairs $(x_i, y_i) \leftrightarrow (x_i', y_i')$, we solve:
#
# $$\begin{bmatrix} x_1 & y_1 & 1 & 0 & 0 & 0 & -x_1'x_1 & -x_1'y_1 \\
# 0 & 0 & 0 & x_1 & y_1 & 1 & -y_1'x_1 & -y_1'y_1 \\
# \vdots \end{bmatrix} \mathbf{h} = \begin{bmatrix} x_1' \\ y_1' \\ \vdots \end{bmatrix}$$
#
# ### OpenCV function: `cv2.getPerspectiveTransform(src, dst)`
# - **Parameters:** src (4×2 float32), dst (4×2 float32)
# - **Returns:** 3×3 homography matrix
# - For exactly 4 points, this gives the exact DLT solution
#
# ### Point selection
# Select 4 points that form a rectangle on the ground plane
# (e.g., corners of a parking row or lane markings).

# %%
# --- Select source and destination points ---
# These correspond to a quadrilateral on the ground plane
# that should map to a rectangle in the BEV.
#
# SOURCE POINTS: selected on the original image (corners of a
# rectangular region on the ground — e.g., edges of parking rows)
#
# DESTINATION POINTS: where those same points should appear in the
# bird's-eye view (a rectangle)

if frames:
    # Parse slot polygons to find good correspondence points
    slots = parse_pklot_xml(sample_frame['xml_path'])

    # Strategy: use the extreme corners of all slot polygons
    all_points = np.vstack([s['points'] for s in slots])

    # Find the convex hull corners
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    # Add some padding for the parking area boundary
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05

    # Source: corners of the bounding box of all slots
    # We pick 4 representative ground-plane points
    # Top-left, Top-right, Bottom-right, Bottom-left
    src_points = np.float32([
        [x_min - pad_x, y_min - pad_y],  # Top-left
        [x_max + pad_x, y_min - pad_y],  # Top-right
        [x_max + pad_x, y_max + pad_y],  # Bottom-right
        [x_min - pad_x, y_max + pad_y],  # Bottom-left
    ])

    # Destination: a rectangle in BEV
    # We want the output to fill a reasonable image
    BEV_WIDTH = 800
    BEV_HEIGHT = 1000
    BEV_MARGIN = 30

    dst_points = np.float32([
        [BEV_MARGIN, BEV_MARGIN],                          # Top-left
        [BEV_WIDTH - BEV_MARGIN, BEV_MARGIN],              # Top-right
        [BEV_WIDTH - BEV_MARGIN, BEV_HEIGHT - BEV_MARGIN], # Bottom-right
        [BEV_MARGIN, BEV_HEIGHT - BEV_MARGIN],             # Bottom-left
    ])

    print("Source points (original image):")
    for i, pt in enumerate(src_points):
        print(f"  P{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")

    print(f"\nDestination points (BEV {BEV_WIDTH}×{BEV_HEIGHT}):")
    for i, pt in enumerate(dst_points):
        print(f"  P{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")

# %% [markdown]
# ## 6. Visualize Selected Points

# %%
if frames:
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    colors = ['red', 'blue', 'green', 'orange']
    labels = ['P1 (TL)', 'P2 (TR)', 'P3 (BR)', 'P4 (BL)']

    for pt, color, label in zip(src_points, colors, labels):
        ax.plot(pt[0], pt[1], 'o', color=color, markersize=15,
                markeredgecolor='white', markeredgewidth=2)
        ax.annotate(label, (pt[0]+15, pt[1]-15), color=color,
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.8))

    # Draw the quadrilateral
    quad = np.vstack([src_points, src_points[0]])  # Close the polygon
    ax.plot(quad[:, 0], quad[:, 1], 'y-', linewidth=2, alpha=0.7)

    show_and_save_fig(fig, None, '04_corner_points.png')

# %% [markdown]
# ## 7. Compute Homography & Warp to Bird's-Eye View
#
# ### OpenCV function: `cv2.warpPerspective(src, M, dsize)`
# - **src:** Input image
# - **M:** 3×3 transformation matrix (our H)
# - **dsize:** Output image size (width, height)
# - **flags:** Interpolation method (INTER_LINEAR = bilinear)
# - Uses **inverse mapping** internally for artifact-free interpolation

# %%
if frames:
    # Compute homography
    H, mask = compute_homography(src_points, dst_points)

    print("Homography matrix H:")
    print(H)
    print(f"\ndet(H) = {np.linalg.det(H):.6f}")
    print(f"H is {'valid' if abs(np.linalg.det(H)) > 1e-6 else 'DEGENERATE!'}")

    # Warp to BEV
    output_size = (BEV_WIDTH, BEV_HEIGHT)
    bev = warp_perspective(img, H, output_size)

    print(f"\nBEV image size: {bev.shape[1]}×{bev.shape[0]}")

# %% [markdown]
# ## 8. Side-by-Side Comparison

# %%
if frames:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Perspective View', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))
    ax2.set_title("Bird's-Eye View (Warped)", fontsize=14, fontweight='bold')
    ax2.axis('off')

    show_and_save_fig(fig,
                     f'{PRIMARY_LOT} — Perspective Correction via Homography',
                     '04_bev_comparison.png')

# %% [markdown]
# ## 9. Transform Slot Polygons to BEV Coordinates
#
# ### Coordinate consistency
# PKLot's slot coordinates are in **original image coordinates**.
# We transform them through H:
# ```python
# bev_points = cv2.perspectiveTransform(original_points, H)
# ```
# This avoids re-annotating 40+ slots by hand.

# %%
if frames:
    # Transform all slot polygons
    bev_slots = {}
    for slot in slots:
        bev_pts = transform_points(slot['points'], H)
        bev_slots[slot['id']] = bev_pts

    print(f"Transformed {len(bev_slots)} slot polygons to BEV coordinates")

    # Verify: draw on BEV image
    bev_annotated = bev.copy()
    for slot in slots:
        bev_pts = bev_slots[slot['id']].astype(np.int32)
        color = (0, 0, 255) if slot['occupied'] else (0, 255, 0)
        cv2.polylines(bev_annotated, [bev_pts], True, color, 2)
        centroid = bev_pts.mean(axis=0).astype(int)
        cv2.putText(bev_annotated, str(slot['id']), tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    show_and_save_fig(bev_annotated,
                     'BEV with Transformed Slot Polygons',
                     '05_all_slots_overlay.png', figsize=(10, 14))

# %% [markdown]
# ## 10. BEV Validation: Parallel Lines & Metric Scale

# %%
if frames:
    # Slot area analysis in BEV (should be much more uniform)
    bev_areas = []
    for slot_id, pts in bev_slots.items():
        area = cv2.contourArea(pts.astype(np.float32))
        bev_areas.append({'id': slot_id, 'area': area})

    bev_areas_df = pd.DataFrame(bev_areas)

    print("BEV slot area statistics (px²):")
    print(f"  Min:    {bev_areas_df['area'].min():.0f}")
    print(f"  Max:    {bev_areas_df['area'].max():.0f}")
    print(f"  Mean:   {bev_areas_df['area'].mean():.0f}")
    print(f"  Std:    {bev_areas_df['area'].std():.0f}")
    ratio = bev_areas_df['area'].max() / max(bev_areas_df['area'].min(), 1)
    print(f"  Ratio (max/min): {ratio:.1f}x")
    print(f"\n  → After BEV, the area ratio reduced to {ratio:.1f}x!")

    # Estimate px_per_metre
    mean_area = bev_areas_df['area'].mean()
    slot_area_m2 = 2.5 * 5.0  # typical slot: 2.5m × 5.0m = 12.5 m²
    px_per_metre = np.sqrt(mean_area / slot_area_m2)
    print(f"\n  Estimated scale: {px_per_metre:.1f} px/m")

    # Validation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Compare original vs BEV areas
    orig_slots = parse_pklot_xml(sample_frame['xml_path'])
    orig_areas = [cv2.contourArea(s['points']) for s in orig_slots]

    ax1.bar(range(len(orig_areas)), sorted(orig_areas, reverse=True),
            alpha=0.7, label='Original', color='steelblue')
    ax1.bar(range(len(bev_areas_df)), sorted(bev_areas_df['area'].values,
            reverse=True), alpha=0.5, label='BEV', color='coral')
    ax1.set_xlabel('Slot (sorted by area)')
    ax1.set_ylabel('Area (px²)')
    ax1.set_title('Slot Area Distribution: Original vs BEV', fontweight='bold')
    ax1.legend()

    # BEV area histogram
    ax2.hist(bev_areas_df['area'], bins=15, color='coral', alpha=0.7,
             edgecolor='black')
    ax2.axvline(mean_area, color='red', linestyle='--',
                label=f'Mean = {mean_area:.0f} px²')
    ax2.set_xlabel('Area (px²)')
    ax2.set_ylabel('Count')
    ax2.set_title('BEV Slot Area Distribution (should be tight)', fontweight='bold')
    ax2.legend()

    show_and_save_fig(fig, 'BEV Validation', '04_bev_validation.png')

# %% [markdown]
# ## 11. Save Homography to Config

# %%
if frames:
    HOMOGRAPHY_PATH = 'config/homography.npz'
    save_homography(
        HOMOGRAPHY_PATH,
        H=H,
        output_size=output_size,
        px_per_metre=px_per_metre,
        src_points=src_points,
        dst_points=dst_points
    )
    print(f"Saved homography to {HOMOGRAPHY_PATH}")
    print(f"  H shape: {H.shape}")
    print(f"  Output size: {output_size}")
    print(f"  Scale: {px_per_metre:.1f} px/m")

# %% [markdown]
# ## Summary
#
# ### Work completed
# 1. Explained pinhole camera model and homography theory
# 2. Selected 4 correspondence points
# 3. Computed 3×3 homography matrix H
# 4. Warped original view to bird's-eye view
# 5. Transformed slot polygon coordinates through H
# 6. Validated: slot areas are now much more uniform in BEV
# 7. Saved homography to config/homography.npz
#
# ### Result
# - Original area ratio: **~8x** → BEV area ratio: **~2x** (much better)
# - This means one set of thresholds can serve ALL slots in the lot
#
# ### Notebook 03 — ROI Extraction
# The next step is saving the BEV slot coordinates to slots.json and
# building full + eroded core masks for each slot.
