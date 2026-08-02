#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Notebook 04: Image Preprocessing Pipeline
==========================================
Convert to notebook with: jupytext --to notebook notebooks/04_preprocessing.py

Covers:
- Phase 6: Grayscale conversion, histogram equalization, CLAHE,
           Gaussian blur, median filter
"""

# %% [markdown]
# # Notebook 04: Image Preprocessing Pipeline
#
# **Course:** Digital Image Processing
# **Project:** Automatic Parking Occupancy Estimation
#
# ---
#
# ## Objectives
# 1. Understand why preprocessing is necessary before feature extraction
# 2. Walk through each step: Grayscale → CLAHE → Blur → Denoise
# 3. Compare histograms before and after CLAHE
# 4. Visualize the "preprocessing ladder" for occupied vs vacant slots
# 5. Demonstrate noise reduction with edge preservation

# %% [markdown]
# ## 1. Setup & Imports

# %%
import sys, os

# Ensure working directory is project root
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

from src.utils import load_config, print_separator
from src.io_utils import list_frames, parse_pklot_xml
from src.geometry import load_homography, warp_perspective, transform_points
from src.roi import extract_slot_image, create_eroded_core_mask, load_slots_json
from src.preprocessing import (
    to_grayscale, equalize_histogram, apply_clahe,
    apply_gaussian_blur, apply_median_blur, preprocess_pipeline,
    preprocess_ladder
)
from src.visualize import show_and_save_fig

config = load_config('config/config.yaml')
hdata = load_homography('config/homography.npz')
H, output_size = hdata['H'], hdata['output_size']
bev_slots = load_slots_json('config/slots.json')
print(f"Loaded {len(bev_slots)} BEV slots")

# %% [markdown]
# ## 2. Why Preprocessing?
#
# ### The Problem
# Raw camera images contain noise, uneven illumination, and compression
# artifacts that confuse both thresholding and edge detection.
#
# ### The Solution — A 4-Step Ladder
#
# | Step | Function | Purpose | OpenCV Call |
# |------|----------|---------|-------------|
# | 1 | **Grayscale** | Reduce 3 channels → 1 | `cv2.cvtColor(img, COLOR_BGR2GRAY)` |
# | 2 | **CLAHE** | Fix uneven illumination | `cv2.createCLAHE(clipLimit, tileGridSize)` |
# | 3 | **Gaussian Blur** | Suppress high-freq noise | `cv2.GaussianBlur(img, ksize, sigma)` |
# | 4 | **Median Filter** | Remove salt-and-pepper noise | `cv2.medianBlur(img, ksize)` |
#
# ### Why this order?
# - **Grayscale first**: Simplifies all downstream operations to one channel
# - **CLAHE before blur**: We want to enhance contrast while detail is still present
# - **Blur after CLAHE**: Noise amplified by CLAHE is then smoothed
# - **Median last**: It's edge-preserving — perfect final cleanup before Canny

# %%
# --- Load sample frames ---
DATA_ROOT = 'data/raw/PKLot'
LOT_PATH = os.path.join(DATA_ROOT, 'parking2')

frames = list_frames(LOT_PATH)
sunny_frames = [f for f in frames if f['weather'] == 'sunny']
cloudy_frames = [f for f in frames if f['weather'] == 'cloudy']

# Pick one sunny, one cloudy frame
sample_sunny = sunny_frames[len(sunny_frames)//2]
sample_cloudy = cloudy_frames[len(cloudy_frames)//2] if cloudy_frames else frames[0]

# Load and warp to BEV
img_sunny = cv2.imread(sample_sunny['image_path'])
bev_sunny = warp_perspective(img_sunny, H, output_size)

img_cloudy = cv2.imread(sample_cloudy['image_path'])
bev_cloudy = warp_perspective(img_cloudy, H, output_size)

print(f"Sunny frame: {sample_sunny['image_path']}")
print(f"Cloudy frame: {sample_cloudy['image_path']}")

# %% [markdown]
# ## 3. Select Demo Slots
#
# We pick one **occupied** and one **vacant** slot from the sunny frame to
# demonstrate the full preprocessing ladder side-by-side.

# %%
# Parse ground truth to find occupied and vacant slots
sunny_slots = parse_pklot_xml(sample_sunny['xml_path'])

# Find one occupied and one vacant slot
occ_slot = None
vac_slot = None
for s in sunny_slots:
    bev_pts = transform_points(s['points'], H)
    sid = s['id']
    if s['occupied'] and occ_slot is None and sid in bev_slots:
        occ_slot = {'id': sid, 'bev_pts': bev_pts, 'label': 'OCCUPIED'}
    elif not s['occupied'] and vac_slot is None and sid in bev_slots:
        vac_slot = {'id': sid, 'bev_pts': bev_pts, 'label': 'VACANT'}
    if occ_slot and vac_slot:
        break

print(f"Demo occupied slot: {occ_slot['id']}")
print(f"Demo vacant slot:   {vac_slot['id']}")

# %% [markdown]
# ## 4. The Preprocessing Ladder — Side by Side
#
# ### OpenCV Functions Used
#
# **`cv2.cvtColor(src, code)`**
# - Converts colour spaces. `COLOR_BGR2GRAY` uses the luminosity formula:
#   $Y = 0.299R + 0.587G + 0.114B$
#
# **`cv2.createCLAHE(clipLimit, tileGridSize)`**
# - **CLAHE** = Contrast Limited Adaptive Histogram Equalization
# - Splits image into tiles, equalizes each independently
# - `clipLimit` caps amplification to prevent noise blowup
# - Much better than global HE for non-uniform lighting
#
# **`cv2.GaussianBlur(src, ksize, sigmaX)`**
# - Convolves with 2D Gaussian kernel: $G(x,y) = \frac{1}{2\pi\sigma^2} e^{-(x^2+y^2)/(2\sigma^2)}$
# - `ksize=(5,5)` with `sigma=0` → sigma auto-calculated from kernel size
# - Removes high-frequency noise but slightly blurs edges
#
# **`cv2.medianBlur(src, ksize)`**
# - Replaces each pixel with the median of its neighbourhood
# - Excellent at removing salt-and-pepper noise
# - **Edge-preserving**: doesn't blur edges like Gaussian does

# %%
# --- Build the preprocessing ladder for both slots ---
fig, axes = plt.subplots(2, 5, figsize=(11, 9))

for row, slot_info in enumerate([occ_slot, vac_slot]):
    # Extract slot patch
    polygon = bev_slots[slot_info['id']]
    patch, bbox, mask = extract_slot_image(bev_sunny, polygon)

    # Run ladder
    stages = preprocess_ladder(patch)

    for col, (name, img_stage) in enumerate(stages[:5]):
        ax = axes[row, col]
        if len(img_stage.shape) == 3:
            ax.imshow(cv2.cvtColor(img_stage, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img_stage, cmap='gray')
        
        prefix = slot_info['label']
        ax.set_title(f"{prefix}\n{name}", fontsize=9, fontweight='bold')
        ax.axis('off')

show_and_save_fig(fig,
                 'Preprocessing Ladder: Occupied vs Vacant Slot',
                 '08_preprocessing_ladder.png')

# %% [markdown]
# ## 5. Histogram Before vs After CLAHE
#
# ### Why CLAHE matters
# In shadows or overcast conditions, the intensity range of a slot
# may be compressed to just 40–80 (out of 0–255). This makes
# thresholding nearly impossible.
#
# CLAHE stretches the local contrast so that even subtle differences
# between car paint and asphalt become visible.

# %%
# --- Histogram comparison: before and after CLAHE ---
fig, axes = plt.subplots(2, 3, figsize=(13, 8),
                         gridspec_kw={'width_ratios': [1, 1, 4]})

for row, slot_info in enumerate([occ_slot, vac_slot]):
    polygon = bev_slots[slot_info['id']]
    patch, bbox, mask = extract_slot_image(bev_sunny, polygon)
    
    gray = to_grayscale(patch)
    clahe_img = apply_clahe(gray, clip_limit=2.0, grid_size=(4, 4))
    
    # Before CLAHE
    ax = axes[row, 0]
    ax.imshow(gray, cmap='gray')
    ax.set_title(f"{slot_info['label']} — Grayscale", fontweight='bold')
    ax.axis('off')
    
    # After CLAHE
    ax = axes[row, 1]
    ax.imshow(clahe_img, cmap='gray')
    ax.set_title(f"{slot_info['label']} — After CLAHE", fontweight='bold')
    ax.axis('off')
    
    # Histograms overlaid — only pixels INSIDE the slot polygon.
    # The crop is a bounding box around a rotated slot, so the corners are
    # zero-padding; counting them puts a huge spike at intensity 0 that
    # flattens the real distribution.
    ax = axes[row, 2]
    inside = mask > 0
    ax.hist(gray[inside], 256, [0, 256], alpha=0.5, color='blue', label='Before CLAHE')
    ax.hist(clahe_img[inside], 256, [0, 256], alpha=0.5, color='orange', label='After CLAHE')
    ax.set_title(f"{slot_info['label']} — Histograms (slot pixels only)",
                 fontweight='bold')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Pixel Count')
    ax.legend(fontsize=8)

show_and_save_fig(fig, 'CLAHE Effect on Intensity Distribution',
                 '08_clahe_histograms.png')

# %% [markdown]
# ## 6. Edge Preservation: Gaussian vs Median Filter
#
# ### The Trade-off
# - **Gaussian blur** is fast and reduces noise well, but it blurs edges
# - **Median filter** preserves edges while removing impulse noise
# - For Canny edge detection downstream, **edge sharpness is critical**
# - We use Gaussian for general smoothing, then median for final cleanup

# %%
# --- Gaussian vs Median on an occupied slot ---
polygon = bev_slots[occ_slot['id']]
patch, bbox, mask = extract_slot_image(bev_sunny, polygon)
gray = to_grayscale(patch)
clahe_img = apply_clahe(gray)

gauss_3 = apply_gaussian_blur(clahe_img, kernel_size=(3, 3))
gauss_5 = apply_gaussian_blur(clahe_img, kernel_size=(5, 5))
median_3 = apply_median_blur(clahe_img, kernel_size=3)
median_5 = apply_median_blur(clahe_img, kernel_size=5)

fig, axes = plt.subplots(2, 3, figsize=(7, 9))

# Row 1: Gaussian
axes[0, 0].imshow(clahe_img, cmap='gray')
axes[0, 0].set_title('CLAHE (Input)', fontweight='bold')
axes[0, 1].imshow(gauss_3, cmap='gray')
axes[0, 1].set_title('Gaussian 3×3', fontweight='bold')
axes[0, 2].imshow(gauss_5, cmap='gray')
axes[0, 2].set_title('Gaussian 5×5', fontweight='bold')

# Row 2: Median
axes[1, 0].imshow(clahe_img, cmap='gray')
axes[1, 0].set_title('CLAHE (Input)', fontweight='bold')
axes[1, 1].imshow(median_3, cmap='gray')
axes[1, 1].set_title('Median 3×3', fontweight='bold')
axes[1, 2].imshow(median_5, cmap='gray')
axes[1, 2].set_title('Median 5×5', fontweight='bold')

for ax in axes.flat:
    ax.axis('off')

show_and_save_fig(fig,
                 'Gaussian Blur vs Median Filter — Edge Preservation Comparison',
                 '08_filter_comparison.png')

# %% [markdown]
# ## 7. Full Pipeline on Multiple Slots
#
# Let's run the complete `preprocess_pipeline()` on several slots
# from different weather conditions.

# %%
# --- Full pipeline on sunny vs cloudy ---
fig, axes = plt.subplots(4, 2, figsize=(5.5, 16))

slot_ids_demo = list(bev_slots.keys())[:4]

for row, sid in enumerate(slot_ids_demo):
    polygon = bev_slots[sid]
    
    # Sunny
    patch_s, _, _ = extract_slot_image(bev_sunny, polygon)
    preprocessed_s = preprocess_pipeline(patch_s)
    
    axes[row, 0].imshow(preprocessed_s, cmap='gray')
    axes[row, 0].set_title(f'Slot {sid} — Sunny', fontsize=9, fontweight='bold')
    axes[row, 0].axis('off')
    
    # Cloudy
    patch_c, _, _ = extract_slot_image(bev_cloudy, polygon)
    preprocessed_c = preprocess_pipeline(patch_c)
    
    axes[row, 1].imshow(preprocessed_c, cmap='gray')
    axes[row, 1].set_title(f'Slot {sid} — Cloudy', fontsize=9, fontweight='bold')
    axes[row, 1].axis('off')

show_and_save_fig(fig, 'Preprocessed Slots: Sunny vs Cloudy',
                 '08_sunny_vs_cloudy.png')

# %% [markdown]
# ## Phase Summary
#
# ### What we accomplished
# 1. ✅ Built and visualized the 4-step preprocessing ladder
# 2. ✅ Demonstrated CLAHE's contrast enhancement effect via histograms
# 3. ✅ Compared Gaussian blur vs median filter for edge preservation
# 4. ✅ Ran the full pipeline across weather conditions
#
# ### Key insights
# - **CLAHE** dramatically improves contrast in shadowed/overcast slots
# - **Median filter** preserves edges better than Gaussian — crucial for Canny
# - The preprocessing pipeline is deterministic and weather-agnostic
#
# ### Next: Notebook 05 — Segmentation & Morphology
