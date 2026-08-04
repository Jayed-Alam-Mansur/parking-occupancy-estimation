#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Notebook 09: Final Report
=========================
Convert to notebook with: jupytext --to notebook notebooks/09_final_report.py

Covers:
- Phase 14-15: Executive summary, method recap, results gallery,
  performance summary, and discussion
"""

# %% [markdown]
# # Notebook 09: Final Report — Automatic Parking Occupancy Estimation
#
# **Course:** Digital Image Processing
# **Project:** Classical Image Processing for Parking Lot Occupancy
#
# ---
#
# ## Executive Summary
#
# This project implements a **fully classical** parking occupancy estimation
# system using only OpenCV, NumPy, and basic image processing techniques.
# **No machine learning, deep learning, or pretrained models** are used.
#
# The system processes parking lot surveillance footage through a multi-stage
# pipeline: perspective correction → slot extraction → preprocessing →
# segmentation → feature extraction → rule-based classification →
# occupancy statistics.

# %% [markdown]
# ## 1. System Architecture
#
# ```
# Camera Frame
#     │
#     ▼
# ┌─────────────────┐
# │ BEV Warp (H)    │  cv2.warpPerspective()
# └────────┬────────┘
#          ▼
# ┌─────────────────┐
# │ Slot Extraction  │  cv2.perspectiveTransform(), polygon masks
# └────────┬────────┘
#          ▼
# ┌─────────────────┐
# │ Preprocessing    │  cv2.cvtColor(), CLAHE, GaussianBlur, medianBlur
# └────────┬────────┘
#          ▼
# ┌─────────────────┐
# │ Segmentation     │  cv2.threshold() (Otsu), adaptiveThreshold(), fusion
# └────────┬────────┘
#          ▼
# ┌─────────────────┐
# │ Morphology       │  cv2.morphologyEx() (opening, closing)
# └────────┬────────┘
#          ▼
# ┌─────────────────┐
# │ 8 Features       │  Canny, Sobel, connectedComponents, calcHist
# └────────┬────────┘
#          ▼
# ┌─────────────────┐
# │ Decision Cascade │  Fast-path + weighted score + threshold
# └────────┬────────┘
#          ▼
#    OCCUPIED / VACANT
# ```

# %% [markdown]
# ## 2. Setup

# %%
import sys, os

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
import pandas as pd

from src.utils import load_config
from src.io_utils import list_frames, parse_pklot_xml
from src.geometry import load_homography, warp_perspective
from src.roi import load_slots_json, extract_slot_image, create_eroded_core_mask
from src.preprocessing import preprocess_pipeline
from src.segmentation import otsu_threshold, adaptive_threshold, fuse_channels
from src.morphology import clean_binary_mask
from src.features import extract_all_features
from src.decide import classify_slot, load_thresholds
from src.evaluate import compute_metrics
from src.stats import compute_statistics, format_report
from src.visualize import annotate_parking_image, create_legend
from src.visualize import show_and_save_fig

config = load_config('config/config.yaml')
hdata = load_homography('config/homography.npz')
H, output_size = hdata['H'], hdata['output_size']
bev_slots = load_slots_json('config/slots.json')
thresholds, weights = load_thresholds('config/thresholds.yaml')

feature_names = ['edge_density', 'foreground_ratio', 'gradient_magnitude',
                 'local_variance', 'largest_component', 'intensity_std',
                 'otsu_separability', 'mean_saturation']

# %% [markdown]
# ## 3. OpenCV Functions Used — Complete Reference
#
# | Stage | Function | Purpose |
# |-------|----------|---------|
# | **Geometry** | `cv2.getPerspectiveTransform(src, dst)` | Compute 3×3 homography H from 4 point pairs |
# | | `cv2.warpPerspective(img, H, dsize)` | Warp image to bird's-eye view |
# | | `cv2.perspectiveTransform(pts, H)` | Transform point coordinates through H |
# | **Preprocessing** | `cv2.cvtColor(img, COLOR_BGR2GRAY)` | Convert to grayscale |
# | | `cv2.createCLAHE(clipLimit, tileGridSize)` | Contrast Limited Adaptive Histogram Equalization |
# | | `cv2.GaussianBlur(img, ksize, sigma)` | Gaussian smoothing for noise reduction |
# | | `cv2.medianBlur(img, ksize)` | Median filter — edge-preserving denoising |
# | **Segmentation** | `cv2.threshold(img, T, 255, THRESH_BINARY+THRESH_OTSU)` | Otsu's automatic thresholding |
# | | `cv2.adaptiveThreshold(img, 255, ADAPTIVE_GAUSSIAN, ...)` | Locally adaptive threshold |
# | | `cv2.cvtColor(img, COLOR_BGR2HSV)` | HSV conversion for shadow detection |
# | **Morphology** | `cv2.getStructuringElement(MORPH_RECT, ksize)` | Create structuring element |
# | | `cv2.morphologyEx(img, MORPH_OPEN, kernel)` | Opening: removes small noise |
# | | `cv2.morphologyEx(img, MORPH_CLOSE, kernel)` | Closing: fills small holes |
# | | `cv2.erode(img, kernel)` / `cv2.dilate(img, kernel)` | Basic morphological ops |
# | **Features** | `cv2.Canny(img, low, high)` | Edge detection with hysteresis |
# | | `cv2.Sobel(img, ddepth, dx, dy)` | Gradient computation |
# | | `cv2.connectedComponentsWithStats(img)` | Connected component analysis |
# | | `cv2.calcHist([img], [0], None, [256], [0,256])` | Histogram computation |
# | | `cv2.contourArea(pts)` | Polygon area for slot size analysis |
# | **Visualisation** | `cv2.fillPoly(img, [pts], color)` | Draw filled polygons |
# | | `cv2.addWeighted(overlay, α, base, 1-α, 0)` | Semi-transparent overlay blending |
# | | `cv2.polylines(img, [pts], True, color, thickness)` | Draw polygon outlines |
# | | `cv2.putText(img, text, pos, font, scale, color)` | Text annotation |

# %% [markdown]
# ## 4. Results Gallery — Best and Worst Frames

# %%
DATA_ROOT = 'data/raw/PKLot'
LOT_PATH = os.path.join(DATA_ROOT, 'parking2')
all_frames = list_frames(LOT_PATH)

def run_pipeline_on_frame(frame_info):
    """Run the full pipeline on one frame, return labels, stats, metrics."""
    img = cv2.imread(frame_info['image_path'])
    bev = warp_perspective(img, H, output_size)
    gt = parse_pklot_xml(frame_info['xml_path'])
    gt_lookup = {s['id']: s['occupied'] for s in gt}

    labels = {}
    scores = {}
    for sid, polygon in bev_slots.items():
        patch, bbox, mask = extract_slot_image(bev, polygon)
        if patch.size == 0:
            continue
        core_mask = create_eroded_core_mask(mask, erosion_px=3)
        preprocessed = preprocess_pipeline(patch)
        otsu_bin, _, _ = otsu_threshold(preprocessed)
        adapt_bin = adaptive_threshold(preprocessed)
        fused = fuse_channels(otsu_bin, adapt_bin)
        cleaned = clean_binary_mask(fused)
        features = extract_all_features(preprocessed, cleaned, patch, core_mask)
        label, conf, score = classify_slot(features, thresholds, weights)
        labels[sid] = label
        scores[sid] = score

    y_true = [gt_lookup[sid] for sid in labels if sid in gt_lookup]
    y_pred = [labels[sid] for sid in labels if sid in gt_lookup]
    metrics = compute_metrics(y_true, y_pred)
    stats = compute_statistics(labels)

    annotated = annotate_parking_image(bev, bev_slots, labels, scores=scores)
    annotated = create_legend(annotated, stats)

    return annotated, stats, metrics

# %%
# --- Run on one frame per weather and display ---
fig, axes = plt.subplots(1, 3, figsize=(24, 12))

for idx, weather in enumerate(['sunny', 'cloudy', 'rainy']):
    w_frames = [f for f in all_frames if f['weather'] == weather]
    if not w_frames:
        continue

    frame_info = w_frames[len(w_frames)//2]
    annotated, stats, metrics = run_pipeline_on_frame(frame_info)

    axes[idx].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(
        f"{weather.upper()}\n"
        f"Acc={metrics['accuracy']:.1%} | F1={metrics['f1_score']:.1%}\n"
        f"Occ={stats['occupied']}/{stats['total_spaces']} ({stats['occupancy_pct']}%)",
        fontsize=12, fontweight='bold'
    )
    axes[idx].axis('off')

show_and_save_fig(fig, 'Pipeline Results Across Weather Conditions',
                 '19_results_gallery.png')

# %% [markdown]
# ## 5. The 8-Feature Vector — Design Rationale
#
# ### Feature selection
# Each feature captures a **different physical signal** that distinguishes
# a car from empty asphalt:
#
# | Feature | Physical Signal | Why Classical? |
# |---------|----------------|---------------|
# | Edge Density | Cars have structured contours | Canny + pixel counting |
# | Foreground Ratio | Cars produce more "foreground" after thresholding | Binary mask ratio |
# | Gradient Magnitude | Car surfaces have strong intensity gradients | Sobel operator |
# | Local Variance | Cars have heterogeneous texture | Statistical moment |
# | Largest Component | Cars form one coherent blob | Connected components |
# | Intensity Std | Cars have diverse grey levels | Standard deviation |
# | Otsu Separability | Cars create bimodal histograms | Between-class variance |
# | Mean Saturation | Cars have coloured paint | HSV colour space |
#
# ### Decision cascade
# Our classifier uses a **transparent, interpretable rule**:
# 1. **Fast path**: extreme edge density → instant decision
# 2. **Weighted score**: $S = \sum w_k f_k / \sum w_k$ where weights come from Fisher ratios
# 3. **Threshold**: $S > \tau$ → OCCUPIED
#
# This is parameter selection (like choosing a filter kernel size),
# **not model training**.

# %% [markdown]
# ## 6. Performance Summary

# %%
# Load feature vectors for summary
fv_path = 'data/ground_truth/feature_vectors.csv'
if os.path.exists(fv_path):
    df = pd.read_csv(fv_path)
    print(f"Feature vector dataset: {len(df)} samples")
    print(f"  Occupied: {(df['occupied']==1).sum()}")
    print(f"  Vacant:   {(df['occupied']==0).sum()}")
    print(f"\nWeather distribution:")
    print(df['weather'].value_counts().to_string())

# %% [markdown]
# ## 7. Tuned Configuration

# %%
print("FINAL TUNED CONFIGURATION")
print("-" * 40)

print("\n  Decision Thresholds:")
for k, v in thresholds.items():
    print(f"    {k}: {v:.4f}")

print("\n  Feature Weights (Fisher-derived):")
sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
for k, v in sorted_w:
    bar = '█' * int(v * 100)
    print(f"    {k:<25s} {v:.4f}  {bar}")

# %% [markdown]
# ## 8. Discussion
#
# ### Strengths
# - **Fully interpretable**: Every decision can be traced to specific pixel statistics
# - **No training data required**: Thresholds set by inspecting histograms
# - **Real-time capable**: Processes frames at practical FPS
# - **Weather-robust features**: Fisher-ranked features work across conditions
#
# ### Limitations
# - **Shadow sensitivity**: Strong shadows in sunny conditions cause false positives
# - **Fixed viewpoint**: Homography is camera-specific; new cameras need recalibration
# - **No temporal context**: Each frame is processed independently (no tracking)
# - **Threshold sensitivity**: The decision boundary depends on careful tuning
#
# ### Potential Improvements (Future Work)
# 1. **Better shadow suppression**: Use chromaticity-based shadow detection
# 2. **Temporal hysteresis**: Require consecutive frames to agree before flipping
# 3. **Reference image subtraction**: Use empty-lot reference for change detection
# 4. **Adaptive thresholds**: Adjust thresholds based on time-of-day or illumination
# 5. **Multi-camera fusion**: Cross-validate detections from multiple viewpoints

# %% [markdown]
# ## 9. Conclusion
#
# This project demonstrates that **classical image processing techniques**
# can achieve practical parking occupancy estimation without any machine
# learning. The main findings are:
#
# 1. **Perspective correction has the largest single effect** — it normalises
#    slot sizes so one threshold set works everywhere
# 2. **Multi-feature fusion outperforms single features** — combining
#    8 complementary features with Fisher-derived weights gives robust
#    decisions
# 3. **The cascade architecture is efficient** — fast-path decisions
#    handle obvious cases, leaving only ambiguous slots for full scoring
#
# The system processes the PKLot parking2 lot (100 slots) and produces
# real-time occupancy statistics with colour-coded BEV annotations.

# %%
print("\n" + "-" * 40)
print("  Automatic Parking Occupancy Estimation")
print("  Using Classical Image Processing")
print("-" * 40)
