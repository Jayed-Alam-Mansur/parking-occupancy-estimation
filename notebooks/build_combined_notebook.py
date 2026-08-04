"""Assemble the nine project notebooks into a single document.

Preserves every executed output. Removes repeated boilerplate, renumbers
sections into one continuous sequence, and inserts a short introduction to each
section.
"""
import base64
import io
import re
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
from PIL import Image

SRC = "notebooks"
OUT = "notebooks/00_COMPLETE_PROJECT.ipynb"
JPEG_ABOVE = 350_000   # recompress embedded PNGs larger than this
JPEG_Q = 90

# ---------------------------------------------------------------- front matter

INTRO = r"""# Automatic Parking Occupancy Estimation using Classical Image Processing

Digital Image Processing course project.
Author: Jayed Alam Mansur

## Objective

Given a single surveillance frame of a parking lot, estimate:

1. the total number of parking spaces, `N`
2. a per-space `OCCUPIED` / `VACANT` label
3. the overall occupancy rate, `O% = N_occupied / N * 100`

The system uses one fixed camera and no per-space hardware sensors.

## Scope and constraints

The implementation uses only classical image processing: homography, CLAHE,
Otsu and adaptive thresholding, morphological operations, Canny and Sobel
derived features, and a rule-based decision cascade. No machine learning is
used. The following are excluded by the project specification and are listed as
prohibited in `requirements.txt`:

- tensorflow
- torch / pytorch
- keras
- ultralytics (YOLO)
- detectron2
- any pretrained model package

Every threshold and kernel size is set explicitly, and Section 7 reports the
measurements used to choose them.

## Dataset

PKLot, collected at the Federal University of Parana. Approximately 12,400
images from three parking lots under sunny, cloudy and rainy conditions, with
per-space polygon annotations and occupancy labels. The annotations are used as
ground truth for the evaluation in Section 8.

## Summary of results

Measured on 11,599 held-out space samples:

| Metric | Value |
|---|---|
| Accuracy | 74.30 % |
| F1 score | 0.7310 |
| Processing time | 77.33 ms per frame (12.9 FPS), CPU only |

Section 8 also reports a comparison in which a single-feature `edge_density`
baseline outperforms the full eight-feature cascade on the large evaluation
set. That result and its likely causes are discussed in Sections 8 and 9.

## Contents

| Section | Topic |
|---|---|
| [1](#sec1) | Dataset and exploratory analysis |
| [2](#sec2) | Perspective transformation |
| [3](#sec3) | Region of interest extraction |
| [4](#sec4) | Preprocessing |
| [5](#sec5) | Segmentation and morphology |
| [6](#sec6) | Feature extraction |
| [7](#sec7) | Threshold calibration and classification |
| [8](#sec8) | Evaluation |
| [9](#sec9) | Discussion and conclusion |

This notebook is a compiled document: the nine development notebooks
(`01_explore` through `09_final_report`) merged into one continuous sequence
with all original executed outputs preserved. The repeated import blocks were
collapsed into the single setup cell below. Large photographic outputs were
recompressed to JPEG to reduce file size.
"""

SETUP = '''# Setup: all imports used in this notebook
import os, sys, time
from pathlib import Path

# Run from the project root so relative paths resolve
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, os.path.abspath('.'))

# Third-party: image processing and plotting only
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
%matplotlib inline

# Project modules, one per pipeline stage (see src/)
from src.utils        import load_config, display_images, print_separator
from src.io_utils     import (list_frames, parse_pklot_xml, quality_gate,
                              export_ground_truth_csv, curate_samples,
                              get_slot_geometry_from_xml)
from src.geometry     import (compute_homography, warp_perspective, transform_points,
                              save_homography, load_homography, validate_bev)
from src.roi          import (extract_slot_image, create_eroded_core_mask,
                              draw_slots_on_image, save_slots_json, load_slots_json)
from src.preprocessing import (to_grayscale, equalize_histogram, apply_clahe,
                              apply_gaussian_blur, apply_median_blur,
                              preprocess_pipeline, preprocess_ladder)
from src.segmentation import (global_threshold, adaptive_threshold, otsu_threshold,
                              shadow_suppress_hsv, fuse_channels)
from src.morphology   import (apply_erosion, apply_dilation, apply_opening,
                              apply_closing, clean_binary_mask, morphology_grid)
from src.features     import (extract_all_features, compute_fisher_ratio,
                              compute_edge_density, compute_gradient_magnitude)
from src.decide       import (weighted_score, classify_slot, classify_all_slots,
                              load_thresholds, save_thresholds,
                              DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
from src.evaluate     import (compute_metrics, plot_confusion_matrix,
                              format_metrics_report, Timer)
from src.stats        import compute_statistics, format_report
from src.visualize    import annotate_parking_image, create_legend, show_and_save_fig

np.random.seed(42)

# The eight per-space features used in Section 6
feature_names = ['edge_density', 'foreground_ratio', 'gradient_magnitude',
                 'local_variance', 'largest_component', 'intensity_std',
                 'otsu_separability', 'mean_saturation']

DATA_ROOT = 'data/raw/PKLot'
config = load_config('config/config.yaml')
'''

SECTIONS = [
    ("01_explore.ipynb", 1, r"""## Section 1. Dataset and Exploratory Analysis

This section establishes what the input data looks like before any processing is
applied. It covers the dataset structure, image properties, the XML annotation
format, and a quality gate for rejecting unusable frames.

The PKLot dataset contains approximately 12,400 images from three parking lots
under three weather conditions, each with per-space polygon annotations and an
occupancy label. The annotations are used as ground truth in Section 8.

One observation from the sample frames affects several later design decisions:
on sunny frames, cast shadows are dark and roughly car-shaped, which makes them
difficult to separate from vehicles using intensity alone. This is addressed in
Sections 4 and 5.
"""),
    ("02_geometry.ipynb", 2, r"""## Section 2. Perspective Transformation

In the raw camera view, spaces near the camera occupy a larger image area than
spaces further away, although they are the same physical size. Any feature that
depends on area or pixel count is therefore influenced by a space's distance
from the camera rather than by its occupancy state.

The parking surface is planar, and two views of a plane are related by a
homography, a single 3x3 matrix:

$$s\begin{bmatrix}x'\\y'\\1\end{bmatrix} = \mathbf{H}\begin{bmatrix}x\\y\\1\end{bmatrix}$$

Four point correspondences are sufficient to solve for `H`. Applying it produces
a bird's-eye view in which all spaces have approximately equal size and shape,
so that subsequent measurements are comparable across the lot.
"""),
    ("03_roi.ipynb", 3, r"""## Section 3. Region of Interest Extraction

The problem is decomposed into 100 independent per-space measurements. The
annotation polygons are transformed by the same homography `H` so that they align
with the bird's-eye view, and each space is then cropped and masked.

Adjacent spaces share painted boundary lines, so their polygons overlap
slightly. A vehicle in one space can therefore contribute pixels to a
neighbouring space's region and cause a false positive. To avoid this, each mask
is eroded inward and only the interior of each space is measured. This reduces
the available signal but makes the per-space measurements independent.
"""),
    ("04_preprocessing.ipynb", 4, r"""## Section 4. Preprocessing

The same empty space photographed under direct sunlight and under cloud cover
produces substantially different pixel values, so a fixed threshold tuned on one
condition fails on the other. A single sunny frame also contains both sunlit and
shaded regions, so one global correction is not sufficient.

Each space is processed through four steps:

| Step | Operation | Purpose |
|---|---|---|
| 1 | Grayscale conversion | Reduce three channels to one; colour is not required for occupancy |
| 2 | CLAHE | Equalise contrast per tile, so sunlit and shaded regions are corrected independently |
| 3 | Gaussian blur | Suppress high-frequency sensor noise that would otherwise produce spurious edges |
| 4 | Median filter | Remove impulse noise from JPEG compression |

The median filter is applied after the Gaussian blur because impulse noise is a
rank-order problem rather than a convolution problem: averaging cannot remove a
single outlying pixel, but a rank filter can.
"""),
    ("05_segmentation.ipynb", 5, r"""## Section 5. Segmentation and Morphology

This section separates vehicle pixels from road surface pixels. Three
thresholding methods are compared:

- Global thresholding, which uses one cutoff for the whole image and fails when
  illumination varies across the frame.
- Otsu's method, which selects the cutoff automatically by maximising
  between-class variance, but assumes a bimodal intensity histogram.
- Adaptive thresholding, which computes a local cutoff per neighbourhood and
  handles illumination gradients, but can report texture in an empty space.

The Otsu and adaptive results are fused rather than selecting a single method.

Shadows remain a difficulty for all intensity-based methods, since a hard shadow
is dark and car-shaped. Converting to HSV provides an alternative: a shadowed
region loses value while retaining hue, so shadowed road surface can be
distinguished from a vehicle by colour rather than by brightness.

Morphological operations then clean the binary mask. Opening removes isolated
noise pixels and closing fills interior gaps in the vehicle region.
"""),
    ("06_features.ipynb", 6, r"""## Section 6. Feature Extraction

Each space region contains several thousand pixels. This section reduces it to
eight scalar features, each with a physical interpretation:

| # | Feature | Interpretation |
|---|---|---|
| 1 | `edge_density` | A vehicle introduces strong edges; empty road surface is smooth |
| 2 | `foreground_ratio` | Proportion of the space marked as foreground by the binary mask |
| 3 | `gradient_magnitude` | Mean Sobel response, a measure of texture strength |
| 4 | `local_variance` | Road surface is uniform; vehicle bodywork is not |
| 5 | `largest_component` | A vehicle forms one large connected region; noise forms many small ones |
| 6 | `intensity_std` | Vehicles introduce both highlights and shadow |
| 7 | `otsu_separability` | How cleanly the intensity histogram separates into two classes |
| 8 | `mean_saturation` | Painted vehicles are more saturated than grey road surface |

The discriminative power of each feature is then quantified using the Fisher
discriminant ratio, which measures the separation between the occupied and
vacant distributions relative to their spread:

$$F = \frac{(\mu_{occ} - \mu_{vac})^2}{\sigma_{occ}^2 + \sigma_{vac}^2}$$

A higher value indicates better class separation. The resulting ranking is used
to set the feature weights in Section 7.
"""),
    ("07_threshold_tuning.ipynb", 7, r"""## Section 7. Threshold Calibration and Classification

The eight features must be reduced to a binary label. A trained classifier is
not used, since the project requires that each decision be traceable to a
specific measurement. A two-stage cascade is used instead:

1. Fast path. If `edge_density` is below a lower bound the space is labelled
   vacant, and if it is above an upper bound the space is labelled occupied.
   Approximately one third of spaces are resolved at this stage.
2. Weighted score. Remaining spaces are scored using the Fisher ratios from
   Section 6 as weights:

$$S = \frac{\sum_k w_k f_k}{\sum_k w_k} \qquad \text{occupied if } S > \tau$$

The decision threshold $\tau$ is selected by sweeping it across its range and
recording accuracy and F1 at each value, rather than by inspection. The
resulting curve has a broad maximum, which indicates the configuration is not
sensitive to small changes in $\tau$.
"""),
    ("08_evaluation.ipynb", 8, r"""## Section 8. Evaluation

The pipeline is evaluated on 120 frames sampled evenly across all three weather
conditions, giving 11,599 individual space classifications, each compared
against the ground-truth annotation. Overall and per-weather confusion matrices
are reported, along with a ranking of the worst-performing spaces and a
stage-by-stage timing profile.

Measured performance: 74.30 % accuracy, 0.7310 F1, and 77.33 ms per frame
(12.9 FPS) on CPU.

Section 8.5 compares the full eight-feature cascade against a baseline that
thresholds `edge_density` alone, with no weighting or scoring stage. The
single-feature baseline achieves higher accuracy than the cascade on this
evaluation set. Three factors are likely to contribute:

- The Fisher weights were computed on a small, comparatively clean sample and
  then applied to a larger and more variable set, so the weighting does not
  transfer.
- The seven weaker features are mutually correlated rather than independent, so
  averaging them does not cancel error and instead reduces the contribution of
  the one feature that separates the classes well.
- Each additional feature introduces its own failure cases. `mean_saturation`
  performs poorly on grey vehicles and `local_variance` performs poorly on wet
  road surface. The combined score inherits all of them.

The result indicates that increasing the number of features does not necessarily
increase discriminative power, and that the weighting procedure requires
revision. This is discussed further in Section 9.
"""),
    ("09_final_report.ipynb", 9, r"""## Section 9. Discussion and Conclusion

The pipeline runs end to end using one camera, no per-space sensors, no training
data and no GPU, at a rate suitable for real-time use on a laptop CPU. Each
decision can be traced to a named pixel statistic, so any misclassification can
be attributed to a specific feature.

Limitations. Overall accuracy of 74.30 % is a working demonstration rather than
a deployable system. Rainy frames are the weakest case, since wet road surface
produces reflections that register as edges. The homography is calibrated
manually per camera, so a new installation requires new point correspondences.
As shown in Section 8, the eight-feature cascade is outperformed by a
single-feature baseline on the full evaluation set.

Proposed further work, in order of expected benefit:

1. Adopt the single-feature baseline, which is both more accurate and faster on
   the evaluation set.
2. Recompute the feature weights on the full 11,599-sample set rather than the
   smaller sample, to determine whether the cascade is mis-weighted or
   unsuitable in principle.
3. Remove the features that measurably reduce accuracy instead of retaining all
   eight.
4. Add temporal smoothing. Occupancy changes slowly relative to the frame rate,
   so a running vote across consecutive frames would suppress isolated
   single-frame errors at low computational cost.
"""),
]

# ---------------------------------------------------------------- helpers

# nb04 rendered to disk (Agg backend) instead of inline; re-attach those figures
NB04_FIGS = {
    9:  "outputs/screenshots/08_preprocessing_ladder.png",
    11: "outputs/screenshots/08_clahe_histograms.png",
    13: "outputs/screenshots/08_filter_comparison.png",
    15: "outputs/screenshots/08_sunny_vs_cloudy.png",
}

stats = {"png_before": 0, "jpeg_after": 0, "converted": 0, "reattached": 0}


def to_jpeg_b64(raw: bytes) -> str:
    im = Image.open(io.BytesIO(raw))
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "white")
        im = im.convert("RGBA")
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_Q, optimize=True, progressive=True)
    return base64.b64encode(buf.getvalue()).decode()


def shrink_outputs(cell):
    """Recompress large photographic PNG outputs to JPEG in place."""
    for o in cell.get("outputs", []):
        d = o.get("data")
        if not d or "image/png" not in d:
            continue
        b64 = d["image/png"]
        stats["png_before"] += len(b64)
        if len(b64) < JPEG_ABOVE:
            stats["jpeg_after"] += len(b64)
            continue
        try:
            new = to_jpeg_b64(base64.b64decode(b64))
        except Exception as e:                       # pragma: no cover
            print(f"    recompress failed: {e}", file=sys.stderr)
            stats["jpeg_after"] += len(b64)
            continue
        del d["image/png"]
        d["image/jpeg"] = new
        stats["jpeg_after"] += len(new)
        stats["converted"] += 1


def attach_figure(cell, path):
    with open(path, "rb") as fh:
        raw = fh.read()
    cell.setdefault("outputs", []).append(
        nbformat.v4.new_output(
            "display_data",
            data={"image/jpeg": to_jpeg_b64(raw)},
            metadata={"note": f"rendered by this cell to {path} (Agg backend)"},
        )
    )
    stats["reattached"] += 1


def is_boilerplate_code(src: str) -> bool:
    body = [l for l in src.strip().splitlines() if l.strip()]
    if not body:
        return True
    if body[0].lstrip().startswith('"""') and "Notebook 0" in src:
        return True
    pat = re.compile(r"^\s*(import |from |sys\.path|os\.chdir|%matplotlib|"
                     r"matplotlib\.use|try:|except NameError:|get_ipython\(\)|"
                     r"np\.random\.seed|#)")
    hits = sum(1 for l in body if pat.match(l))
    return hits / len(body) >= 0.75


def renumber(md: str, sec: int, counter: list) -> str:
    """`## 3. Foo` -> `## 2.1 Foo`, renumbered sequentially within the section."""
    out = []
    for line in md.splitlines():
        m = re.match(r"^## (\d+)\.\s+(.*)$", line)
        if m:
            counter[0] += 1
            line = f"## {sec}.{counter[0]} {m.group(2)}"
        out.append(line)
    return "\n".join(out)


def clean_summary(md: str, sec: int, counter: list) -> str:
    """Number the trailing summary heading and drop forward references."""
    md = re.sub(r"^### Next:.*?(?=^#{1,3} |\Z)", "", md, flags=re.M | re.S)
    md = re.sub(r"^### Next Steps.*?(?=^#{1,3} |\Z)", "", md, flags=re.M | re.S)
    if re.search(r"^## Summary\s*$", md, flags=re.M):
        counter[0] += 1
        md = re.sub(r"^## Summary\s*$", f"## {sec}.{counter[0]} Summary",
                    md, flags=re.M)
    return md.rstrip() + "\n"


# ---------------------------------------------------------------- assemble

out = nbformat.v4.new_notebook()
base = nbformat.read(f"{SRC}/{SECTIONS[0][0]}", as_version=4)
out.metadata = base.metadata

cells = [new_markdown_cell(INTRO), new_code_cell(SETUP)]
cells[-1].execution_count = None

dropped = 0
for fname, sec, opener in SECTIONS:
    nb = nbformat.read(f"{SRC}/{fname}", as_version=4)
    # anchor target for the contents table
    cells.append(new_markdown_cell(f'<a id="sec{sec}"></a>\n\n' + opener))
    kept = 0
    counter = [0]
    for idx, c in enumerate(nb.cells):
        if c.cell_type == "markdown":
            s = c.source
            if re.match(r"^#\s+Notebook \d+", s.strip()):
                dropped += 1
                continue
            if re.match(r"^##\s+\d+\.\s*Setup", s.strip()):
                dropped += 1
                continue
            s = clean_summary(renumber(s, sec, counter), sec, counter)
            if not s.strip():
                dropped += 1
                continue
            cells.append(new_markdown_cell(s))
            kept += 1
        else:
            if is_boilerplate_code(c.source):
                dropped += 1
                continue
            if fname.startswith("04_") and idx in NB04_FIGS:
                attach_figure(c, NB04_FIGS[idx])
            shrink_outputs(c)
            cells.append(c)
            kept += 1
    print(f"  Section {sec}  {fname:28} kept {kept:3} / {len(nb.cells):3}")

# Display tall outputs in full rather than inside a scroll box
for c in cells:
    if c.cell_type == "code":
        c.metadata.setdefault("jupyter", {})["outputs_scrolled"] = False

n = 0
for c in cells:
    if c.cell_type == "code" and c.get("execution_count") is not None:
        n += 1
        c.execution_count = n
        for o in c.get("outputs", []):
            if o.get("output_type") == "execute_result":
                o["execution_count"] = n

out.cells = cells
nbformat.validate(out)
nbformat.write(out, OUT)

mb = lambda b: b / 1_048_576
print(f"\n  cells written : {len(cells)}  (dropped {dropped} boilerplate)")
print(f"  images        : {stats['converted']} recompressed, "
      f"{stats['reattached']} re-attached to Section 4")
print(f"  image payload : {mb(stats['png_before']):.2f} MB -> "
      f"{mb(stats['jpeg_after']):.2f} MB")
print(f"  wrote         : {OUT}")
