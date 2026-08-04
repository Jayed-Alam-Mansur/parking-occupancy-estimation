# Presentation Outline

Material presented: `notebooks/00_COMPLETE_PROJECT.ipynb`, or
`docs/presentation/parking-occupancy-presentation.html` for a standalone version
that does not require Jupyter.

Target length: approximately 12 minutes, plus questions. A shorter 7-minute
ordering is given at the end.

## Preparation

- Open the HTML export rather than the notebook. It requires no kernel and a cell
  cannot be run accidentally during the presentation.
- Scroll once to the end and back so all images are loaded before starting.
- Set browser zoom to about 80 % so full-width figures fit on screen.
- Keep the notebook open in a second window in case code needs to be shown.

## Section order

### 1. Introduction

State the objective: estimate the number of spaces, the per-space occupancy
label, and the overall occupancy rate from a single surveillance frame using one
fixed camera and no per-space sensors.

State the constraint: no machine learning. TensorFlow, PyTorch, Keras, YOLO and
pretrained models are excluded by the project specification and listed in
`requirements.txt`. Every threshold is set explicitly from measurements.

### 2. Dataset and exploratory analysis

PKLot: approximately 12,400 images from three lots under sunny, cloudy and rainy
conditions, with per-space polygon annotations and occupancy labels. The
annotations provide the ground truth used in Section 8.

Point out the cast shadows on sunny frames. They are dark and roughly car-shaped,
which is why Sections 4 and 5 include shadow handling.

### 3. Perspective transformation

Explain the problem: spaces near the camera occupy more pixels than distant
spaces of the same physical size, so any area-dependent feature would be
influenced by position in the frame rather than by occupancy.

Explain the solution: the parking surface is planar, so the camera view and an
overhead view are related by a homography, a single 3x3 matrix. Four point
correspondences are sufficient to solve for it.

Show the before-and-after comparison and note that spaces are now approximately
equal in size.

### 4. Region of interest extraction

The annotation polygons are transformed by the same homography and each space is
cropped and masked. Adjacent spaces share painted boundary lines, so the polygons
overlap; each mask is eroded inward so that a vehicle in one space does not
contribute pixels to its neighbour.

### 5. Preprocessing

Four steps per space: grayscale conversion, CLAHE, Gaussian blur, median filter.

CLAHE equalises contrast per tile rather than globally, so sunlit and shaded
regions of the same frame are corrected independently.

If asked why the median filter is applied after the Gaussian blur: impulse noise
is a rank-order problem rather than a convolution problem, so averaging cannot
remove a single outlying pixel but a rank filter can.

### 6. Segmentation and morphology

Three thresholding methods were compared. Global thresholding uses one cutoff and
fails when illumination varies. Otsu's method selects the cutoff automatically but
assumes a bimodal histogram. Adaptive thresholding handles illumination gradients
but can report texture in an empty space. The Otsu and adaptive results are fused.

Shadows defeat all intensity-based methods, so HSV is used: a shadowed region
loses value while retaining hue, so shadowed road surface can be distinguished
from a vehicle by colour.

Morphological opening removes isolated noise and closing fills interior gaps.

### 7. Feature extraction

Each space is reduced to eight scalar features, each with a physical
interpretation. Edge density, foreground ratio, gradient magnitude, local
variance, largest connected component, intensity standard deviation, Otsu
separability and mean saturation.

The Fisher discriminant ratio is computed for each feature to quantify class
separation. Edge density ranks highest by a wide margin. This is relevant to the
comparison in Section 9.

### 8. Threshold calibration

A two-stage cascade is used rather than a trained classifier, so that each
decision remains traceable to a specific measurement.

The fast path resolves spaces whose edge density is clearly low or clearly high,
which accounts for about one third of spaces. Remaining spaces are scored with
the Fisher ratios as weights and compared against a decision threshold.

The threshold is selected by sweeping its range and recording accuracy and F1,
not by inspection. The resulting curve has a broad maximum.

### 9. Evaluation

120 frames across all three weather conditions, giving 11,599 space
classifications compared against ground truth.

| Metric | Value |
|---|---|
| Accuracy | 74.30 % |
| F1 score | 0.7310 |
| Processing time | 77.33 ms per frame (12.9 FPS), CPU only |

Rainy frames are the weakest case, because wet road surface produces reflections
that register as edges.

Section 8.5 compares the full cascade against a baseline that thresholds edge
density alone. The single-feature baseline achieves higher accuracy. Three likely
causes:

1. The Fisher weights were computed on a small, comparatively clean sample and
   then applied to a larger and more variable set, so the weighting does not
   transfer.
2. The seven weaker features are mutually correlated rather than independent, so
   averaging them reduces rather than reinforces the contribution of the most
   discriminative feature.
3. Each additional feature introduces its own failure cases. Mean saturation
   performs poorly on grey vehicles; local variance performs poorly on wet road
   surface.

The conclusion is that increasing the number of features does not necessarily
increase discriminative power, and that the weighting procedure requires
revision.

### 10. Discussion and conclusion

What the system provides: one camera, no per-space sensors, no training data, no
GPU, real-time operation on a laptop CPU, and decisions traceable to named pixel
statistics.

Limitations: 74.30 % accuracy is a demonstration rather than a deployable system;
rainy conditions are the weakest case; the homography is calibrated manually per
camera; and the cascade is outperformed by a single-feature baseline.

Further work, in order of expected benefit:

1. Adopt the single-feature baseline, which is more accurate and faster.
2. Recompute the feature weights on the full 11,599-sample set.
3. Remove features that measurably reduce accuracy.
4. Add temporal smoothing across consecutive frames.

## Expected questions

Why not use a neural network such as YOLO?
: It would be more accurate, and would be the appropriate choice in production.
  The project specification excludes machine learning, and building the pipeline
  from primitives requires each stage to be justified.

Is 74.30 % accuracy sufficient?
: It is a demonstration rather than a product. The figure was measured on 11,599
  samples rather than on a small selected set, and the comparison in which the
  baseline outperforms the cascade is reported rather than omitted.

How well would this generalise to a different lot?
: Not without recalibration. The homography is calibrated per camera, so a new
  installation requires four new point correspondences. Recalibration is a
  configuration change rather than a hardware change.

Why use eight features if only one proved discriminative?
: The ranking was not known in advance. Eight features with physical
  justifications were implemented and then measured. A revised approach would
  start from the Fisher ranking and add features only where they improve
  accuracy.

How were the four correspondence points chosen?
: They are corners of the outermost painted space markings. They are coplanar
  with the ground, visible in every frame, and widely separated, which keeps the
  homography well conditioned.

Is the ground truth reliable?
: It is the published per-space annotation from the PKLot dataset, produced by the
  original researchers. It is a standard benchmark, which is why it was used
  rather than manual labelling.

## Shorter ordering

For a 7-minute presentation, retain sections 1, 2, 3, 9 and 10. Compress section
6 to the HSV shadow argument and section 7 to the Fisher ranking. Omit section 4
and the three-method threshold comparison in section 6.
