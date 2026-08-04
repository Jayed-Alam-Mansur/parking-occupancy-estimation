# Demonstration Notes for the Streamlit Application

Application: `app.py`, started with `streamlit run app.py`.
Companion to [`PRESENTATION_SCRIPT.md`](PRESENTATION_SCRIPT.md), which covers the
notebook.

All figures below were measured on the sample set in `data/samples/`. Where a
single frame and the full 21-frame set disagree, both are given; the full-set
figure is the one to quote.

## Preparation

- Start the application before the session so imports and caches are warm.
- Open the Feature Comparison tab once and return. The comparison takes about
  1.6 seconds across 1,999 spaces on first run and is cached afterwards.
- Set browser zoom to about 80 % so the sidebar and a 100-space image both fit.
- Confirm the frame selector reads `cloudy, 2012-10-16_17_53_55, 60% full`, which
  is the default. The figures below assume it.
- `R` reruns the application. Reset to tuned values in the sidebar restores all
  parameters.

## Suggested order

### 1. Perspective correction

Open the Pipeline Stages tab. The original and corrected views are shown side by
side. Note that distant spaces are compressed in the original view and
approximately uniform after correction, which is what makes area-dependent
measurements comparable across the lot.

### 2. Per-space processing stages

Select a space whose ground-truth label is occupied. The first row shows the
preprocessing stages and the second row the segmentation and morphology stages.
The final binary image is the input to feature extraction.

The feature table below shows the eight extracted values, their weights and their
contributions, together with the predicted label and whether it matches ground
truth.

Repeat with a vacant space and note the reduced foreground area and lower edge
density.

### 3. Parameter sensitivity

Switch to the Full Lot Result tab so the accuracy figure is visible, then change
one parameter at a time in the sidebar.

| Parameter change | Effect on this frame | Effect on the full sample set |
|---|---|---|
| Core-mask erosion 3 to 0 | 63 % to 56 % | -7.05 points |
| Morphological opening 3 to 7 | 63 % to 45 % | -10.01 points |
| Gaussian and median filters both disabled | - | -10.91 points |

With no erosion, the mean edge density rises from 0.161 to 0.180, because each
space now includes pixels from vehicles in adjacent spaces. This is the effect
that the eroded core mask in Section 3 is intended to prevent.

Increasing the opening kernel removes the vehicle region itself, which increases
the false-vacant count.

With both denoising filters disabled, sensor and compression noise survives into
the binary mask and is measured as texture.

Use Reset to tuned values afterwards.

### 4. Weather conditions

Change the weather selector to sunny.

| Condition | Accuracy on the sample set |
|---|---|
| Cloudy | 77.4 % |
| Rainy | 72.8 % |
| Sunny | 61.2 % |

Sunny frames are the weakest case because cast shadows are dark and car-shaped,
which is what the HSV stage in Section 5 addresses. The HSV stage reduces the
error but does not eliminate it.

Return the selector to cloudy.

### 5. Feature comparison

Open the Feature Comparison tab.

| Configuration | Accuracy | F1 |
|---|---|---|
| `edge_density` alone | 82.69 % | 0.809 |
| Eight-feature cascade | 70.94 % | 0.595 |
| Difference | +11.76 points | - |

The single-feature baseline is higher across a broad range of thresholds, not
only at the optimum, which the sweep curve shows. Move the threshold slider to
show the curve rising and falling relative to the constant cascade accuracy.

The likely causes are stated in the tab and in Section 8 of the notebook: weights
fitted on a small clean sample, mutual correlation among the weaker features, and
additional failure cases introduced by each extra feature.

## Parameters that do not respond as expected

The CLAHE clip limit has little effect, and on most spaces no effect at all. On an
86 x 20 pixel space with an 8 x 8 tile grid, every clip value between 0.5 and 8.0
produces an identical image. OpenCV computes the effective clip limit as
`max(1, clipLimit * tileArea / 256)`; with 10 x 2 pixel tiles this expression
evaluates to 1 for every value on the slider. Only the larger spaces respond,
which is why the full-set accuracy changes by only 2.00 points at a clip limit of
8.0.

If the clip limit is raised during the demonstration, the explanation above is the
useful answer. A coarser tile grid would be required for the clip limit to take
effect on regions this small.

Two further notes:

- Disabling the median filter alone increases full-set accuracy by 0.80 points, so
  it should only be shown together with the Gaussian filter.
- Single-frame figures are noisy. Disabling the Gaussian filter improves accuracy
  by 11 points on the default frame but reduces it by 3.25 points across the full
  sample set. Quote the full-set figures.

## Expected questions

Are the results recomputed, or cached?
: Recomputed. Each parameter change reruns all 100 spaces through the pipeline,
  which is the frame time shown in the Full Lot Result tab, between 75 and 100 ms.
  Results are cached per parameter combination, so returning to a previous setting
  is immediate.

Why does the application report 70.94 % when the notebook reports 74.30 %?
: Different evaluation sets, as noted in the tab. The application uses the 21
  sample frames in the repository, giving 1,999 spaces. The 74.30 % figure is from
  120 evenly spaced frames, giving 11,599 spaces. The pipeline and configuration
  are the same.

Does the application require the full dataset?
: No. It uses the 21 frames committed to `data/samples/`, about 7.6 MB.

Was the pipeline reimplemented for the application?
: No. Each stage calls the same function in `src/` that the notebooks call.

Do the parameters start at the tuned values?
: Yes. All fourteen are initialised from `config/thresholds.yaml` and the tuned
  defaults in `src/`, so the application reproduces the stored configuration until
  a parameter is changed.

## Recovery

| Problem | Action |
|---|---|
| Unexpected figures after changing parameters | Reset to tuned values in the sidebar |
| Page appears unresponsive | Press `R` to rerun |
| Server has stopped | Run `streamlit run app.py` again; caches rebuild in about 2 seconds |
| Application will not start | Present the notebook instead; the same figures are in Section 8 |
