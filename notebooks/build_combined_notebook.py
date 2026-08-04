"""Assemble the 9 project notebooks into one presentation notebook.

Preserves every executed output. Removes repeated boilerplate, renumbers
sections into a single arc, and inserts narrative cells between acts.
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

# ---------------------------------------------------------------- narrative

PROLOGUE = r"""# Finding a Parking Space With Nothing but Mathematics

### Automatic Parking Occupancy Estimation using Classical Image Processing
**Course:** Digital Image Processing  **Author:** Jayed Alam Mansur

---

## The problem

You drive into a 100-space car park. It looks full. You circle it twice, burn
fuel, block traffic, and eventually find a space that was free the whole time —
just out of sight behind a van.

The industry solution is **one sensor per bay**. A hundred spaces means a
hundred ultrasonic or magnetometer units to trench in, wire, power, network and
maintain. It works, and it is expensive.

**This project replaces all hundred of them with one camera and some geometry.**

## The rules I gave myself

There is a version of this project that takes twenty lines and an afternoon:
import YOLO, load pretrained weights, draw boxes. It would work. It would also
teach me nothing about images.

So the constraint here is absolute, and it is written into `requirements.txt`:

> **No TensorFlow. No PyTorch. No Keras. No YOLO. No pretrained models of any
> kind.**

Everything you are about to see is built from `cv2` primitives and NumPy
arrays. Every threshold was chosen by a human who can defend it. There is no
black box anywhere in this notebook — when the system is wrong, I can point at
the exact pixel statistic that caused it.

## The three questions the system answers

Given a single surveillance frame, it reports:

1. **How many spaces are there?** → `N`
2. **Which specific bays are occupied?** → a per-bay `OCCUPIED` / `VACANT` label
3. **How full is the lot?** → `O% = N_occupied / N × 100`

## Where we end up

I will not bury the result. On **11,599 held-out slot samples** from the PKLot
benchmark:

| Metric | Value |
|---|---|
| Accuracy | **74.30 %** |
| F1 score | **0.7310** |
| Throughput | **77.33 ms / frame → 12.9 FPS** on a laptop CPU, no GPU |

And one more finding I could have hidden and did not:

> **My single simplest feature beat my entire eight-feature system.**

That result, why it happens, and what it taught me is **Act 8**. It is the most
interesting thing in this project.

---

## The route — click any act to jump straight to it

| Act | Question it answers |
|---|---|
| [**1 · Know Your Data**](#act1) | What does the data actually look like? |
| [**2 · The Camera Is Lying**](#act2) | How do I undo the camera's perspective distortion? |
| [**3 · Carving 100 Bays**](#act3) | How do I cut the lot into 100 independent bays? |
| [**4 · Noon vs Dusk**](#act4) | How do I make bays photographed hours apart comparable? |
| [**5 · Grey to Black-and-White**](#act5) | How do I turn grey pixels into a clean binary decision? |
| [**6 · Eight Numbers**](#act6) | What eight numbers describe "is there a car here?" |
| [**7 · Drawing the Line**](#act7) | Where exactly do I draw the yes/no line? |
| [**8 · The Verdict & the Twist**](#act8) | Does it work on 11,599 samples — and what went wrong? |
| [**9 · What It Means**](#act9) | What it means, and what I would do next |

---

> ℹ️ **About this notebook.** It is a compiled build: the nine development
> notebooks (`01_explore` → `09_final_report`) merged into one continuous
> story, with all original executed outputs preserved. Repeated import blocks
> were collapsed into the single setup cell below. Large photographic outputs
> were recompressed to JPEG to keep the file openable.
"""

SETUP = '''# Setup — every import used anywhere in this project, in one place
# -----------------------------------------------------------------
import os, sys, time
from pathlib import Path

# Run from the project root so every relative path below resolves
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, os.path.abspath('.'))

# Third-party: image processing and plotting only
#    Note what is absent: no tensorflow, no torch, no ultralytics.
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

# The eight features every parking bay gets reduced to (Act 6)
feature_names = ['edge_density', 'foreground_ratio', 'gradient_magnitude',
                 'local_variance', 'largest_component', 'intensity_std',
                 'otsu_separability', 'mean_saturation']

DATA_ROOT = 'data/raw/PKLot'
config = load_config('config/config.yaml')
'''

ACTS = [
    ("01_explore.ipynb", 1, r"""---
# Act 1 — Know Your Data Before You Touch It

Every bad computer-vision project starts the same way: someone writes the
clever algorithm first and looks at the images second.

So before a single filter runs, I want to answer plain questions. How many
frames do I have? What resolution? How different does the same car park look on
a sunny day versus a rainy one? How are the ground-truth labels stored, and can
I trust them?

**The dataset:** PKLot, from the Federal University of Paraná — roughly 12,400
images of three car parks across sunny, cloudy and rainy weather, each with
per-bay polygon annotations and an occupied/vacant label. That last part is
what makes honest evaluation possible later.

> **Note:** the sample grid. Sunny frames have brutal shadows that look
> exactly like dark cars. That single observation drives half the design
> decisions in Acts 4 and 5.
"""),
    ("02_geometry.ipynb", 2, r"""---
# Act 2 — The Camera Is Lying To You

Look again at the raw frame from Act 1. Bays near the camera are large
rectangles. Bays at the far end are small slivers. They are the same physical
size — the camera's perspective projection is distorting them.

This matters enormously. If I measure "how many edge pixels are in this bay?"
on the raw image, a far bay scores lower than a near bay **for reasons that
have nothing to do with whether a car is parked there**. Every feature I build
would be contaminated by position.

The fix is the highest-value step in the entire project. A car park surface is
**planar**, and any two views of a plane are related by a **homography** — a
single 3×3 matrix. Find four corresponding points, solve for `H`, and I can
re-photograph the lot from an imaginary camera hovering directly overhead.

$$s\begin{bmatrix}x'\\y'\\1\end{bmatrix} = \mathbf{H}\begin{bmatrix}x\\y\\1\end{bmatrix}$$

> **Note:** the before/after comparison. Bays that were trapezoids
> become rectangles of near-identical size. That is what makes every later
> measurement fair.
"""),
    ("03_roi.ipynb", 3, r"""---
# Act 3 — Carving the Lot Into 100 Bays

I now have a clean overhead view. Next I need to stop thinking about "an
image" and start thinking about **100 independent little problems** — one per
parking bay.

The annotations give me a polygon per bay, which I push through the same
homography `H` so they land correctly on the overhead view. Then each bay gets
cropped and masked.

But there is a trap. Painted bay lines are shared: the polygon for bay 12
slightly overlaps bay 13. If a large car in bay 13 spills a few pixels into
bay 12's region, bay 12 reads as occupied when it is empty.

The fix is deliberately unglamorous — **erode each mask inward** and only
measure the confident core of each bay. I trade a little signal for a lot of
independence.

> **Note:** the ROI grid. 100 clean, separated, near-identical
> rectangles. Compare that to the raw frame in Act 1 and the value of the last
> two acts becomes obvious.
"""),
    ("04_preprocessing.ipynb", 4, r"""---
# Act 4 — Making Noon and Dusk Comparable

I have 100 clean bay images. I cannot compare them yet.

A bay photographed in direct sun and the same empty bay photographed under
cloud produce wildly different pixel values. If my threshold is tuned for one,
it fails on the other. Worse, a single sunny frame has a bright half and a
shadowed half — so even *one global correction is not enough*.

So each bay climbs a four-step ladder:

| Step | Operation | What problem it solves |
|---|---|---|
| 1 | **Grayscale** | Colour is not what tells me a car is there. Drop 3 channels to 1. |
| 2 | **CLAHE** | Uneven lighting. Equalise *per tile*, not globally, so sun-side and shadow-side are fixed independently. |
| 3 | **Gaussian blur** | High-frequency sensor noise that would become fake edges. |
| 4 | **Median filter** | Salt-and-pepper speckle from JPEG compression. |

The order is not arbitrary. Median comes *after* Gaussian because impulse noise
is a rank-order problem, not a convolution problem — you cannot average away a
single wildly-wrong pixel, but you can rank it out.

> **Note:** the CLAHE histograms. A cramped, bunched-up histogram
> spreads across the full range — that is contrast being manufactured out of
> nothing but arithmetic.
"""),
    ("05_segmentation.ipynb", 5, r"""---
# Act 5 — From Grey to Black-and-White

Now the real decision: which pixels are **car** and which are **asphalt**?

I tried three classical thresholding methods and none of them is right alone:

- **Global threshold** — one cutoff for everything. Fast, and fails the instant
  lighting varies.
- **Otsu** — picks the cutoff automatically by maximising between-class
  variance. Elegant, but assumes the histogram has two clear humps.
- **Adaptive** — a different cutoff for every neighbourhood. Handles gradients
  beautifully, and happily hallucinates texture in a perfectly empty bay.

So I **fuse** them rather than pick a winner.

Then there is the shadow problem from Act 1. A hard shadow is dark, car-shaped,
and fools every intensity-based method. The escape is to leave intensity
behind: convert to **HSV**, where a shadow is a region that lost *value* while
keeping its *hue* — asphalt in shadow is still asphalt-coloured.

Finally, **morphology** cleans the binary mask: opening removes speckle,
closing seals holes inside the car body.

> **Note:** the four-way threshold comparison, then the morphology
> stages. Watch a noisy, scattered mask resolve into one solid car-shaped blob.
"""),
    ("06_features.ipynb", 6, r"""---
# Act 6 — Eight Numbers That Describe a Parking Bay

Here is the compression step that makes the whole project tractable.

Each bay is thousands of pixels. I reduce it to **eight scalars**, chosen so
that each one has a physical story I can tell:

| # | Feature | Why a car changes it |
|---|---|---|
| 1 | `edge_density` | A car is a box of hard edges. Empty asphalt is smooth. |
| 2 | `foreground_ratio` | How much of the bay the binary mask marked as "not ground". |
| 3 | `gradient_magnitude` | Average Sobel response — texture strength. |
| 4 | `local_variance` | Empty tarmac is uniform; car bodywork is not. |
| 5 | `largest_component` | One big connected blob = car. Scattered specks = noise. |
| 6 | `intensity_std` | Cars introduce highlights and shadow together. |
| 7 | `otsu_separability` | How cleanly the histogram splits into two — a proxy for "is there really an object here?" |
| 8 | `mean_saturation` | Painted cars are colourful. Asphalt is grey. |

And then the honest question: **are these eight actually any good?** I don't
guess — I compute the **Fisher discriminant ratio** for each, which measures how
far apart the occupied and vacant distributions sit relative to their spread:

$$F = \frac{(\mu_{occ} - \mu_{vac})^2}{\sigma_{occ}^2 + \sigma_{vac}^2}$$

High F means the feature separates the two classes cleanly. Low F means it is
noise wearing a useful-sounding name.

> **Note:** the per-feature histograms, then the Fisher ranking.
> `edge_density` wins by a wide margin. **Remember that.** It comes back in
> Act 8 in a way I did not expect.
"""),
    ("07_threshold_tuning.ipynb", 7, r"""---
# Act 7 — Where Exactly Do We Draw the Line?

Eight numbers per bay. One binary answer required. Something has to convert
one into the other, and this is the part where most projects quietly reach for
a classifier.

I don't — deliberately. A trained model would be a black box, and the entire
premise here is that every decision must be explainable. So instead:

**A two-stage cascade.**

1. **Fast path.** If `edge_density` is extremely low, the bay is obviously
   empty — call it and move on. If it is extremely high, it is obviously
   occupied. Roughly a third of bays are decided here, cheaply.
2. **Scoring path.** Everything ambiguous gets a weighted vote, where each
   feature's weight is *its Fisher ratio from Act 6* — features that proved
   themselves get more say:

$$S = \frac{\sum_k w_k f_k}{\sum_k w_k} \qquad \text{occupied if } S > \tau$$

The remaining question is the value of τ. I don't guess that either — I sweep
it across its whole range and plot accuracy and F1 as functions of the cutoff,
then read the optimum off the curve.

> **Note:** the threshold sweep. The curve has a broad, flat peak — the
> system is not balanced on a knife-edge, which is exactly what you want.
"""),
    ("08_evaluation.ipynb", 8, r"""---
# Act 8 — The Verdict, and the Twist

Three hand-picked frames prove nothing. This act is where the project either
holds up or does not.

**The test:** 120 frames sampled evenly across all three weather conditions →
**11,599 individual bay judgements**, each checked against ground truth. Plus
per-weather breakdowns, a ranking of the worst-performing bays, and a
stage-by-stage timing profile.

The headline: **74.30 % accuracy, 0.7310 F1, 12.9 FPS on CPU.**

---

### And then the result I did not want

Section 6 below compares the full eight-feature cascade against a baseline so
simple it is almost a joke: **threshold `edge_density` alone. One number. No
scoring, no weights, no cascade.**

**The one-number baseline wins.**

I could have deleted that comparison. Instead it is the most valuable thing I
learned, so here is my reading of why it happens:

- **Fisher ratios were measured on a small, clean sample** and then applied to
  a large, messy one. Weights fitted on easy data do not transfer.
- **The seven weaker features are correlated with each other**, not
  independent. Averaging seven noisy, correlated votes does not cancel error —
  it just dilutes the one feature that actually worked.
- **Every added feature adds its own failure modes.** `mean_saturation` breaks
  on grey cars. `local_variance` breaks on wet, textured tarmac. The cascade
  inherits all of them at once.

The lesson is the one nobody enjoys learning: **more features is not more
signal.** A complicated system that underperforms its own simplest component
is telling you something, and the professional move is to listen rather than
to bury the comparison.

> **Note:** the confusion matrices, the per-weather bars (rainy is the
> hard case), and Section 6 — the twist.
"""),
    ("09_final_report.ipynb", 9, r"""---
# Act 9 — What It Means, and What I'd Do Next

The pipeline runs end to end. Time to be straight about what was built, what it
is worth, and where it breaks.

**What works.** One camera, no sensors, no training data, no GPU, real-time on
a laptop — and every single decision traceable to a named pixel statistic. Point
at any wrong answer and I can tell you which feature caused it. Almost no
learned system can offer that.

**What doesn't.** 74.30 % is a working demonstration, not a product. Rainy
frames are the weak point — wet asphalt reflects, and reflections have edges.
The homography is hand-calibrated per camera, so a new site means new
correspondence points. And, as Act 8 showed, my cascade is beaten by its own
simplest ingredient.

**What I would do next**, in the order I would actually do it:

1. **Ship the baseline.** Edge density alone is more accurate *and* faster.
   The honest engineering decision is to keep the simple thing.
2. **Re-fit the weights on the full 11,599 samples**, not the small clean set —
   test directly whether the cascade was mis-weighted rather than misconceived.
3. **Drop the features that fail measurably** instead of keeping all eight for
   symmetry.
4. **Add temporal smoothing.** Cars do not teleport. A bay that reads occupied
   for one frame out of thirty is a glitch, and a running vote across frames
   would erase a whole class of error for almost no compute.
"""),
]

RECAP_TITLES = {
    1: "Act 1 recap — what the data told us",
    2: "Act 2 recap — the lot, seen from above",
    3: "Act 3 recap — 100 independent bays",
    4: "Act 4 recap — every bay now comparable",
    5: "Act 5 recap — clean binary masks",
    6: "Act 6 recap — eight numbers, ranked",
    7: "Act 7 recap — the line is drawn",
    8: "Act 8 recap — the honest numbers",
    9: "Act 9 recap",
}

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
    """Recompress heavy photographic PNG outputs to JPEG in place."""
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
            print(f"    ! recompress failed: {e}", file=sys.stderr)
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


def renumber(md: str, act: int, counter: list) -> str:
    """`## 3. Foo` -> `## 2.1 Foo`, renumbered sequentially within the act."""
    out = []
    for line in md.splitlines():
        m = re.match(r"^## (\d+)\.\s+(.*)$", line)
        if m:
            counter[0] += 1
            line = f"## {act}.{counter[0]} {m.group(2)}"
        out.append(line)
    return "\n".join(out)


def clean_recap(md: str, act: int) -> str:
    md = re.sub(r"^### Next:.*?(?=^#{1,3} |\Z)", "", md, flags=re.M | re.S)
    md = re.sub(r"^### Next Steps.*?(?=^#{1,3} |\Z)", "", md, flags=re.M | re.S)
    md = re.sub(r"^## (Phase Summary|Day 2 Complete!).*$",
                f"## {RECAP_TITLES[act]}", md, flags=re.M)
    md = re.sub(r"^### Day 2 Complete!.*$", "", md, flags=re.M)
    return md.rstrip() + "\n"


# ---------------------------------------------------------------- assemble

out = nbformat.v4.new_notebook()
base = nbformat.read(f"{SRC}/{ACTS[0][0]}", as_version=4)
out.metadata = base.metadata

cells = [new_markdown_cell(PROLOGUE), new_code_cell(SETUP)]
cells[-1].execution_count = None

dropped = 0
for fname, act, opener in ACTS:
    nb = nbformat.read(f"{SRC}/{fname}", as_version=4)
    # explicit anchor so the prologue's route table can jump here
    cells.append(new_markdown_cell(f'<a id="act{act}"></a>\n\n' + opener))
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
            s = clean_recap(renumber(s, act, counter), act)
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
    print(f"  Act {act}  {fname:28} kept {kept:3} / {len(nb.cells):3}")

n = 0
for c in cells:
    if c.cell_type == "code" and c.get("execution_count") is not None:
        n += 1
        c.execution_count = n
        for o in c.get("outputs", []):
            if o.get("output_type") == "execute_result":
                o["execution_count"] = n

# Never let Jupyter bury a tall output inside a small scroll box mid-talk
for c in cells:
    if c.cell_type == "code":
        c.metadata.setdefault("jupyter", {})["outputs_scrolled"] = False

out.cells = cells
nbformat.validate(out)
nbformat.write(out, OUT)

mb = lambda b: b / 1_048_576
print(f"\n  cells written : {len(cells)}  (dropped {dropped} boilerplate)")
print(f"  images        : {stats['converted']} recompressed, "
      f"{stats['reattached']} re-attached to Act 4")
print(f"  image payload : {mb(stats['png_before']):.2f} MB -> "
      f"{mb(stats['jpeg_after']):.2f} MB")
print(f"  wrote         : {OUT}")
