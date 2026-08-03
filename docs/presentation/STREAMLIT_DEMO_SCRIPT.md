# 🎛️ Live Demo Script — The Streamlit Explorer

**App:** `app.py` · launch with `streamlit run app.py`
**Length:** ~6 minutes as written. Trim markers included for a 3-minute version.
**Companion to:** [`PRESENTATION_SCRIPT.md`](PRESENTATION_SCRIPT.md) — that one drives the notebook, this one drives the app.

> **Every number in this guide was measured on this machine, not estimated.** Where a single frame and the full sample corpus disagree, both are given — and the corpus number is the one to trust out loud.

---

## Before You Start

| ✅ | Check |
|---|---|
| ☐ | `streamlit run app.py` **before** the class walks in. First launch imports OpenCV and warms caches. |
| ☐ | **Click the 🎚️ Twist tab once, then come back.** Cold compute is ~1.6 s across 1,999 bays; after that it is instant. Do this warm-up off-stage. |
| ☐ | Browser zoom **80 %**. The sidebar plus a 100-bay image needs the width. |
| ☐ | Confirm the header reads **cloudy · 2012-10-16_17_53_55 · 60% full**. That is the default and the numbers below assume it. |
| ☐ | Know the panic button: **`R`** reruns, and **↺ Reset to tuned values** in the sidebar restores everything. |

**The one-line pitch when you switch to the app:**

> *"Everything I've shown you so far is a static result. This is the same pipeline with every parameter exposed — so instead of telling you why each stage is there, I can delete it and let you watch what breaks."*

---

## Act 1 · The homography, side by side (45 s)

**Do:** Land on the **🔬 Pipeline Explorer** tab. Both images are already on screen.

> *"Left is what the camera sees. Right is the same frame after the homography.*
>
> *Look at the far end on the left — those bays are slivers. On the right they're
> rectangles, roughly the same size as the near ones. That's not cosmetic. If I
> measure edge density on the left image, a far bay scores lower than a near bay
> for reasons that have nothing to do with whether a car is parked in it. Every
> feature would secretly be measuring distance from the camera."*

---

## Act 2 · One bay, all nine stages (75 s)

**Do:** Scroll down. Leave the radio on **OCCUPIED**, pick any bay.

> *"Now I'll follow a single bay through the whole pipeline. This bay genuinely
> has a car in it — that's from the ground-truth annotation, not my system.*
>
> *Top row is the preprocessing ladder: the raw crop, grayscale, CLAHE, Gaussian,
> median. Bottom row is segmentation: Otsu, adaptive, the two fused, then
> morphology cleaning it up.*
>
> *And that last binary image is what the eight features get measured on."*

**Do:** Point at the feature table and its contribution bars.

> *"Thousands of pixels reduced to eight numbers, each weighted by how well it
> separated the classes back in Act 6. The verdict at the top says whether the
> system got it right — checked against ground truth, so you can catch it lying."*

**Do:** Switch the radio to **VACANT** and pick a bay.

> *"Same pipeline, empty bay. Watch the binary image — far less white — and the
> edge density drops accordingly."*

✂️ *3-minute version: keep this, drop Act 1's narration and just gesture at it.*

---

## Act 3 · Break it on purpose (2 min) — **the reason this app exists**

> *"Here's what a static figure can't do. I'm going to break my own pipeline,
> one stage at a time, and you can watch the accuracy move."*

Switch to the **🅿️ Whole Lot** tab first so the accuracy metric is visible while you drag.

### These three are verified and land hard

| # | Do this | What happens | Say this |
|---|---|---|---|
| **1** | Sidebar → **Core-mask erosion 3 → 0** | **63 % → 56 %** on this frame · **−7.05 pts** corpus-wide. Mean edge density *rises* 0.161 → 0.180 | *"That's Act 3. Painted bay lines are shared, so the polygons overlap. With no erosion, a car in bay 13 leaks pixels into bay 12 — and notice the edge density went **up**, because it's now measuring the neighbour's car."* |
| **2** | **Morph. opening 3 → 7** | **63 % → 45 %** this frame · **−10.01 pts** corpus | *"Opening removes speckle. Push the kernel too far and it eats the car itself — false-vacant errors jump. This is why the kernel size is a tuned parameter and not a guess."* |
| **3** | **Gaussian 5 → 1 AND Median 3 → 1** (both, together) | **−10.91 pts** corpus | *"Both denoisers off. Sensor noise and JPEG speckle survive into the binary mask and get counted as texture — as evidence of a car that isn't there."* |

**Then reset:** click **↺ Reset to tuned values**.

> *"And that's the argument for every stage in the pipeline, made by deletion
> rather than assertion."*

✂️ *3-minute version: do **#1 only**. It's the most surprising and it ties directly back to Act 3.*

---

## Act 4 · The weather, which is the honest part (30 s)

**Do:** Sidebar → **Weather → sunny**.

> *"Cloudy is my best case — flat, even light. Watch what sunshine does."*

| Weather | Corpus accuracy |
|---|---|
| Cloudy | **77.4 %** |
| Rainy | **72.8 %** |
| Sunny | **61.2 %** |

> *"Sixteen points worse in sunshine. Those hard shadows are dark, car-shaped,
> and they defeat every intensity-based method — which is exactly why Act 5 goes
> to HSV, where a shadow keeps its hue and only loses value. It helps. It clearly
> doesn't solve it."*

**Do:** Set weather back to **cloudy** before moving on.

---

## Act 5 · The twist, live (90 s) — **your closing move**

**Do:** Click **🎚️ The Twist (Act 8)**.

> *"Last thing. In Act 6 the Fisher ranking told me edge density was my strongest
> feature by a wide margin. So I ran the experiment I didn't want to run: throw
> away the other seven, threshold that one number, and see who wins."*

**Do:** Let the three metrics sit on screen. Do not rush this.

| | |
|---|---|
| One number: `edge_density` | **82.69 %** · F1 0.809 |
| Full 8-feature cascade | **70.94 %** · F1 0.595 |
| Gap | **+11.76 pts — baseline wins** |

> *"Eleven and a half points. My entire weighted cascade, beaten by thresholding
> a single number.*
>
> *And the curve shows it isn't a fluke of one lucky cutoff — the baseline beats
> the cascade across a broad range of thresholds. That flat line is my cascade.
> The arc above it is one number."*

**Do:** Drag the cutoff slider slowly left, then right, so they see the arc rise and fall over the flat line.

> *"Three reasons I think this happens. The Fisher weights were fitted on a small
> clean sample and applied to messy data. The seven weaker features are
> correlated with each other rather than independent, so averaging them doesn't
> cancel error — it dilutes the one feature that worked. And every feature I
> added brought its own failure mode: saturation breaks on grey cars, local
> variance breaks on wet tarmac.*
>
> *More features is not more signal. When a complicated system loses to its own
> simplest ingredient, the professional move is to listen to it."*

---

## ⚠️ Landmines — do not touch these live

**The CLAHE clip-limit slider does almost nothing, and on most bays literally nothing.**

I verified this: on an 86 × 20 px bay at tile grid 8, **every clip value from 0.5 to 8.0 produces a bit-identical image.** OpenCV's effective clip is `max(1, clipLimit × tileArea / 256)`; with 10 × 2 px tiles that expression clamps to 1 for every value on the slider. Only the larger bays respond at all, which is why the corpus moves a mere −2.00 pts at clip 8.0.

- **Don't** drag it expecting a dramatic visual. Nothing will happen and you'll be standing there.
- **Do** have the explanation ready — it is genuinely good material:

> *"Interesting one — the clip limit barely matters here, and I had to work out
> why. OpenCV scales it by tile area. My bays are about 86 by 20 pixels, so at an
> 8×8 grid each tile is 10 by 2 pixels, and the effective clip floors at 1 for
> every value on that slider. On crops this small, CLAHE is really just per-tile
> equalisation. If I wanted the clip limit to do work, I'd need a coarser grid."*

**Two more, briefly:**

| Control | Why to avoid |
|---|---|
| **Median kernel alone** | Turning it off *improves* the corpus by **+0.80 pts**. Median is marginal here. Only ever demo it together with Gaussian. |
| **Single-frame numbers** | One frame is noisy — turning Gaussian off *helped* by 11 pts on the demo frame but *hurts* by 3.25 pts corpus-wide. Quote corpus numbers out loud. |

---

## Prepared Q&A

**"Is it recomputing live, or are those cached results?"**
> *"Genuinely recomputing. Every slider change reruns all 100 bays through the
> full pipeline — that's the frame-time readout on the Whole Lot tab, about 75 to
> 95 milliseconds. Results are cached per parameter combination, so going back to
> a setting you've already tried is instant."*

**"Why is the app's cascade 70.94 % when your report says 74.30 %?"**
> *"Different evaluation sets, and it's noted in the app. The app runs on the 21
> curated frames committed to the repo — 1,999 bays. The 74.30 % is from 120
> evenly-spaced frames, 11,599 bays. Same pipeline, same config, different sample."*

**"Does this need the full dataset to run?"**
> *"No — that's deliberate. It runs on 21 frames committed to the repo, about
> 7.6 MB. Anyone can clone it and run this without the 7.5 GB PKLot download."*

**"Did you reimplement the pipeline for the app?"**
> *"No, and that was the constraint I set myself. Every stage calls the same
> function in `src/` that the notebooks call. If the app and the notebook ever
> disagreed, one of them would be lying."*

**"Are the sliders starting at your tuned values?"**
> *"Yes — all fourteen start at exactly what's in `config/thresholds.yaml` and the
> tuned defaults in `src/`. So the app reproduces the committed configuration
> until I deliberately break something."*

---

## If it breaks

| Symptom | Fix |
|---|---|
| Odd numbers after fiddling | **↺ Reset to tuned values** (sidebar, bottom) |
| Page looks stuck | Press **`R`** to rerun |
| Server died | `streamlit run app.py` again — caches rebuild in ~2 s |
| Total failure | Fall back to the notebook. The Act 8 figures are already in it; you lose the interactivity, not the argument. |

**Never** let a broken demo eat more than 20 seconds. Say *"I'll come back to that"*, switch to the notebook, keep moving.

---

## Timing Card

| # | Beat | Time | Cut to 3 min? |
|---|---|---|---|
| 1 | Homography side by side | 0:45 | gesture only |
| 2 | One bay, nine stages | 1:15 | keep |
| 3 | **Break it: erosion, morphology, blurs** | 2:00 | **erosion only** |
| 4 | Weather → sunny | 0:30 | cut |
| 5 | **The twist, live** | 1:30 | **never cut** |
| | **Total** | **~6:00** | **~3:00** |
