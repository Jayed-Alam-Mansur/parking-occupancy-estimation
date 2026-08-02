# 🎤 Presentation Script — Automatic Parking Occupancy Estimation

**Deliverable presented:** `notebooks/00_COMPLETE_PROJECT.ipynb`
(or `docs/presentation/parking-occupancy-presentation.html` — safer, see *Before You Start*)

**Target length:** ~12 minutes + questions. Cut markers are given for a 7-minute version.

---

## Before You Start

| ✅ | Check |
|---|---|
| ☐ | Open the **HTML** file, not the notebook. No kernel, no server, and no way to wipe an output by fat-fingering `Shift+Enter` mid-talk. |
| ☐ | Scroll once to the bottom, then back to the top. Forces every image to load before you're live. |
| ☐ | Browser zoom to **80%** — the figures are wide and the back row needs the whole thing on screen. |
| ☐ | Have the `.ipynb` open in a second tab, in case someone asks to see code running. |
| ☐ | Know your three jump points cold: **Act 2** (the money shot), **Act 6** (Fisher ranking), **Act 8** (the twist). |

**The single most important delivery note:** the twist in Act 8 is your strongest material.
Do not rush to reach it and do not apologise when you get there. Slow down instead.

---

## 0 · Opening — the hook (60 seconds)

> *"Everyone here has done this. You drive into a car park, it looks completely
> full, you circle it twice — and then you find a space that was free the whole
> time, just hidden behind a van.*
>
> *The way the industry solves this is one sensor per bay. A hundred spaces
> means a hundred sensors to trench into concrete, wire, power, network and
> maintain. It works. It costs a fortune.*
>
> *My project does the same job with one camera and some geometry."*

Then state the constraint immediately — it reframes everything that follows:

> *"And I gave myself one hard rule: **no machine learning at all.** No YOLO, no
> PyTorch, no pretrained anything. It's written into my requirements file as a
> prohibition list.*
>
> *Because there's a version of this project that's twenty lines long: import
> YOLO, load weights, draw boxes. It works, and it teaches you nothing about
> images. I wanted every single decision in this system to be one I could
> defend."*

**Objective, in one sentence:**

> *"Given one surveillance frame, answer three things: how many spaces exist,
> which specific ones are occupied, and how full the lot is overall."*

---

## 1 · Act 1 — Know Your Data (60 s) · *scroll to the sample grid*

> *"Before writing a single filter, I looked at the data. This is PKLot — about
> 12,400 images from three car parks, in sunny, cloudy and rainy weather, and
> critically, every bay is hand-labelled occupied or vacant. That's what lets me
> prove numbers later instead of claiming them."*

Point at the sunny frame:

> *"And here's the observation that drove half my design decisions. Look at
> these shadows. They are dark, they are car-shaped, and they are not cars.
> Everything in Acts 4 and 5 exists because of this one picture."*

✂️ *7-min version: keep this — it sets up the shadow problem you pay off twice.*

---

## 2 · Act 2 — The Camera Is Lying (2 min) · **THE MONEY SHOT**

This is your strongest visual. Give it room.

> *"Look at the raw frame. Bays near the camera are big rectangles. Bays at the
> far end are tiny slivers. They're the same physical size — that's perspective
> projection distorting them.*
>
> *This is a serious problem. If I measure 'how many edges are in this bay',
> a far bay scores lower than a near bay for reasons that have nothing to do
> with whether a car is parked there. Every feature I build would secretly be
> measuring **distance from the camera**."*

Now the fix:

> *"The car park surface is flat. And any two views of a plane are related by a
> **homography** — a single 3-by-3 matrix. I pick four corresponding points,
> solve for H, and I can re-photograph the entire lot from an imaginary camera
> hovering directly overhead."*

**Scroll to the before/after.** Pause. Let them look.

> *"Trapezoids become rectangles. Every bay is now roughly the same size and
> shape. This one step is what makes every measurement after it fair."*

---

## 3 · Act 3 — Carving 100 Bays (45 s)

> *"Now I stop thinking about 'an image' and start thinking about a hundred
> small independent problems — one per bay. I push the annotation polygons
> through the same matrix H so they land correctly on the overhead view.*
>
> *But there's a trap. Painted lines are shared between neighbouring bays, so
> the polygons overlap slightly. A big car in bay 13 spills pixels into bay 12,
> and bay 12 reads as occupied when it's empty.*
>
> *The fix is deliberately boring: I **shrink every mask inward** and only
> measure the confident core. I lose a little signal and I buy real
> independence. Worth it."*

✂️ *7-min version: compress to two sentences — overlap problem, erode inward.*

---

## 4 · Act 4 — Noon vs Dusk (60 s)

> *"A bay in direct sun and the same empty bay under cloud produce completely
> different pixel values. Tune a threshold for one and it fails on the other.
> And it's worse than that — a single sunny frame has a bright half and a shaded
> half, so even one global correction isn't enough.*
>
> *So every bay climbs a four-step ladder: grayscale, then **CLAHE**, then
> Gaussian blur, then a median filter.*
>
> *CLAHE is the important one — it equalises contrast **per tile** rather than
> globally, so the sunlit side and the shadowed side get fixed independently."*

If asked why median comes after Gaussian — this is a likely question, have it ready:

> *"Because impulse noise is a rank-order problem, not a convolution problem.
> You can't average away a single wildly-wrong pixel. You can rank it out."*

---

## 5 · Act 5 — Grey to Black-and-White (75 s)

> *"Now the actual decision: which pixels are car and which are asphalt?*
>
> *I tried three classical methods and none of them is right on its own.
> **Global** thresholding is fast and dies the moment lighting varies. **Otsu**
> picks its own cutoff by maximising between-class variance — elegant, but it
> assumes the histogram has two clear humps. **Adaptive** handles gradients
> beautifully and cheerfully hallucinates texture in a completely empty bay.*
>
> *So I fuse them instead of picking a winner."*

Then pay off the shadow setup from Act 1:

> *"And remember those shadows? A hard shadow is dark, car-shaped, and defeats
> every intensity-based method I just described. So I stop using intensity. I
> convert to **HSV** — where a shadow is a region that lost its *value* but kept
> its *hue*. Asphalt in shadow is still asphalt-coloured. That's the escape."*

Point at the morphology stages:

> *"Then morphology cleans up: opening kills the speckle, closing seals the
> holes inside the car body. Watch a scattered mess resolve into one solid
> car-shaped blob."*

✂️ *7-min version: cut the three-method comparison, keep HSV shadows + morphology.*

---

## 6 · Act 6 — Eight Numbers (90 s) · *set up the twist here*

> *"Here's the compression step that makes this tractable. Each bay is thousands
> of pixels. I reduce it to **eight numbers**, and I chose each one so it has a
> physical story I can tell.*
>
> *Edge density — a car is a box of hard edges, empty asphalt is smooth.
> Foreground ratio, gradient magnitude, local variance — empty tarmac is
> uniform, bodywork isn't. Largest connected component — one big blob is a car,
> scattered specks are noise. And mean saturation — painted cars are colourful,
> asphalt is grey."*

Now the part that makes you look rigorous:

> *"And then the honest question — **are these eight actually any good?** I
> didn't guess. I computed the **Fisher discriminant ratio** for each one, which
> measures how far apart the occupied and vacant distributions sit relative to
> how much they spread."*

**Point directly at the ranking. This is the setup — plant it deliberately:**

> *"Edge density wins by a wide margin. Remember that. It comes back in a way I
> genuinely did not expect."*

---

## 7 · Act 7 — Drawing the Line (60 s)

> *"Eight numbers in, one yes-or-no answer out. This is where most projects
> quietly reach for a classifier. I don't — on purpose, because a trained model
> is a black box and the whole premise here is that everything is explainable.*
>
> *So: a **two-stage cascade**. If edge density is very low, the bay is
> obviously empty — decide it immediately and move on. Very high, obviously
> occupied. About a third of bays get resolved on that fast path, cheaply.*
>
> *Everything genuinely ambiguous goes to a weighted vote — and each feature's
> weight is its Fisher ratio from the previous act. Features that proved
> themselves get more say.*
>
> *And I didn't guess the final cutoff either — I swept it across its whole
> range and read the optimum off the curve. Notice the peak is broad and flat,
> not a spike. The system isn't balanced on a knife-edge."*

---

## 8 · Act 8 — The Verdict, and the Twist (3 min) · **YOUR BEST MATERIAL**

First the honest headline:

> *"Three hand-picked frames prove nothing. So: 120 frames across all three
> weather conditions — **11,599 individual bay judgements**, every one checked
> against ground truth.*
>
> ***74.3 % accuracy. 0.731 F1. 77 milliseconds a frame — about 13 FPS on a
> laptop CPU with no GPU.***
>
> *Rainy is the weak case, and the reason is almost funny: wet asphalt reflects,
> and reflections have edges."*

Now stop. Change your tone. This is the moment.

> *"And then I ran one more comparison — and I want to show you this one because
> I could easily have deleted it.*
>
> *I tested my whole eight-feature weighted cascade against the dumbest possible
> baseline: **threshold edge density. One number. No weights, no scoring, no
> cascade.***

**Pause.**

> ***The one-number baseline beat my entire system.***
>
> *So I went and worked out why, and I think there are three reasons.*
>
> *First — I fitted my Fisher weights on a small clean sample and then applied
> them to a large messy one. Weights tuned on easy data don't transfer.*
>
> *Second — my seven weaker features aren't independent, they're correlated with
> each other. Averaging seven noisy correlated votes doesn't cancel the error.
> It just **dilutes the one feature that actually worked**.*
>
> *Third — every feature I added brought its own failure mode with it.
> Saturation breaks on grey cars. Local variance breaks on wet tarmac. The
> cascade inherits all of them at once.*
>
> *The lesson is the one nobody enjoys learning: **more features is not more
> signal.** And when a complicated system loses to its own simplest ingredient,
> that's the system telling you something. The professional move is to listen to
> it, not to hide the comparison."*

✂️ *7-min version: **never cut this.** Cut Acts 3 and 5 first.*

---

## 9 · Act 9 — What It Means (75 s)

**What we built:**

> *"One camera instead of a hundred sensors. No training data. No GPU.
> Real-time on a laptop. And every decision traceable to a named pixel
> statistic — point at any wrong answer and I can tell you exactly which feature
> caused it. Almost no learned system can offer you that."*

**What we got:**

| | |
|---|---|
| Accuracy | **74.30 %** on 11,599 held-out samples |
| F1 | **0.7310** |
| Speed | **77.33 ms/frame — 12.9 FPS**, CPU only |
| Best condition | Cloudy — flat, even lighting |
| Worst condition | Rainy — reflections read as edges |

**What I'd do next** — order matters here, it shows judgement:

> *"First, ship the baseline. Edge density alone is more accurate **and**
> faster. The honest engineering decision is to keep the simple thing.*
>
> *Second, re-fit the weights on the full 11,599 samples instead of the small
> clean set — and find out whether my cascade was mis-weighted or actually
> misconceived.*
>
> *Third, drop the features that measurably fail, instead of keeping all eight
> for symmetry.*
>
> *And fourth — temporal smoothing. Cars don't teleport. A bay that reads
> occupied for one frame out of thirty is a glitch, and a running vote across
> frames would erase a whole class of error for almost no compute."*

**Closing line:**

> *"One camera, no machine learning, nothing that can't be explained on a
> whiteboard — and the most useful thing I got out of it was being proven wrong
> by my own baseline. Thank you."*

---

## Likely Questions — prepared answers

**"Why not just use YOLO? It'd be far more accurate."**
> *"It would, and I'd use it in production. But this is a Digital Image
> Processing course — the constraint was the assignment. And building it from
> primitives forced me to justify every stage. I know exactly why CLAHE and not
> global equalisation, and why median after Gaussian. Twenty lines of YOLO
> teaches you none of that."*

**"74 % isn't very good, is it?"**
> *"No, and I've been explicit about that — it's a working demonstration, not a
> product. What I'd point to is that I measured it honestly on 11,599 samples
> rather than showing you three frames that happened to work, and that I
> published the comparison where my own baseline beat me."*

**"How well does it generalise to a different car park?"**
> *"Poorly, without recalibration — and that's a real limitation. The homography
> is hand-calibrated per camera, so a new site needs four new correspondence
> points. The upside is that recalibration is a config file, not a site visit —
> no hardware moves."*

**"Why eight features if only one turned out to matter?"**
> *"I didn't know that in advance. That's the honest answer. I designed eight
> that each had a physical justification, then measured them. Act 8 is where I
> found out. If I did it again I'd start with the Fisher ranking and add
> features only where they beat what I already had."*

**"How did you pick the four correspondence points?"**
> *"Corners of the outermost painted bay markings — they're coplanar with the
> ground, they're visible in every frame, and they're far apart, which keeps
> the homography well-conditioned."*

**"Is the ground truth reliable?"**
> *"It's PKLot's published per-bay annotation, hand-labelled by the original
> researchers at UFPR. It's a standard benchmark, which is exactly why I chose
> it over labelling frames myself."*

---

## Timing Card — tear this off

| Act | Topic | Time | Cut at 7 min? |
|---|---|---|---|
| 0 | Hook + the no-ML rule | 1:00 | keep |
| 1 | The data, the shadows | 1:00 | keep |
| 2 | Homography — money shot | 2:00 | keep |
| 3 | 100 bays, eroded masks | 0:45 | **cut** |
| 4 | CLAHE ladder | 1:00 | trim to 30 s |
| 5 | Thresholding + HSV shadows | 1:15 | **cut to HSV only** |
| 6 | Eight features + Fisher | 1:30 | trim to 45 s |
| 7 | The cascade + sweep | 1:00 | trim to 30 s |
| 8 | **Results + the twist** | 3:00 | **never cut** |
| 9 | Conclusions + next steps | 1:15 | trim to 45 s |
| | **Total** | **~13:45** | **~7:00** |
