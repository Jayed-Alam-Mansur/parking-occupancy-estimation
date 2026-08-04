"""
Interactive explorer for the classical parking-occupancy pipeline.

Run with:  streamlit run app.py

Every stage below calls the same functions in src/ that the notebooks call —
nothing is reimplemented here. The point of this app is that the parameters
are live: move a slider and watch the classical pipeline react.
"""
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.decide import classify_slot, load_thresholds, weighted_score
from src.features import extract_all_features
from src.geometry import (compute_homography, load_homography,
                          transform_points, warp_perspective)
from src.io_utils import parse_pklot_xml, quality_gate
from src.morphology import clean_binary_mask
from src.preprocessing import (apply_clahe, apply_gaussian_blur,
                               apply_median_blur, to_grayscale)
from src.roi import (create_eroded_core_mask, extract_slot_image,
                     load_slots_json)
from src.segmentation import adaptive_threshold, fuse_channels, otsu_threshold
from src.stats import compute_statistics
from src.visualize import annotate_parking_image, create_legend

st.set_page_config(page_title="Parking Occupancy — Classical CV",
                   page_icon="P", layout="wide")

FEATURES = ['edge_density', 'foreground_ratio', 'gradient_magnitude',
            'local_variance', 'largest_component', 'intensity_std',
            'otsu_separability', 'mean_saturation']


# Asset loading

@st.cache_resource
def load_assets():
    h = load_homography('config/homography.npz')
    slots = load_slots_json('config/slots.json')
    thresholds, weights = load_thresholds('config/thresholds.yaml')
    return (h['H'], h['output_size'], slots, thresholds, weights,
            np.asarray(h['src_points'], np.float32),
            np.asarray(h['dst_points'], np.float32))


@st.cache_data
def list_samples():
    out = []
    for jpg in sorted(glob.glob('data/samples/*.jpg')):
        xml = jpg[:-4] + '.xml'
        if os.path.exists(xml):
            name = os.path.basename(jpg)[:-4]
            out.append({'weather': name.split('_')[0], 'name': name,
                        'jpg': jpg, 'xml': xml})
    return out


def rgb(img):
    """BGR (OpenCV) → RGB (Streamlit)."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Pipeline functions

def slot_stages(bev, polygon, p):
    """Run one slot through every stage, keeping each intermediate."""
    slot_img, _, mask = extract_slot_image(bev, polygon)
    if slot_img.size == 0:
        return None

    core = create_eroded_core_mask(mask, erosion_px=p['erosion'])

    gray = to_grayscale(slot_img)
    clahe = apply_clahe(gray, clip_limit=p['clip'], grid_size=(p['grid'], p['grid']))
    blur = apply_gaussian_blur(clahe, kernel_size=(p['gauss'], p['gauss']))
    den = apply_median_blur(blur, kernel_size=p['median'])

    otsu, _, _ = otsu_threshold(den)
    adapt = adaptive_threshold(den, block_size=p['block'], constant=p['C'])
    fused = fuse_channels(otsu, adapt)
    cleaned = clean_binary_mask(fused,
                                open_ksize=(p['open'], p['open']),
                                close_ksize=(p['close'], p['close']))

    feats = extract_all_features(den, cleaned, bgr_image=slot_img, mask=core,
                                 canny_low=p['canny_lo'], canny_high=p['canny_hi'])

    return {'slot': slot_img, 'gray': gray, 'clahe': clahe, 'blur': blur,
            'denoised': den, 'otsu': otsu, 'adaptive': adapt, 'fused': fused,
            'cleaned': cleaned, 'core': core, 'features': feats}


@st.cache_data(show_spinner=False)
def process_frame(jpg, xml, p, thr):
    """Full 100-bay pass. Cached on (frame, params, thresholds)."""
    H, size, slots, _, weights, _, _ = load_assets()
    img = cv2.imread(jpg)
    t0 = time.perf_counter()
    bev = warp_perspective(img, H, size)

    feats, labels, confs = {}, {}, {}
    for sid, poly in slots.items():
        s = slot_stages(bev, poly, p)
        if s is None:
            continue
        feats[sid] = s['features']
        lab, conf, _ = classify_slot(s['features'], dict(thr), weights)
        labels[sid], confs[sid] = lab, conf
    elapsed = (time.perf_counter() - t0) * 1000

    gt = {s['id']: s['occupied'] for s in parse_pklot_xml(xml)}
    return {'img': img, 'bev': bev, 'features': feats, 'labels': labels,
            'confidences': confs, 'gt': gt, 'ms': elapsed}


CALIB_W, CALIB_H = 1280, 720          # resolution the homography was solved on


def rescale_homography(iw, ih):
    """Adapt the committed calibration to a different resolution.

    src_points are absolute pixel coordinates, so the same camera at a
    different resolution needs them scaled — otherwise they address the wrong
    scene points and every bay lands slightly off. Only valid when the aspect
    ratio matches; a different aspect ratio means a genuinely different view.
    """
    same_aspect = abs(iw / ih - CALIB_W / CALIB_H) < 0.02
    if not same_aspect:
        return None, False
    scaled = SRC_PTS * np.array([iw / CALIB_W, ih / CALIB_H], np.float32)
    Hn = compute_homography(scaled, DST_PTS)
    return (Hn[0] if isinstance(Hn, tuple) else Hn), True


def decode_upload(raw):
    """Uploaded bytes → BGR image."""
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def slots_from_grid(size, rows, cols, margin_x, margin_y, gap):
    """Synthesise a regular bay layout in BEV coordinates.

    For a lot this pipeline has never been calibrated against, a regular grid is
    the honest fallback: no annotation file exists, so bays are assumed evenly
    spaced. Verify the overlay visually before trusting any number it produces.
    """
    w, h = size
    usable_w = w - 2 * margin_x
    usable_h = h - 2 * margin_y
    bay_w = (usable_w - gap * (cols - 1)) / cols
    bay_h = (usable_h - gap * (rows - 1)) / rows
    slots, sid = {}, 1
    for r in range(rows):
        for c in range(cols):
            x0 = margin_x + c * (bay_w + gap)
            y0 = margin_y + r * (bay_h + gap)
            slots[sid] = np.array([[x0, y0], [x0 + bay_w, y0],
                                   [x0 + bay_w, y0 + bay_h], [x0, y0 + bay_h]],
                                  dtype=np.float32)
            sid += 1
    return slots


def slots_from_xml_bytes(raw, H):
    """PKLot-format XML → BEV bay polygons + ground-truth labels."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as fh:
        fh.write(raw)
        path = fh.name
    try:
        parsed = parse_pklot_xml(path)
    finally:
        os.unlink(path)
    slots = {s['id']: transform_points(s['points'], H) for s in parsed}
    gt = {s['id']: s['occupied'] for s in parsed}
    return slots, gt


def alignment_report(bev, slots, min_coverage=0.5):
    """How many bays actually sit on real image content?

    This guards against a silently wrong result. Checking that a polygon falls
    inside the canvas is NOT enough — `warp_perspective` always returns the full
    BEV canvas, so a mismatched homography still puts every bay "in bounds",
    just over black padding where no source pixel mapped. So measure the
    fraction of each bay that is actually non-black, and treat bays sitting on
    emptiness as unusable.
    """
    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY) if bev.ndim == 3 else bev
    usable = 0
    for poly in slots.values():
        crop, _, mask = extract_slot_image(gray, poly)
        if not crop.size or min(crop.shape[:2]) < 4:
            continue
        inside = mask > 0
        if not inside.any():
            continue
        if (crop[inside] > 0).mean() >= min_coverage:
            usable += 1
    return usable, len(slots)


def run_custom(bev, slots, p, thr, weights):
    """Full pass over an arbitrary bay layout."""
    feats, labels, confs = {}, {}, {}
    for sid, poly in slots.items():
        s = slot_stages(bev, poly, p)
        if s is None or min(s['slot'].shape[:2]) < 4:
            continue
        feats[sid] = s['features']
        lab, conf, _ = classify_slot(s['features'], dict(thr), weights)
        labels[sid], confs[sid] = lab, conf
    return feats, labels, confs


def score_against_gt(labels, gt):
    ids = [i for i in labels if i in gt]
    tp = sum(1 for i in ids if labels[i] == 1 and gt[i] == 1)
    tn = sum(1 for i in ids if labels[i] == 0 and gt[i] == 0)
    fp = sum(1 for i in ids if labels[i] == 1 and gt[i] == 0)
    fn = sum(1 for i in ids if labels[i] == 0 and gt[i] == 1)
    n = max(len(ids), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {'n': len(ids), 'acc': (tp + tn) / n, 'tp': tp, 'tn': tn,
            'fp': fp, 'fn': fn,
            'f1': 2 * prec * rec / max(prec + rec, 1e-9)}


@st.cache_data(show_spinner="Extracting features across all sample frames…")
def corpus_features(p, thr):
    """Every slot of every sample frame — the evidence base for Act 8."""
    rows = []
    for s in list_samples():
        r = process_frame(s['jpg'], s['xml'], p, thr)
        for sid, f in r['features'].items():
            if sid in r['gt']:
                rows.append({'weather': s['weather'], 'frame': s['name'],
                             'slot': sid, 'truth': r['gt'][sid],
                             'cascade': r['labels'][sid],
                             **{k: f[k] for k in FEATURES}})
    return pd.DataFrame(rows)


# Sidebar controls

H, SIZE, SLOTS, TUNED, WEIGHTS, SRC_PTS, DST_PTS = load_assets()
samples = list_samples()

st.sidebar.title("Controls")

weathers = sorted({s['weather'] for s in samples})
w = st.sidebar.selectbox(
    "Weather", weathers,
    index=weathers.index('cloudy') if 'cloudy' in weathers else 0,
    help="Cloudy is the easy case — flat, even light. Switch to sunny for the "
         "shadow problem and rainy for reflections; both are where it struggles.")
opts = [s for s in samples if s['weather'] == w]


@st.cache_data
def gt_occupancy(xml):
    """Ground-truth occupancy rate — cheap, XML only, no image processing."""
    slots = parse_pklot_xml(xml)
    return sum(s['occupied'] for s in slots) / max(len(slots), 1)


# Default to the most balanced frame. A 6 a.m. empty lot scores 100 % and an
# F1 of zero, which reads as broken rather than impressive.
balance = [abs(gt_occupancy(o['xml']) - 0.5) for o in opts]
default_i = int(np.argmin(balance)) if opts else 0

frame = opts[st.sidebar.selectbox(
    "Frame", range(len(opts)), index=default_i,
    format_func=lambda i: f"{opts[i]['name'].split('_', 1)[1]} "
                          f"· {gt_occupancy(opts[i]['xml']) * 100:.0f}% full")]

st.sidebar.markdown("---")
st.sidebar.caption("**Preprocessing** — Act 4")
clip = st.sidebar.slider("CLAHE clip limit", 0.5, 8.0, 2.0, 0.5,
                         help="Caps contrast amplification. Raise it too far and noise is amplified into fake edges.")
grid = st.sidebar.select_slider("CLAHE tile grid", [4, 8, 16], value=8)
gauss = st.sidebar.select_slider("Gaussian kernel", [1, 3, 5, 7, 9], value=5)
median = st.sidebar.select_slider("Median kernel", [1, 3, 5, 7], value=3)

st.sidebar.caption("**Segmentation** — Act 5")
block = st.sidebar.select_slider("Adaptive block size", [7, 11, 15, 21, 31], value=11)
C = st.sidebar.slider("Adaptive constant C", -10, 15, 2)
open_k = st.sidebar.select_slider("Morph. opening", [1, 3, 5, 7], value=3)
close_k = st.sidebar.select_slider("Morph. closing", [1, 3, 5, 7], value=5)
erosion = st.sidebar.slider("Core-mask erosion (px)", 0, 10, 3,
                            help="Act 3: shrink each bay inward so a neighbour's car cannot leak in.")

st.sidebar.caption("**Features** — Act 6")
canny_lo = st.sidebar.slider("Canny low", 10, 150, 50)
canny_hi = st.sidebar.slider("Canny high", 60, 300, 150)

st.sidebar.caption("**Decision** — Act 7")
score_t = st.sidebar.slider("Score threshold", 0.0, 1.0,
                            float(TUNED['score_threshold']), 0.01)
ed_lo = st.sidebar.slider("Fast-path VACANT below", 0.0, 0.6,
                          float(TUNED['edge_density_low']), 0.005)
ed_hi = st.sidebar.slider("Fast-path OCCUPIED above", 0.0, 0.8,
                          float(TUNED['edge_density_high']), 0.005)

P = {'clip': clip, 'grid': grid, 'gauss': gauss, 'median': median,
     'block': block, 'C': C, 'open': open_k, 'close': close_k,
     'erosion': erosion, 'canny_lo': canny_lo, 'canny_hi': canny_hi}
THR = (('edge_density_low', ed_lo), ('edge_density_high', ed_hi),
       ('score_threshold', score_t),
       ('confidence_low', float(TUNED['confidence_low'])))

if st.sidebar.button("↺ Reset to tuned values", width='stretch'):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("No ML anywhere in this app. Every stage is an OpenCV "
                   "primitive called from `src/`.")


# Main content

st.title("Parking Occupancy Estimation — Classical Image Processing")
st.caption("One camera. No sensors. No machine learning. "
           "Move any slider and the whole pipeline recomputes live.")

tab1, tab2, tab5, tab3, tab4 = st.tabs([
    "Pipeline Explorer", "Whole Lot", "Your Own Image",
    "The Twist (Act 8)", "The Story"])


# Tab 1 — stage by stage on one bay
with tab1:
    res = process_frame(frame['jpg'], frame['xml'], P, THR)
    gt = res['gt']

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Act 2 — the camera's view")
        st.image(rgb(res['img']), width='stretch')
        st.caption("Near bays are large, far bays are slivers. Same physical size.")
    with right:
        st.subheader("…and after the homography")
        st.image(rgb(res['bev']), width='stretch')
        st.caption("Every bay now roughly the same size — measurements become comparable.")

    st.markdown("---")
    st.subheader("Act 4–6 — follow one bay through every stage")

    ids = sorted(SLOTS.keys())
    occ = [i for i in ids if gt.get(i) == 1]
    vac = [i for i in ids if gt.get(i) == 0]
    c1, c2 = st.columns([1, 3])
    with c1:
        kind = st.radio("Pick a bay that is truly…",
                        ["OCCUPIED", "VACANT"], horizontal=True)
        pool = occ if kind == "OCCUPIED" else vac
        sid = st.selectbox("Bay #", pool) if pool else None

    if sid is not None:
        s = slot_stages(res['bev'], SLOTS[sid], P)
        ladder = [("1 · ROI", s['slot']), ("2 · Grayscale", s['gray']),
                  ("3 · CLAHE", s['clahe']), ("4 · Gaussian", s['blur']),
                  ("5 · Median", s['denoised'])]
        seg = [("6 · Otsu", s['otsu']), ("7 · Adaptive", s['adaptive']),
               ("8 · Fused", s['fused']), ("9 · Morphology", s['cleaned'])]

        st.markdown("**The preprocessing ladder**")
        for col, (t, im) in zip(st.columns(len(ladder)), ladder):
            col.image(rgb(im), caption=t, width='stretch')
        st.markdown("**Segmentation and cleanup**")
        for col, (t, im) in zip(st.columns(len(seg)), seg):
            col.image(im, caption=t, width='stretch')

        f = s['features']
        lab, conf, sc = classify_slot(f, dict(THR), WEIGHTS)
        truth = gt.get(sid)
        ok = (lab == truth)
        m1, m2, m3 = st.columns(3)
        m1.metric("Verdict", "OCCUPIED" if lab else "VACANT",
                  "correct ✓" if ok else "WRONG ✗", delta_color="normal" if ok else "inverse")
        m2.metric("Weighted score", f"{sc:.3f}", f"threshold {score_t:.3f}")
        m3.metric("Confidence", f"{conf:.2f}")

        st.markdown("**The eight numbers this bay reduces to**")
        fd = pd.DataFrame({'feature': FEATURES,
                           'value': [f[k] for k in FEATURES],
                           'weight': [WEIGHTS[k] for k in FEATURES]})
        fd['contribution'] = fd.value * fd.weight
        st.dataframe(fd.style.format({'value': '{:.4f}', 'weight': '{:.4f}',
                                      'contribution': '{:.4f}'})
                     .bar(subset=['contribution'], color='#4c78a8'),
                     width='stretch', hide_index=True)


# Tab 2 — the whole lot
with tab2:
    res = process_frame(frame['jpg'], frame['xml'], P, THR)
    m = score_against_gt(res['labels'], res['gt'])
    stats = compute_statistics(res['labels'])

    k = st.columns(5)
    k[0].metric("Occupancy", f"{stats['occupancy_pct']:.0f}%")
    k[1].metric("Occupied", stats['occupied'])
    k[2].metric("Vacant", stats['vacant'])
    k[3].metric("Accuracy vs truth", f"{m['acc'] * 100:.1f}%")
    k[4].metric("Frame time", f"{res['ms']:.0f} ms")

    annotated = annotate_parking_image(res['bev'], SLOTS, res['labels'],
                                       confidences=res['confidences'])
    st.image(rgb(annotated), width='stretch',
             caption="Green = predicted VACANT · Red = predicted OCCUPIED")

    st.markdown("**Where it went wrong on this frame**")
    e1, e2 = st.columns(2)
    e1.metric("False OCCUPIED", m['fp'], help="Called occupied, actually empty — sends drivers away from a free space.")
    e2.metric("False VACANT", m['fn'], help="Called empty, actually occupied — sends drivers to a taken space. The worse error.")
    f1txt = f"{m['f1']:.3f}" if (m['tp'] + m['fn']) else "n/a (no occupied bays)"
    st.caption(f"Confusion on {m['n']} bays — TP {m['tp']} · TN {m['tn']} · "
               f"FP {m['fp']} · FN {m['fn']} · F1 {f1txt}")


# Tab 5 — bring your own image
with tab5:
    st.subheader("Run the pipeline on your own image")
    st.markdown(
        "Upload any parking-lot photo and push it through the identical "
        "pipeline. Two things have to be true before the numbers mean "
        "anything: the **perspective** must be corrected for *your* camera, "
        "and the system must know **where the bays are**. Both are set below.")

    up = st.file_uploader("Parking-lot image", type=['jpg', 'jpeg', 'png'],
                          key='own_img')

    if up is None:
        st.info("Waiting for an image. If you just want to see it work, any "
                "frame from `data/samples/` is a same-camera image and will "
                "align perfectly with the committed calibration.")
    else:
        img = decode_upload(up.getvalue())
        if img is None:
            st.error("Could not decode that file as an image.")
            st.stop()

        ih, iw = img.shape[:2]

        # ---- Act 1's quality gate, applied to user input ----
        passes, diag = quality_gate(img)
        q1, q2, q3 = st.columns(3)
        q1.metric("Resolution", f"{iw} × {ih}")
        q2.metric("Brightness", f"{diag['brightness']:.0f}",
                  "too dark" if diag['too_dark'] else "ok",
                  delta_color="inverse" if diag['too_dark'] else "normal")
        q3.metric("Sharpness (Laplacian var)", f"{diag['blur_score']:.0f}",
                  "too blurry" if diag['too_blurry'] else "ok",
                  delta_color="inverse" if diag['too_blurry'] else "normal")
        if not passes:
            st.warning("This image fails the Act 1 quality gate. The pipeline "
                       "will still run, but treat the output as unreliable.")

        st.markdown("---")
        c_geo, c_lay = st.columns(2)

        # ---- geometry ----
        with c_geo:
            st.markdown("**1 · Perspective correction**")
            geo = st.radio(
                "Homography", ["Committed calibration", "Calibrate for this image"],
                key='geo_mode', label_visibility='collapsed',
                captions=["Correct only if this is the same camera as PKLot "
                          "`parking2` (1280×720).",
                          "Pick the four ground-plane corners yourself."])

            if geo == "Committed calibration":
                size_u = SIZE
                if (iw, ih) == (CALIB_W, CALIB_H):
                    Hu = H
                    st.success(f"Exact match for the calibration resolution "
                               f"({CALIB_W}×{CALIB_H}).")
                else:
                    Hs, ok_aspect = rescale_homography(iw, ih)
                    if ok_aspect:
                        Hu = Hs
                        st.info(f"This image is {iw}×{ih}, not the "
                                f"{CALIB_W}×{CALIB_H} the homography was solved "
                                f"on — but the aspect ratio matches, so the "
                                f"calibration points were **rescaled by "
                                f"{iw / CALIB_W:.2f}×**. Valid if this is the "
                                f"same camera at a different resolution.")
                    else:
                        Hu = H
                        st.error(f"This image is {iw}×{ih}, a different aspect "
                                 f"ratio from the calibration "
                                 f"({CALIB_W}×{CALIB_H}). The committed "
                                 f"homography cannot be adapted to it — "
                                 f"switch to *Calibrate for this image*.")
            else:
                st.caption("Corners as fractions of width/height, clockwise "
                           "from top-left. Defaults are the committed values.")
                defaults = [(-0.011, 0.216), (0.943, 0.216),
                            (0.943, 0.819), (-0.011, 0.819)]
                names = ["top-left", "top-right", "bottom-right", "bottom-left"]
                pts = []
                for nm, (dx, dy) in zip(names, defaults):
                    a, b = st.columns(2)
                    fx = a.number_input(f"{nm} x", -0.3, 1.3, dx, 0.005,
                                        key=f'fx_{nm}', format="%.3f")
                    fy = b.number_input(f"{nm} y", -0.3, 1.3, dy, 0.005,
                                        key=f'fy_{nm}', format="%.3f")
                    pts.append([fx * iw, fy * ih])
                src = np.array(pts, dtype=np.float32)
                bw = st.select_slider("BEV width", [600, 800, 1000], value=800,
                                      key='bev_w')
                bh = st.select_slider("BEV height", [800, 1000, 1200], value=1000,
                                      key='bev_h')
                m = 30
                dst = np.array([[m, m], [bw - m, m], [bw - m, bh - m], [m, bh - m]],
                               dtype=np.float32)
                Hu = compute_homography(src, dst)
                if isinstance(Hu, tuple):
                    Hu = Hu[0]
                size_u = (bw, bh)

        bev_u = warp_perspective(img, Hu, size_u)

        # ---- bay layout ----
        with c_lay:
            st.markdown("**2 · Where are the bays?**")
            lay = st.radio(
                "Layout", ["Committed slots.json (100 bays)",
                           "Upload matching PKLot XML",
                           "Generate a regular grid"],
                key='lay_mode', label_visibility='collapsed')

            gt_u = None
            if lay == "Committed slots.json (100 bays)":
                slots_u = SLOTS
                st.caption("The 100 hand-calibrated bays from Act 3. Only "
                           "valid for the same camera *and* the committed "
                           "homography.")
            elif lay == "Upload matching PKLot XML":
                xup = st.file_uploader("PKLot-format XML", type=['xml'],
                                       key='own_xml')
                if xup is None:
                    st.info("Upload the XML that pairs with this frame. It "
                            "gives both the bay polygons and ground truth, so "
                            "accuracy can be scored.")
                    slots_u = {}
                else:
                    slots_u, gt_u = slots_from_xml_bytes(xup.getvalue(), Hu)
                    st.success(f"{len(slots_u)} bays and ground-truth labels "
                               f"loaded from XML.")
            else:
                g1, g2 = st.columns(2)
                rows = g1.slider("Rows", 1, 12, 4, key='g_rows')
                cols = g2.slider("Bays per row", 2, 40, 20, key='g_cols')
                mx = g1.slider("Margin X", 0, 200, 40, key='g_mx')
                my = g2.slider("Margin Y", 0, 200, 60, key='g_my')
                gap = st.slider("Gap between bays", 0, 20, 2, key='g_gap')
                slots_u = slots_from_grid(size_u, rows, cols, mx, my, gap)
                st.caption(f"{len(slots_u)} synthetic bays. No annotation file "
                           f"exists for an unknown lot, so bays are assumed "
                           f"evenly spaced — check the overlay before trusting "
                           f"the count.")

        if not slots_u:
            st.stop()

        # ---- alignment guard ----
        usable, total = alignment_report(bev_u, slots_u)
        frac = usable / max(total, 1)
        st.markdown("---")
        if frac < 0.8:
            st.error(f"**Alignment check failed — {usable} of {total} bays "
                     f"landed inside the warped view.** The layout does not "
                     f"match this image, so any occupancy number below is "
                     f"meaningless. Recalibrate the corners, or switch to a "
                     f"generated grid.")
        else:
            st.success(f"Alignment check: {usable} of {total} bays landed "
                       f"inside the warped view.")

        # ---- run it ----
        t0 = time.perf_counter()
        feats_u, labels_u, confs_u = run_custom(bev_u, slots_u, P, THR, WEIGHTS)
        ms_u = (time.perf_counter() - t0) * 1000

        if not labels_u:
            st.error("No bay produced a usable crop. Check the calibration.")
            st.stop()

        stats_u = compute_statistics(labels_u)
        k = st.columns(5)
        k[0].metric("Occupancy", f"{stats_u['occupancy_pct']:.0f}%")
        k[1].metric("Occupied", stats_u['occupied'])
        k[2].metric("Vacant", stats_u['vacant'])
        k[3].metric("Bays scored", len(labels_u))
        k[4].metric("Total time", f"{ms_u:.0f} ms")

        if gt_u:
            mu = score_against_gt(labels_u, gt_u)
            f1u = f"{mu['f1']:.3f}" if (mu['tp'] + mu['fn']) else "n/a"
            st.info(f"**Scored against the uploaded ground truth:** "
                    f"{mu['acc'] * 100:.1f}% accuracy · F1 {f1u} · "
                    f"false-occupied {mu['fp']} · false-vacant {mu['fn']} "
                    f"({mu['n']} bays matched)")

        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**Your image**")
            st.image(rgb(img), width='stretch')
        with v2:
            st.markdown("**Warped, with the verdict drawn on**")
            ann_u = annotate_parking_image(bev_u, slots_u, labels_u,
                                           confidences=confs_u)
            ann_u = create_legend(ann_u, stats_u)
            st.image(rgb(ann_u), width='stretch')

        ok, png = cv2.imencode('.png', ann_u)
        if ok:
            st.download_button("Download annotated result",
                               png.tobytes(),
                               file_name=f"occupancy_{up.name.rsplit('.', 1)[0]}.png",
                               mime="image/png")

        with st.expander("Per-bay features and verdicts"):
            st.dataframe(pd.DataFrame([
                {'bay': sid, 'verdict': 'OCCUPIED' if labels_u[sid] else 'VACANT',
                 'confidence': confs_u[sid],
                 **{k_: feats_u[sid][k_] for k_ in FEATURES}}
                for sid in sorted(labels_u)]),
                width='stretch', hide_index=True)


# Tab 3 — the twist
with tab3:
    st.subheader("My eight-feature cascade versus one single number")
    st.markdown(
        "In Act 6 the **Fisher ranking** said `edge_density` was the strongest "
        "feature by a wide margin. So here is the uncomfortable experiment: "
        "throw away the other seven, threshold `edge_density` alone, and see "
        "who wins. **Drag the slider.**")

    df = corpus_features(P, THR)
    if df.empty:
        st.warning("No sample frames found in `data/samples/`.")
    else:
        lo, hi = float(df['edge_density'].min()), float(df['edge_density'].max())
        grid_ = np.linspace(lo, hi, 200)
        accs = [((df['edge_density'] > t).astype(int) == df['truth']).mean() for t in grid_]
        best_cut = float(grid_[int(np.argmax(accs))])

        cut = st.slider("Single-feature cutoff on `edge_density`", lo, hi, best_cut, 0.001,
                        help="Defaults to the swept optimum — the same method Act 7 uses "
                             "to tune the cascade's own threshold.")
        if abs(cut - best_cut) > 1e-9:
            st.caption(f"↩︎ Swept optimum is **{best_cut:.4f}** "
                       f"({max(accs) * 100:.2f}%). Tuned fast-path value is "
                       f"{TUNED['edge_density_high']:.4f}.")

        base = (df['edge_density'] > cut).astype(int)
        acc_base = (base == df['truth']).mean()
        acc_casc = (df['cascade'] == df['truth']).mean()

        def f1(pred):
            tp = ((pred == 1) & (df['truth'] == 1)).sum()
            fp = ((pred == 1) & (df['truth'] == 0)).sum()
            fn = ((pred == 0) & (df['truth'] == 1)).sum()
            p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            return 2 * p * r / max(p + r, 1e-9)

        a, b, c = st.columns(3)
        a.metric("One number: `edge_density`", f"{acc_base * 100:.2f}%", f"F1 {f1(base):.3f}")
        b.metric("Full 8-feature cascade", f"{acc_casc * 100:.2f}%", f"F1 {f1(df['cascade']):.3f}")
        gap = (acc_base - acc_casc) * 100
        c.metric("Gap", f"{gap:+.2f} pts",
                 "baseline wins" if gap > 0 else "cascade wins",
                 delta_color="inverse" if gap > 0 else "normal")

        sweep = pd.DataFrame({
            'cutoff': np.linspace(df['edge_density'].min(), df['edge_density'].max(), 120)})
        sweep['single feature'] = [((df['edge_density'] > t).astype(int) == df['truth']).mean()
                                   for t in sweep.cutoff]
        sweep['8-feature cascade'] = acc_casc
        st.line_chart(sweep.set_index('cutoff'), height=320)
        st.caption(f"Accuracy across {len(df):,} labelled bays from "
                   f"{df['frame'].nunique()} curated sample frames in `data/samples/`. "
                   f"Notebook Act 8 reports 74.30 % for the cascade — measured on a "
                   f"different, larger set (11,599 bays from 120 evenly-spaced frames), "
                   f"which is why the number here differs.")

        st.info(
            "**Why the simple thing wins.** The Fisher weights were fitted on a "
            "small clean sample and then applied to messy data. The seven weaker "
            "features are correlated with each other rather than independent, so "
            "averaging them does not cancel error — it dilutes the one feature "
            "that worked. And every extra feature drags in its own failure mode: "
            "saturation breaks on grey cars, local variance breaks on wet tarmac.")

        st.markdown("**Per-weather breakdown**")
        pw = df.groupby('weather').apply(
            lambda g: pd.Series({
                'bays': len(g),
                'single feature': ((g['edge_density'] > cut).astype(int) == g['truth']).mean(),
                'cascade': (g['cascade'] == g['truth']).mean()}),
            include_groups=False)
        st.dataframe(pw.style.format({'bays': '{:.0f}', 'single feature': '{:.1%}',
                                      'cascade': '{:.1%}'}), width='stretch')


# Tab 4 — the story
with tab4:
    st.markdown("""
### The problem
You circle a full-looking car park twice and eventually find a space that was
free the whole time. The industry fixes this with **one sensor per bay** — a
hundred spaces means a hundred units to trench in, wire, power and maintain.

**This project replaces all hundred with one camera and some geometry.**

### The rule
No TensorFlow, no PyTorch, no YOLO, no pretrained anything. A YOLO model would
solve this in twenty lines and teach nothing about images. Every threshold here
was chosen by a human who can defend it.

| Act | What happens | Why it matters |
|---|---|---|
| **1** | Look at the data first | Sunny shadows are dark and car-shaped. That drives everything after. |
| **2** | Homography → bird's-eye view | Far bays were slivers; now every bay is comparable. |
| **3** | 100 masked bays, eroded inward | Stops a neighbour's car leaking across a shared painted line. |
| **4** | Grayscale → CLAHE → blur → median | Makes noon and dusk comparable. CLAHE works per tile, not globally. |
| **5** | Otsu + adaptive fused, then morphology | HSV escapes the shadow trap: shadows lose *value*, keep *hue*. |
| **6** | Eight features, ranked by Fisher ratio | `edge_density` wins by a wide margin. Remember that. |
| **7** | Two-stage cascade, swept threshold | Fast path resolves ~⅓ of bays; the rest get a weighted vote. |
| **8** | 11,599 bays evaluated — **and the twist** | The one-number baseline beat the whole cascade. |
| **9** | Honest limits and next steps | Ship the baseline. Re-fit weights on full data. Add temporal smoothing. |

### The measured result
| Metric | Value |
|---|---|
| Accuracy | **74.30 %** on 11,599 held-out samples |
| F1 | **0.7310** |
| Speed | **77.33 ms/frame · 12.9 FPS**, CPU only |
| Best / worst | Cloudy (even light) / Rainy (reflections read as edges) |

### What I would do next
1. **Ship the baseline** — edge density alone is more accurate *and* faster.
2. **Re-fit the weights on all 11,599 samples**, not the small clean set.
3. **Drop features that measurably fail** instead of keeping eight for symmetry.
4. **Add temporal smoothing** — cars do not teleport; a one-frame flip is a glitch.
""")
    st.caption("Full write-up: README.md · Notebook: notebooks/00_COMPLETE_PROJECT.ipynb "
               "· Talk track: docs/presentation/PRESENTATION_SCRIPT.md")
