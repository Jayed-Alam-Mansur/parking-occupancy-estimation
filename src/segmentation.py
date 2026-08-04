# segmentation.py - Image Segmentation (Thresholding)
"""
Thresholding methods to convert grayscale images to binary masks.

Functions:
    global_threshold()        - Fixed threshold value
    adaptive_threshold()      - Locally adaptive threshold
    otsu_threshold()          - Otsu's automatic threshold
    reference_difference()    - Difference from empty reference
    shadow_suppress_hsv()     - Remove shadow regions using HSV analysis
    fuse_channels()           - Combine multiple segmentation channels
    compare_thresholds()      - Side-by-side comparison of all methods

Theory:
    Segmentation separates foreground (vehicle) from background (asphalt).

    Three-channel fusion approach:
        A. Per-ROI Otsu threshold — parameter-free, bimodal assumption
           holds well on small slot crops
        B. Adaptive threshold — survives partial shadow with local
           block-wise operation  
        C. Reference differencing — subtracts the known empty-lot
           appearance; the gold standard but needs a reference frame

    HSV shadow suppression:
        Shadows ATTENUATE brightness (V drops) but PRESERVE hue (H stable).
        Actual objects CHANGE BOTH hue and saturation.
        → In HSV: shadow pixels have low V but similar H to background.
        → Mask them out before counting foreground pixels.
"""

import cv2
import numpy as np


def global_threshold(gray_image, thresh_value=127):
    """
    Apply fixed global threshold.

    Parameters
    ----------
    gray_image : np.ndarray
        Single-channel grayscale image.
    thresh_value : int
        Threshold value (0-255).

    Returns
    -------
    binary : np.ndarray
        Binary image (uint8, 0 or 255).
    threshold_used : int
        The threshold value used.

    Notes
    -----
    cv2.threshold(src, thresh, maxval, type)
        - THRESH_BINARY: pixel > thresh → 255, else → 0
        - Simple but brittle: one value for ALL lighting conditions
    """
    _, binary = cv2.threshold(gray_image, thresh_value, 255,
                               cv2.THRESH_BINARY)
    return binary, thresh_value


def adaptive_threshold(gray_image, block_size=11, constant=2,
                       method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C):
    """
    Apply locally adaptive threshold.

    Parameters
    ----------
    gray_image : np.ndarray
        Single-channel grayscale image.
    block_size : int
        Size of local neighbourhood (must be odd, ≥3).
    constant : float
        Constant subtracted from the weighted mean.
    method : int
        cv2.ADAPTIVE_THRESH_MEAN_C or cv2.ADAPTIVE_THRESH_GAUSSIAN_C.

    Returns
    -------
    binary : np.ndarray
        Binary image (uint8).

    Notes
    -----
    For each pixel, the threshold = (weighted mean of block) - constant.

    GAUSSIAN_C uses a Gaussian-weighted sum → centre pixel matters more.
    MEAN_C uses uniform weights.

    Why this survives partial shadow:
    - The local block sees only nearby pixels
    - If a shadow covers half the block, the local mean adapts
    - Global threshold would miscategorise the entire shadow
    """
    binary = cv2.adaptiveThreshold(
        gray_image, 255, method,
        cv2.THRESH_BINARY, block_size, constant
    )
    return binary


def otsu_threshold(gray_image):
    """
    Apply Otsu's automatic threshold.

    Parameters
    ----------
    gray_image : np.ndarray
        Single-channel grayscale image.

    Returns
    -------
    binary : np.ndarray
        Binary image (uint8).
    threshold : float
        The automatically determined threshold.
    separability : float
        Otsu's between-class variance ratio η (0 to 1).
        Higher = better separation = more confident.

    Notes
    -----
    Otsu's method finds threshold T* that maximizes between-class variance:

        σ²_B(T) = ω₀(T) · ω₁(T) · [μ₀(T) - μ₁(T)]²

    where:
        ω₀, ω₁ = class weights (fraction of pixels in each class)
        μ₀, μ₁ = class means

    Separability measure:
        η = σ²_B / σ²_total

    η is a BONUS FEATURE for occupancy estimation:
    - High η (>0.7): clear bimodal distribution → probably a car
    - Low η (<0.3): unimodal → probably empty asphalt

    Applied PER SLOT (not globally) because the bimodality assumption
    holds much better on a small, single-object region.
    """
    threshold, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Compute separability η
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0:
        return binary, threshold, 0.0

    # Normalised histogram
    p = hist / total

    # Total mean
    mu_total = np.sum(np.arange(256) * p)

    # Total variance
    sigma_total_sq = np.sum(((np.arange(256) - mu_total) ** 2) * p)

    if sigma_total_sq == 0:
        return binary, threshold, 0.0

    # Between-class variance at T*
    T = int(threshold)
    w0 = p[:T].sum()
    w1 = p[T:].sum()

    if w0 == 0 or w1 == 0:
        return binary, threshold, 0.0

    mu0 = np.sum(np.arange(T) * p[:T]) / w0
    mu1 = np.sum(np.arange(T, 256) * p[T:]) / w1

    sigma_between_sq = w0 * w1 * (mu0 - mu1) ** 2
    separability = sigma_between_sq / sigma_total_sq

    return binary, threshold, separability


def reference_difference(gray_image, reference_image, diff_threshold=30):
    """
    Compute absolute difference from an empty-lot reference.

    Parameters
    ----------
    gray_image : np.ndarray
        Current frame (grayscale).
    reference_image : np.ndarray
        Reference empty-lot image (grayscale).
    diff_threshold : int
        Minimum difference to count as foreground.

    Returns
    -------
    binary : np.ndarray
        Binary mask of changed pixels.
    diff_image : np.ndarray
        Raw absolute difference image.

    Notes
    -----
    Reference differencing is the gold standard for change detection.

    The reference should be a MEDIAN of multiple empty-lot frames
    (not a single frame) to average out noise, shadows, and small
    temporal variations.

    Limitation: sensitive to global lighting changes (day vs night,
    clouds). CLAHE preprocessing helps mitigate this.
    """
    diff_image = cv2.absdiff(gray_image, reference_image)
    _, binary = cv2.threshold(diff_image, diff_threshold, 255,
                               cv2.THRESH_BINARY)
    return binary, diff_image


def shadow_suppress_hsv(bgr_image, v_low=40, v_high=220,
                         s_threshold=40):
    """
    Create a mask of shadow pixels using HSV colour space analysis.

    Parameters
    ----------
    bgr_image : np.ndarray
        Input BGR colour image.
    v_low : int
        Minimum V (value/brightness). Below this → too dark (deep shadow).
    v_high : int
        Maximum V. Above this → probably not a shadow.
    s_threshold : int
        Maximum S (saturation). Shadows have low saturation.

    Returns
    -------
    shadow_mask : np.ndarray
        Binary mask where 255 = shadow pixel.
    non_shadow_mask : np.ndarray
        Binary mask where 255 = NON-shadow pixel.

    Notes
    -----
    Shadow detection in HSV:
        A shadow pixel has:
            - Reduced V (brightness drops)
            - Similar H (hue preserved — shadow doesn't change colour)
            - Low S (saturation drops because less light)

        An actual object (car) has:
            - Different H (metallic paint, rubber, glass — many hues)
            - Often higher S (saturated car colours)
            - Variable V

    This is the BIGGEST SINGLE ACCURACY WIN on sunny days.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Shadow pixels: low V, low S
    shadow_mask = np.zeros_like(v)
    shadow_condition = (v >= v_low) & (v <= v_high) & (s <= s_threshold)
    shadow_mask[shadow_condition] = 255

    non_shadow_mask = cv2.bitwise_not(shadow_mask)

    return shadow_mask, non_shadow_mask


def fuse_channels(otsu_binary, adaptive_binary, refdiff_binary=None,
                  weights=(0.4, 0.3, 0.3)):
    """
    Fuse multiple segmentation channels into a single binary mask.

    Parameters
    ----------
    otsu_binary : np.ndarray
        Binary mask from Otsu thresholding.
    adaptive_binary : np.ndarray
        Binary mask from adaptive thresholding.
    refdiff_binary : np.ndarray or None
        Binary mask from reference differencing.
    weights : tuple of float
        Weights for (otsu, adaptive, refdiff). Must sum to ~1.0.

    Returns
    -------
    fused : np.ndarray
        Fused binary mask.

    Notes
    -----
    Each pixel is classified as foreground if:
        weighted_sum > 0.5 * 255

    This soft voting reduces false positives from any single method.
    """
    # Normalise to float
    channels = [
        (otsu_binary.astype(np.float32) / 255.0) * weights[0],
        (adaptive_binary.astype(np.float32) / 255.0) * weights[1],
    ]

    if refdiff_binary is not None:
        channels.append(
            (refdiff_binary.astype(np.float32) / 255.0) * weights[2]
        )
    else:
        # Redistribute weight
        total_w = weights[0] + weights[1]
        channels = [
            (otsu_binary.astype(np.float32) / 255.0) * (weights[0] / total_w),
            (adaptive_binary.astype(np.float32) / 255.0) * (weights[1] / total_w),
        ]

    combined = sum(channels)
    fused = (combined > 0.5).astype(np.uint8) * 255

    return fused
