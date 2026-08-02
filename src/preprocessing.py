# =============================================================================
# preprocessing.py - Image Preprocessing Pipeline
# =============================================================================
"""
Image preprocessing functions for enhancing parking slot images.

Functions:
    to_grayscale()            - Convert BGR to grayscale (BT.601)
    equalize_histogram()      - Global histogram equalization
    apply_clahe()             - Contrast Limited Adaptive Histogram Equalization
    apply_gaussian_blur()     - Gaussian smoothing
    apply_median_blur()       - Median filtering (edge-preserving)
    preprocess_pipeline()     - Full preprocessing pipeline
    preprocess_ladder()       - Return intermediate results for visualization

Theory:
    Raw camera images contain noise, uneven lighting, and shadows.
    Preprocessing normalizes these variations so that downstream
    operations (segmentation, feature extraction) work reliably.

    Pipeline order and rationale:
        1. Grayscale (BT.601 weights: 0.299R + 0.587G + 0.114B)
           → 3× less data; removes daylight colour-temperature drift
        2. Gaussian 5×5 (σ ≈ 1.1)
           → Optimal pre-filter for gradient operators (Canny's derivation
             assumes Gaussian smoothing); separable → O(2N) not O(N²)
        3. Median 3×3
           → Kills salt-and-pepper noise, dead pixels, JPEG artefacts
             WHILE PRESERVING EDGES (non-linear rank filter)
        4. CLAHE (clip_limit=2.0, tile=8×8)
           → Local contrast enhancement. Sun-side and shadow-side of
             the lot get independent normalisation. Better than global
             HE which clips both extremes.

    We also keep the BGR image available for HSV shadow suppression
    in the segmentation stage.
"""

import cv2
import numpy as np


def to_grayscale(image):
    """
    Convert BGR image to grayscale using BT.601 luminance weights.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image (uint8).

    Returns
    -------
    gray : np.ndarray
        Grayscale image (uint8, single channel).

    Notes
    -----
    cv2.cvtColor(img, COLOR_BGR2GRAY) uses:
        Y = 0.299*R + 0.587*G + 0.114*B
    This is the ITU-R BT.601 standard luminance formula.
    """
    if len(image.shape) == 2:
        return image.copy()  # Already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def equalize_histogram(gray_image):
    """
    Apply global histogram equalization.

    Parameters
    ----------
    gray_image : np.ndarray
        Single-channel grayscale image.

    Returns
    -------
    equalized : np.ndarray
        Histogram-equalized image.

    Notes
    -----
    Global HE maps each intensity level so the output histogram is
    approximately uniform. This maximizes contrast but can:
    - Over-amplify noise in flat regions
    - Wash out details in high-contrast scenes
    Use CLAHE instead for outdoor parking lots.
    """
    return cv2.equalizeHist(gray_image)


def apply_clahe(gray_image, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    gray_image : np.ndarray
        Single-channel grayscale image (uint8).
    clip_limit : float
        Contrast limit for each tile. Higher = more contrast.
        2.0 is a good default for outdoor scenes.
    grid_size : tuple of int
        Number of tiles (rows, cols). (8, 8) means 64 tiles.

    Returns
    -------
    enhanced : np.ndarray
        CLAHE-enhanced image.

    Notes
    -----
    CLAHE divides the image into tiles, equalizes each independently,
    then blends tile boundaries with bilinear interpolation.

    The clip_limit prevents over-amplification: if any histogram bin
    exceeds the limit, the excess counts are redistributed uniformly.

    Why CLAHE beats global HE for parking lots:
    - A lot in sunshine has sun-lit + shadowed regions simultaneously
    - Global HE picks ONE mapping → one side is always wrong
    - CLAHE adapts per-tile → both sun and shadow get proper contrast
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=grid_size)
    return clahe.apply(gray_image)


def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    Apply Gaussian blur for noise reduction.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or color).
    kernel_size : tuple of int
        Kernel size (must be odd). (5, 5) is standard.
    sigma : float
        Gaussian standard deviation. 0 = auto from kernel size.

    Returns
    -------
    blurred : np.ndarray
        Smoothed image.

    Notes
    -----
    The 2D Gaussian kernel is separable:
        G(x,y) = G(x) · G(y)
    So a 5×5 convolution costs 10 multiplications per pixel, not 25.

    Why we need this before edge detection:
    - Canny's derivation assumes the input is smoothed
    - Noise → false edges → inflated edge density → false positives
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)


def apply_median_blur(image, kernel_size=3):
    """
    Apply median filter for salt-and-pepper noise removal.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or color).
    kernel_size : int
        Kernel size (must be odd). 3 or 5.

    Returns
    -------
    filtered : np.ndarray
        Median-filtered image.

    Notes
    -----
    Median filter replaces each pixel with the MEDIAN of its
    neighbourhood. This is a non-linear rank-order filter that:
    - Removes impulse (salt-and-pepper) noise completely
    - Preserves edges (unlike Gaussian, which blurs them)
    - Costs more than Gaussian (sorting, not convolution)

    Order matters: Gaussian first (smoothing), then median (impulse).
    """
    return cv2.medianBlur(image, kernel_size)


def preprocess_pipeline(image, clip_limit=2.0, grid_size=(8, 8),
                        gaussian_ksize=(5, 5), median_ksize=3):
    """
    Run the full preprocessing pipeline on a single slot image.

    Steps: BGR → Grayscale → Gaussian → Median → CLAHE

    Parameters
    ----------
    image : np.ndarray
        Input BGR slot image.
    clip_limit : float
        CLAHE clip limit.
    grid_size : tuple
        CLAHE tile grid size.
    gaussian_ksize : tuple
        Gaussian blur kernel size.
    median_ksize : int
        Median filter kernel size.

    Returns
    -------
    processed : np.ndarray
        Preprocessed grayscale image.
    """
    gray = to_grayscale(image)
    blurred = apply_gaussian_blur(gray, gaussian_ksize)
    median = apply_median_blur(blurred, median_ksize)
    enhanced = apply_clahe(median, clip_limit, grid_size)
    return enhanced


def preprocess_ladder(image, clip_limit=2.0, grid_size=(8, 8),
                      gaussian_ksize=(5, 5), median_ksize=3):
    """
    Run preprocessing and return ALL intermediate stages for visualization.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    Returns
    -------
    stages : list of (str, np.ndarray)
        List of (stage_name, image) tuples.
    """
    stages = [('Original (BGR)', image)]

    gray = to_grayscale(image)
    stages.append(('Grayscale', gray))

    blurred = apply_gaussian_blur(gray, gaussian_ksize)
    stages.append(('Gaussian 5×5', blurred))

    median = apply_median_blur(blurred, median_ksize)
    stages.append(('Median 3×3', median))

    he = equalize_histogram(gray)
    stages.append(('Global HE', he))

    clahe = apply_clahe(median, clip_limit, grid_size)
    stages.append(('CLAHE', clahe))

    return stages
