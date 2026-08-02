# =============================================================================
# features.py - Feature Extraction
# =============================================================================
"""
Extract numerical features from parking slot images for occupancy estimation.

Functions:
    compute_edge_density()        - ρ_e: Canny edge pixels / total pixels
    compute_foreground_ratio()    - ρ_f: white pixels / total pixels
    compute_gradient_magnitude()  - ḡ: mean Sobel gradient magnitude
    compute_local_variance()      - σ²: variance of pixel intensities
    compute_largest_component()   - α: largest connected component / total
    compute_intensity_std()       - σ_I: standard deviation of intensities
    compute_otsu_separability()   - η: Otsu between-class variance ratio
    compute_mean_saturation()     - S̄: mean saturation in HSV
    extract_all_features()        - All 8 features for a single slot
    compute_fisher_ratio()        - Fisher discriminant ratio per feature

Theory:
    Features convert visual information into numbers for thresholding.
    All features are AREA-NORMALISED so they are comparable across slots.

    The physical signal we exploit:

        EMPTY SLOT                     OCCUPIED SLOT
        ─────────────────────          ─────────────────────
        Homogeneous asphalt            Heterogeneous surfaces
        Very few edges                 Dense structured edges
        Low intensity variance         High intensity variance
        Low gradient energy            High gradient energy
        Low saturation (grey)          Often saturated colour
        No large coherent blob         One large coherent blob

    EDGE DENSITY is the single strongest discriminator.
    A car is an edge factory; asphalt is not.

    Feature vector per slot:
        f_i = [ρ_e, ρ_f, ḡ, σ², α, σ_I, η, S̄]
    All normalised to [0, 1] range for weighted scoring.
"""

import cv2
import numpy as np


def compute_edge_density(gray_image, low=50, high=150, mask=None):
    """
    Compute fraction of edge pixels using Canny edge detection.

    Parameters
    ----------
    gray_image : np.ndarray
        Preprocessed grayscale image.
    low : int
        Canny lower hysteresis threshold.
    high : int
        Canny upper hysteresis threshold.
    mask : np.ndarray or None
        Binary mask to restrict computation to ROI.

    Returns
    -------
    density : float
        Edge pixel count / total pixel count (0.0 to 1.0).
    edges : np.ndarray
        Canny edge image.

    Notes
    -----
    Canny edge detection steps:
        1. Gaussian smooth (already done in preprocessing)
        2. Sobel gradients → magnitude + direction
        3. Non-maximum suppression (thin to 1-pixel ridges)
        4. Double thresholding + hysteresis:
           - Strong edges (> high): definitely edges
           - Weak edges (low < x < high): edges only if connected to strong
           - Below low: rejected

    Why Canny and not just Sobel:
    - Canny produces THIN edges (1 pixel wide) → cleaner density
    - Hysteresis links weak vehicle contours to strong ones
    - Noise edges (below low) are cleanly rejected
    """
    edges = cv2.Canny(gray_image, low, high)

    if mask is not None:
        edges = cv2.bitwise_and(edges, mask)
        total_pixels = cv2.countNonZero(mask)
    else:
        total_pixels = gray_image.size

    if total_pixels == 0:
        return 0.0, edges

    edge_pixels = cv2.countNonZero(edges)
    density = edge_pixels / total_pixels

    return density, edges


def compute_foreground_ratio(binary_image, mask=None):
    """
    Compute fraction of white (foreground) pixels.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary mask from segmentation.
    mask : np.ndarray or None
        ROI mask.

    Returns
    -------
    ratio : float
        Foreground pixel count / total pixel count (0.0 to 1.0).
    """
    if mask is not None:
        fg = cv2.bitwise_and(binary_image, mask)
        total = cv2.countNonZero(mask)
    else:
        fg = binary_image
        total = binary_image.size

    if total == 0:
        return 0.0

    return cv2.countNonZero(fg) / total


def compute_gradient_magnitude(gray_image, mask=None):
    """
    Compute mean Sobel gradient magnitude.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale image.
    mask : np.ndarray or None
        ROI mask.

    Returns
    -------
    mean_grad : float
        Mean gradient magnitude (normalised to 0-1 range).
    grad_image : np.ndarray
        Gradient magnitude image (float32).

    Notes
    -----
    Sobel computes partial derivatives:
        Gx = ∂I/∂x (horizontal edges)
        Gy = ∂I/∂y (vertical edges)
        |G| = √(Gx² + Gy²)

    Mean gradient is a graded alternative to binary Canny:
    - Avoids cliff-edge sensitivity to hysteresis parameters
    - Provides continuous discrimination signal
    """
    # Use Sobel with 64-bit precision to avoid overflow
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)

    # Normalise to [0, 1]
    max_possible = np.sqrt(2) * 255 * 4  # Max Sobel response for uint8
    grad_normalised = grad / max_possible

    if mask is not None:
        mask_bool = mask > 0
        if mask_bool.sum() == 0:
            return 0.0, grad_normalised.astype(np.float32)
        mean_grad = float(np.mean(grad_normalised[mask_bool]))
    else:
        mean_grad = float(np.mean(grad_normalised))

    return mean_grad, grad_normalised.astype(np.float32)


def compute_local_variance(gray_image, mask=None):
    """
    Compute variance of pixel intensity values within the ROI.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale image.
    mask : np.ndarray or None
        ROI mask.

    Returns
    -------
    variance : float
        Intensity variance (normalised to 0-1).

    Notes
    -----
    Empty asphalt → low variance (uniform grey surface).
    Occupied slot → high variance (car has mixed colours, shadows,
    reflections, windows).

    Normalisation: divide by max possible variance (255²/4 ≈ 16256).
    """
    if mask is not None:
        pixels = gray_image[mask > 0].astype(np.float64)
    else:
        pixels = gray_image.flatten().astype(np.float64)

    if len(pixels) == 0:
        return 0.0

    variance = float(np.var(pixels))
    # Normalise: max variance for uint8 is when half pixels=0, half=255
    # Var = (255/2)² = 16256.25
    return min(variance / 16256.25, 1.0)


def compute_largest_component(binary_image, mask=None):
    """
    Compute ratio of largest connected component area to ROI area.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary mask from segmentation.
    mask : np.ndarray or None
        ROI mask.

    Returns
    -------
    ratio : float
        Largest component area / total ROI area (0.0 to 1.0).
    num_components : int
        Total number of connected components.

    Notes
    -----
    A car produces ONE LARGE connected component.
    Gravel, leaves, noise produce MANY SMALL components.

    This feature distinguishes "one big thing" (car) from
    "many small things" (noise after thresholding).

    Uses 8-connectivity (including diagonals).
    """
    if mask is not None:
        active = cv2.bitwise_and(binary_image, mask)
        total_area = cv2.countNonZero(mask)
    else:
        active = binary_image
        total_area = binary_image.size

    if total_area == 0:
        return 0.0, 0

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        active, connectivity=8
    )

    if num_labels <= 1:  # Only background
        return 0.0, 0

    # Component 0 is background; find largest foreground component
    # stats[:, cv2.CC_STAT_AREA] gives area of each component
    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
    largest_area = int(np.max(areas))

    ratio = largest_area / total_area

    return ratio, num_labels - 1  # -1 to exclude background


def compute_intensity_std(gray_image, mask=None):
    """
    Compute standard deviation of pixel intensities.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale image.
    mask : np.ndarray or None
        ROI mask.

    Returns
    -------
    std_normalised : float
        Normalised standard deviation (0.0 to 1.0).
    """
    if mask is not None:
        pixels = gray_image[mask > 0].astype(np.float64)
    else:
        pixels = gray_image.flatten().astype(np.float64)

    if len(pixels) == 0:
        return 0.0

    std = float(np.std(pixels))
    # Normalise: max std for uint8 is 127.5 (half 0, half 255)
    return min(std / 127.5, 1.0)


def compute_otsu_separability(gray_image, mask=None):
    """
    Compute Otsu's between-class variance separability measure η.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale image.
    mask : np.ndarray or None
        ROI mask.

    Returns
    -------
    separability : float
        η ∈ [0, 1]. Higher = stronger bimodal distribution.

    Notes
    -----
    η is a FREE bonus feature from Otsu thresholding.
    - Occupied slots: strong bimodality (dark+light regions) → high η
    - Empty slots: unimodal asphalt → low η
    """
    if mask is not None:
        roi_pixels = gray_image[mask > 0]
    else:
        roi_pixels = gray_image.flatten()

    if len(roi_pixels) == 0:
        return 0.0

    # Create a small image from the ROI pixels for Otsu
    roi_img = roi_pixels.reshape(-1, 1).astype(np.uint8)

    # Compute histogram
    hist = cv2.calcHist([roi_img], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0:
        return 0.0

    p = hist / total
    mu_total = np.sum(np.arange(256) * p)
    sigma_total_sq = np.sum(((np.arange(256) - mu_total) ** 2) * p)

    if sigma_total_sq == 0:
        return 0.0

    # Find optimal Otsu threshold
    best_sigma_b = 0.0
    w0_acc = 0.0
    mu0_acc = 0.0

    for t in range(256):
        w0_acc += p[t]
        if w0_acc == 0:
            continue
        w1 = 1.0 - w0_acc
        if w1 == 0:
            break

        mu0_acc += t * p[t]
        mu0 = mu0_acc / w0_acc
        mu1 = (mu_total - mu0_acc) / w1

        sigma_b = w0_acc * w1 * (mu0 - mu1) ** 2
        if sigma_b > best_sigma_b:
            best_sigma_b = sigma_b

    return best_sigma_b / sigma_total_sq


def compute_mean_saturation(bgr_image, mask=None):
    """
    Compute mean saturation in HSV colour space.

    Parameters
    ----------
    bgr_image : np.ndarray
        Input BGR colour image.
    mask : np.ndarray or None
        ROI mask.

    Returns
    -------
    mean_sat : float
        Normalised mean saturation (0.0 to 1.0).

    Notes
    -----
    Empty asphalt is desaturated (grey → S ≈ 0).
    Cars often have saturated paint colours (S > 0).

    This feature helps in good lighting but is unreliable
    in shadow or at night.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]

    if mask is not None:
        pixels = s_channel[mask > 0].astype(np.float64)
    else:
        pixels = s_channel.flatten().astype(np.float64)

    if len(pixels) == 0:
        return 0.0

    return float(np.mean(pixels)) / 255.0


def extract_all_features(gray_image, binary_image, bgr_image=None,
                          mask=None, canny_low=50, canny_high=150):
    """
    Extract all 8 features for a single parking slot.

    Parameters
    ----------
    gray_image : np.ndarray
        Preprocessed grayscale slot image.
    binary_image : np.ndarray
        Segmented binary mask.
    bgr_image : np.ndarray or None
        Original BGR image (needed for saturation feature).
    mask : np.ndarray or None
        ROI mask.
    canny_low : int
        Canny lower threshold.
    canny_high : int
        Canny upper threshold.

    Returns
    -------
    features : dict
        Dictionary with all 8 features + edge image.
        Keys: edge_density, foreground_ratio, gradient_magnitude,
              local_variance, largest_component, intensity_std,
              otsu_separability, mean_saturation
    """
    edge_density, edges = compute_edge_density(
        gray_image, canny_low, canny_high, mask
    )
    foreground_ratio = compute_foreground_ratio(binary_image, mask)
    gradient_mag, grad_img = compute_gradient_magnitude(gray_image, mask)
    local_var = compute_local_variance(gray_image, mask)
    largest_comp, num_comp = compute_largest_component(binary_image, mask)
    intensity_std = compute_intensity_std(gray_image, mask)
    otsu_sep = compute_otsu_separability(gray_image, mask)

    mean_sat = 0.0
    if bgr_image is not None:
        mean_sat = compute_mean_saturation(bgr_image, mask)

    features = {
        'edge_density': edge_density,
        'foreground_ratio': foreground_ratio,
        'gradient_magnitude': gradient_mag,
        'local_variance': local_var,
        'largest_component': largest_comp,
        'num_components': num_comp,
        'intensity_std': intensity_std,
        'otsu_separability': otsu_sep,
        'mean_saturation': mean_sat,
        # Auxiliary images for visualization
        '_edges': edges,
        '_gradient': grad_img,
    }

    return features


def compute_fisher_ratio(feature_values_occupied, feature_values_vacant):
    """
    Compute Fisher's linear discriminant ratio for a single feature.

    Parameters
    ----------
    feature_values_occupied : array-like
        Feature values for occupied slots.
    feature_values_vacant : array-like
        Feature values for vacant slots.

    Returns
    -------
    fisher_ratio : float
        (μ₁ - μ₀)² / (σ₁² + σ₀²)
        Higher = better discrimination.

    Notes
    -----
    Fisher ratio quantifies how well a feature separates two classes.
    Used to rank features and justify which ones carry the most weight
    in the decision rule.
    """
    occ = np.array(feature_values_occupied, dtype=np.float64)
    vac = np.array(feature_values_vacant, dtype=np.float64)

    mu_occ = np.mean(occ) if len(occ) > 0 else 0.0
    mu_vac = np.mean(vac) if len(vac) > 0 else 0.0

    var_occ = np.var(occ) if len(occ) > 0 else 1e-10
    var_vac = np.var(vac) if len(vac) > 0 else 1e-10

    denom = var_occ + var_vac
    if denom < 1e-10:
        return 0.0

    return (mu_occ - mu_vac) ** 2 / denom
