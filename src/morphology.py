# morphology.py - Morphological Operations
"""
Morphological operations for cleaning binary masks.

Functions:
    apply_erosion()           - Shrink white regions (removes noise)
    apply_dilation()          - Expand white regions (fills gaps)
    apply_opening()           - Erosion → dilation (removes small noise)
    apply_closing()           - Dilation → erosion (fills small holes)
    clean_binary_mask()       - Full morphological cleanup pipeline
    morphology_grid()         - Return all stages for visualization

Theory:
    After thresholding, binary masks are noisy:
    - Spurious white pixels (pepper noise) from texture/compression
    - Holes inside vehicle regions from windows/dark paint
    - Fragmented blobs from shadows breaking up the car silhouette

    Morphological ops use a STRUCTURING ELEMENT (small kernel, usually
    rectangular or elliptical) to systematically clean the mask.

    Set-theoretic definitions (B = structuring element):
        Erosion:  A ⊖ B = { z | B_z ⊆ A }
                  → pixel is white only if ALL of B fits inside A
                  → shrinks white, removes small bright specks

        Dilation: A ⊕ B = { z | B_z ∩ A ≠ ∅ }
                  → pixel is white if ANY of B overlaps A
                  → grows white, fills small gaps

        Opening:  A ∘ B = (A ⊖ B) ⊕ B
                  → erosion then dilation → removes noise, preserves size

        Closing:  A • B = (A ⊕ B) ⊖ B
                  → dilation then erosion → fills holes, preserves size

    Why structuring element sizes should come from PHYSICAL dimensions:
        In BEV at known px/m scale, a 3cm noise speck = ~3px.
        A 10cm gap in a car roof = ~10px.
        Using metric-derived sizes makes the pipeline transferable.
"""

import cv2
import numpy as np


def get_structuring_element(shape='rect', ksize=(3, 3)):
    """
    Create a morphological structuring element.

    Parameters
    ----------
    shape : str
        'rect', 'ellipse', or 'cross'.
    ksize : tuple of int
        Kernel size (height, width). Must be odd.

    Returns
    -------
    kernel : np.ndarray
        Structuring element.
    """
    shape_map = {
        'rect': cv2.MORPH_RECT,
        'ellipse': cv2.MORPH_ELLIPSE,
        'cross': cv2.MORPH_CROSS
    }
    cv_shape = shape_map.get(shape, cv2.MORPH_RECT)
    return cv2.getStructuringElement(cv_shape, ksize)


def apply_erosion(binary_image, kernel_size=(3, 3), iterations=1,
                  shape='rect'):
    """
    Apply erosion to shrink white regions.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary mask (uint8, 0 or 255).
    kernel_size : tuple of int
        Structuring element size.
    iterations : int
        Number of times to apply.
    shape : str
        Structuring element shape.

    Returns
    -------
    eroded : np.ndarray
        Eroded binary mask.
    """
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.erode(binary_image, kernel, iterations=iterations)


def apply_dilation(binary_image, kernel_size=(3, 3), iterations=1,
                   shape='rect'):
    """
    Apply dilation to expand white regions.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary mask (uint8).
    kernel_size : tuple of int
        Structuring element size.
    iterations : int
        Number of times to apply.
    shape : str
        Structuring element shape.

    Returns
    -------
    dilated : np.ndarray
        Dilated binary mask.
    """
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.dilate(binary_image, kernel, iterations=iterations)


def apply_opening(binary_image, kernel_size=(3, 3), shape='rect'):
    """
    Apply opening (erosion → dilation) to remove small noise.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary mask (uint8).
    kernel_size : tuple of int
        Structuring element size.
    shape : str
        Structuring element shape.

    Returns
    -------
    opened : np.ndarray
        Opened binary mask.

    Notes
    -----
    Opening removes bright objects smaller than the structuring element.
    A 3×3 opening removes isolated white pixels and thin protrusions.
    The overall shape/size of larger objects is approximately preserved.
    """
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)


def apply_closing(binary_image, kernel_size=(3, 3), shape='rect'):
    """
    Apply closing (dilation → erosion) to fill small holes.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary mask (uint8).
    kernel_size : tuple of int
        Structuring element size.
    shape : str
        Structuring element shape.

    Returns
    -------
    closed : np.ndarray
        Closed binary mask.

    Notes
    -----
    Closing fills dark holes smaller than the structuring element.
    Useful for:
    - Filling gaps in car silhouettes (windows appear dark)
    - Connecting fragmented vehicle blobs
    """
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)


def clean_binary_mask(binary_image, open_ksize=(3, 3), close_ksize=(5, 5),
                      dilate_ksize=(3, 3), erode_ksize=(3, 3),
                      shape='rect'):
    """
    Full morphological cleanup pipeline.

    Sequence: Opening → Closing → (Dilation → Erosion)

    Parameters
    ----------
    binary_image : np.ndarray
        Raw binary mask from thresholding.
    open_ksize : tuple
        Kernel size for opening (noise removal).
    close_ksize : tuple
        Kernel size for closing (hole filling).
    dilate_ksize : tuple
        Kernel size for final dilation.
    erode_ksize : tuple
        Kernel size for final erosion.
    shape : str
        Structuring element shape for all operations.

    Returns
    -------
    cleaned : np.ndarray
        Cleaned binary mask.

    Notes
    -----
    Pipeline rationale:
    1. Opening FIRST — kills small noise before closing would preserve it
    2. Closing — fills gaps and holes in the vehicle blob
    3. Dilation + Erosion — fine-tune the final boundary
       (slight grow then shrink to smooth ragged edges)
    """
    # Step 1: Remove small noise
    opened = apply_opening(binary_image, open_ksize, shape)

    # Step 2: Fill small holes
    closed = apply_closing(opened, close_ksize, shape)

    # Step 3: Smooth boundaries
    dilated = apply_dilation(closed, dilate_ksize, 1, shape)
    cleaned = apply_erosion(dilated, erode_ksize, 1, shape)

    return cleaned


def morphology_grid(binary_image, kernel_size=(3, 3)):
    """
    Apply all morphological operations and return results for visualization.

    Parameters
    ----------
    binary_image : np.ndarray
        Input binary mask.
    kernel_size : tuple
        Kernel size for all operations.

    Returns
    -------
    stages : list of (str, np.ndarray)
        List of (name, image) pairs showing each operation's result.
    """
    stages = [
        ('Original Binary', binary_image),
        ('Erosion', apply_erosion(binary_image, kernel_size)),
        ('Dilation', apply_dilation(binary_image, kernel_size)),
        ('Opening (E→D)', apply_opening(binary_image, kernel_size)),
        ('Closing (D→E)', apply_closing(binary_image, kernel_size)),
    ]

    # Also show the full cleanup pipeline
    cleaned = clean_binary_mask(binary_image)
    stages.append(('Full Cleanup', cleaned))

    return stages
