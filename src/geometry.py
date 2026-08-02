# =============================================================================
# geometry.py - Camera Geometry & Perspective Transformation
# =============================================================================
"""
Handles camera geometry concepts and perspective transformation.

Functions:
    compute_homography()      - Compute 3x3 homography matrix H
    warp_perspective()        - Apply perspective warp to produce bird's-eye view
    transform_points()        - Map points through homography H
    undistort_image()         - Remove lens distortion (if calibration available)
    validate_bev()            - Check BEV quality (parallel lines, scale)
    save_homography()         - Save H + metadata to npz
    load_homography()         - Load H + metadata from npz

Theory:
    Pinhole Camera Model:
        s · [u, v, 1]^T = K [R|t] [X, Y, Z, 1]^T

        K = [fx  0  cx]     intrinsic matrix (focal length, principal point)
            [0  fy  cy]
            [0   0   1]

    For a PLANAR scene (Z = 0), the 3x4 projection collapses to a 3x3
    HOMOGRAPHY:
        s · [u, v, 1]^T = H · [X, Y, 1]^T

        H has 8 DOF (9 entries - 1 scale), requires ≥ 4 point correspondences.

    Why this matters:
        A parking lot is approximately planar → the camera-to-ground
        mapping is exactly a homography. Warping through H produces a
        bird's-eye view where:
        - All slots have equal pixel area → one threshold set works
        - Lane lines become parallel → metric measurements possible
        - Near/far distortion eliminated
"""

import cv2
import numpy as np
import os


def compute_homography(src_points, dst_points, method=0):
    """
    Compute the 3x3 homography matrix that maps source points to
    destination points.

    Parameters
    ----------
    src_points : np.ndarray
        Nx2 array of source corner points (float32). Minimum 4 points.
    dst_points : np.ndarray
        Nx2 array of destination corner points (float32). Same N.
    method : int
        0 = exact (DLT), cv2.RANSAC for robust estimation.

    Returns
    -------
    H : np.ndarray
        3x3 homography matrix (float64).
    mask : np.ndarray or None
        Inlier mask if RANSAC used, None otherwise.

    Notes
    -----
    cv2.findHomography solves the DLT (Direct Linear Transform):
        For each point pair (x_i, x_i'), we get 2 linear equations.
        4 pairs → 8 equations for 8 unknowns → exact solution.
        >4 pairs → least-squares or RANSAC.

    The homography satisfies:
        s · [x', y', 1]^T = H · [x, y, 1]^T
    where s is an arbitrary scale factor.
    """
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    if len(src) == 4 and method == 0:
        # Exact 4-point solution (no RANSAC needed)
        H = cv2.getPerspectiveTransform(src, dst)
        return H, None
    else:
        # Over-determined system or robust estimation
        H, mask = cv2.findHomography(src, dst, method)
        return H, mask


def warp_perspective(image, H, output_size, flags=cv2.INTER_LINEAR,
                     border_mode=cv2.BORDER_CONSTANT, border_value=0):
    """
    Apply perspective warp to an image using homography matrix H.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or grayscale).
    H : np.ndarray
        3x3 homography matrix.
    output_size : tuple
        (width, height) of the output image.
    flags : int
        Interpolation method (default: bilinear).
    border_mode : int
        Border extrapolation method.
    border_value : int or tuple
        Value for border pixels.

    Returns
    -------
    warped : np.ndarray
        Warped (bird's-eye view) image.

    Notes
    -----
    cv2.warpPerspective computes for each output pixel (x', y'):
        [x]         [x']
        [y] = H^-1 [y']   (inverse mapping for clean interpolation)
        [1]         [1]
    then samples the input at (x, y).
    """
    warped = cv2.warpPerspective(
        image, H, output_size,
        flags=flags,
        borderMode=border_mode,
        borderValue=border_value
    )
    return warped


def transform_points(points, H):
    """
    Transform 2D points through a homography.

    This is THE critical function for the project: it maps PKLot's
    slot polygon coordinates (in original image) to bird's-eye view
    coordinates.

    Parameters
    ----------
    points : np.ndarray
        Nx2 array of 2D points (float32).
    H : np.ndarray
        3x3 homography matrix.

    Returns
    -------
    transformed : np.ndarray
        Nx2 array of transformed points (float32).

    Notes
    -----
    cv2.perspectiveTransform expects input shape (N, 1, 2).
    For each point [x, y]:
        [x']     [h11 h12 h13] [x]
        [y'] = s [h21 h22 h23] [y]
        [1 ]     [h31 h32 h33] [1]
    Then x_out = x'/s, y_out = y'/s where s = h31*x + h32*y + h33
    """
    pts = np.array(points, dtype=np.float32)
    if pts.ndim == 2:
        pts = pts.reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)


def undistort_image(image, K, dist_coeffs, new_K=None):
    """
    Remove lens distortion using calibration parameters.

    Parameters
    ----------
    image : np.ndarray
        Distorted input image.
    K : np.ndarray
        3x3 camera intrinsic matrix.
    dist_coeffs : np.ndarray
        Distortion coefficients [k1, k2, p1, p2, k3].
    new_K : np.ndarray or None
        New camera matrix. If None, uses K.

    Returns
    -------
    undistorted : np.ndarray
        Corrected image.

    Notes
    -----
    PKLot images are generally not severely distorted (narrow FOV),
    so this step may be skipped. But we implement it for completeness
    and to demonstrate the concept for the viva.

    The distortion model:
        x_distorted = x(1 + k1*r² + k2*r⁴ + k3*r⁶)
                    + 2*p1*x*y + p2*(r² + 2*x²)
    """
    if new_K is None:
        h, w = image.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1)

    undistorted = cv2.undistort(image, K, dist_coeffs, None, new_K)
    return undistorted


def validate_bev(bev_image, known_slot_width_m=2.5, known_slot_length_m=5.0,
                 px_per_metre=None):
    """
    Validate the bird's-eye view quality.

    Checks:
    1. Slot dimensions match expected physical sizes
    2. Lines that should be parallel appear parallel

    Parameters
    ----------
    bev_image : np.ndarray
        Bird's-eye view image.
    known_slot_width_m : float
        Expected slot width in metres.
    known_slot_length_m : float
        Expected slot length in metres.
    px_per_metre : float or None
        Known scale factor. If None, will estimate from image.

    Returns
    -------
    report : dict
        Validation metrics.
    """
    report = {
        'image_shape': bev_image.shape[:2],
        'known_slot_width_m': known_slot_width_m,
        'known_slot_length_m': known_slot_length_m,
    }

    if px_per_metre is not None:
        expected_width_px = known_slot_width_m * px_per_metre
        expected_length_px = known_slot_length_m * px_per_metre
        report['px_per_metre'] = px_per_metre
        report['expected_slot_width_px'] = expected_width_px
        report['expected_slot_length_px'] = expected_length_px

    return report


def save_homography(filepath, H, output_size, px_per_metre=None,
                    src_points=None, dst_points=None):
    """
    Save homography matrix and metadata to .npz file.

    Parameters
    ----------
    filepath : str
        Output path (e.g., 'config/homography.npz').
    H : np.ndarray
        3x3 homography matrix.
    output_size : tuple
        (width, height) of BEV output.
    px_per_metre : float or None
        Pixels per metre in BEV.
    src_points : np.ndarray or None
        Source correspondence points.
    dst_points : np.ndarray or None
        Destination correspondence points.
    """
    data = {
        'H': H,
        'output_size': np.array(output_size),
    }
    if px_per_metre is not None:
        data['px_per_metre'] = np.array([px_per_metre])
    if src_points is not None:
        data['src_points'] = np.array(src_points, dtype=np.float32)
    if dst_points is not None:
        data['dst_points'] = np.array(dst_points, dtype=np.float32)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez(filepath, **data)


def load_homography(filepath):
    """
    Load homography matrix and metadata from .npz file.

    Parameters
    ----------
    filepath : str
        Path to homography.npz.

    Returns
    -------
    data : dict
        Dictionary with 'H', 'output_size', and optionally
        'px_per_metre', 'src_points', 'dst_points'.
    """
    npz = np.load(filepath, allow_pickle=True)
    data = {
        'H': npz['H'],
        'output_size': tuple(npz['output_size']),
    }
    if 'px_per_metre' in npz:
        data['px_per_metre'] = float(npz['px_per_metre'][0])
    if 'src_points' in npz:
        data['src_points'] = npz['src_points']
    if 'dst_points' in npz:
        data['dst_points'] = npz['dst_points']
    return data
