# roi.py - Region of Interest (Parking Slot) Extraction
"""
Manages parking slot definitions and ROI extraction.

Functions:
    load_slot_coordinates()       - Load parking slot polygon coordinates
    transform_slots_to_bev()      - Apply homography H to all slot polygons
    extract_slot_image()          - Crop a single slot from the full image
    extract_all_slots()           - Extract all slot sub-images
    create_slot_mask()            - Create a binary mask for a slot polygon
    create_eroded_core_mask()     - Erode mask to exclude painted lines
    draw_slots_on_image()         - Visualize all slot boundaries on image
    save_slots_json()             - Save slot definitions to JSON
    load_slots_json()             - Load slot definitions from JSON

Theory:
    ROI (Region of Interest) is a specific area within an image that we
    want to analyze. Each parking slot is an ROI defined by a polygon
    (quadrilateral from PKLot annotations).

    Critical design decision:
        PKLot's slot coordinates are in ORIGINAL image coordinates.
        Our pipeline works in BIRD'S-EYE VIEW coordinates.
        We must transform the polygons through H ONCE, then use
        the transformed coordinates for all downstream processing.

        bev_points = cv2.perspectiveTransform(original_points, H)

    Two mask types per slot:
        1. FULL MASK  — the complete polygon
        2. CORE MASK  — eroded by ~5px to exclude painted lane lines
           that could confuse edge detection
"""

import cv2
import numpy as np
import json
import os


def load_slot_coordinates(filepath):
    """
    Load parking slot coordinates from a file.

    Supports:
        - JSON (from save_slots_json)
        - PKLot XML (via io_utils.parse_pklot_xml)

    Parameters
    ----------
    filepath : str
        Path to the coordinates file.

    Returns
    -------
    slots : dict
        Mapping of slot_id (int) → np.ndarray (Nx2 float32 polygon).
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.json':
        return load_slots_json(filepath)
    elif ext == '.xml':
        from src.io_utils import parse_pklot_xml
        parsed = parse_pklot_xml(filepath)
        return {s['id']: s['points'] for s in parsed}
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def transform_slots_to_bev(slots, H):
    """
    Transform all slot polygons from original image coordinates to
    bird's-eye view coordinates using homography H.

    THIS IS THE CRITICAL STEP that avoids re-annotating 40+ slots by hand.

    Parameters
    ----------
    slots : dict
        Mapping of slot_id → np.ndarray (Nx2) in original coords.
    H : np.ndarray
        3x3 homography matrix.

    Returns
    -------
    bev_slots : dict
        Mapping of slot_id → np.ndarray (Nx2) in BEV coords.
    """
    from src.geometry import transform_points

    bev_slots = {}
    for slot_id, points in slots.items():
        bev_points = transform_points(points, H)
        bev_slots[slot_id] = bev_points

    return bev_slots


def extract_slot_image(image, polygon, pad=0):
    """
    Extract a single parking slot sub-image using a polygon mask.

    Parameters
    ----------
    image : np.ndarray
        Full parking lot image (BGR or grayscale).
    polygon : np.ndarray
        Nx2 array of polygon vertices defining the slot.
    pad : int
        Padding in pixels around the bounding box.

    Returns
    -------
    slot_image : np.ndarray
        Cropped and masked slot image (only pixels inside polygon).
    bbox : tuple
        (x, y, w, h) bounding box of the slot.
    mask : np.ndarray
        Binary mask within the bounding box (255 inside, 0 outside).
    """
    # Get bounding box
    pts = np.array(polygon, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)

    # Apply padding
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)

    # Crop
    cropped = image[y1:y2, x1:x2].copy()

    # Create mask in crop coordinates
    shifted_pts = pts - np.array([x1, y1])
    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [shifted_pts], 255)

    # Apply mask
    if len(cropped.shape) == 3:
        slot_image = cv2.bitwise_and(cropped, cropped, mask=mask)
    else:
        slot_image = cv2.bitwise_and(cropped, cropped, mask=mask)

    return slot_image, (x1, y1, x2 - x1, y2 - y1), mask


def extract_all_slots(image, slots, pad=0):
    """
    Extract all slot sub-images from a parking lot image.

    Parameters
    ----------
    image : np.ndarray
        Full parking lot image.
    slots : dict
        Mapping of slot_id → polygon (Nx2 array).
    pad : int
        Padding around each slot.

    Returns
    -------
    slot_images : dict
        Mapping of slot_id → (slot_image, bbox, mask).
    """
    slot_images = {}
    for slot_id, polygon in slots.items():
        slot_img, bbox, mask = extract_slot_image(image, polygon, pad)
        slot_images[slot_id] = {
            'image': slot_img,
            'bbox': bbox,
            'mask': mask,
            'polygon': polygon
        }
    return slot_images


def create_slot_mask(image_shape, polygon):
    """
    Create a binary mask for a single slot polygon on a full-sized image.

    Parameters
    ----------
    image_shape : tuple
        (height, width) of the full image.
    polygon : np.ndarray
        Nx2 array of polygon vertices.

    Returns
    -------
    mask : np.ndarray
        Binary mask (uint8), 255 inside polygon, 0 outside.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def create_eroded_core_mask(mask, erosion_px=5):
    """
    Create an eroded "core" mask that excludes painted lane lines
    at the edges of the slot.

    Parameters
    ----------
    mask : np.ndarray
        Full slot mask (binary, uint8).
    erosion_px : int
        Number of pixels to erode from each edge.

    Returns
    -------
    core_mask : np.ndarray
        Eroded mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * erosion_px + 1, 2 * erosion_px + 1)
    )
    core_mask = cv2.erode(mask, kernel, iterations=1)
    return core_mask


def draw_slots_on_image(image, slots, labels=None, colors=None,
                        thickness=2, show_ids=True, font_scale=0.5):
    """
    Draw all parking slot boundaries on the image.

    Parameters
    ----------
    image : np.ndarray
        Full parking lot image (will be copied, not modified).
    slots : dict
        Mapping of slot_id → polygon (Nx2 array).
    labels : dict or None
        Mapping of slot_id → 0 (vacant) or 1 (occupied).
        If provided, colors slots green (vacant) or red (occupied).
    colors : dict or None
        Custom per-slot colors. Overrides label-based coloring.
    thickness : int
        Line thickness.
    show_ids : bool
        Whether to draw slot ID numbers.
    font_scale : float
        Font scale for slot ID text.

    Returns
    -------
    annotated : np.ndarray
        Image with slot boundaries drawn.
    """
    annotated = image.copy()

    # Default colors
    vacant_color = (0, 255, 0)    # Green
    occupied_color = (0, 0, 255)  # Red
    default_color = (255, 255, 0) # Cyan

    for slot_id, polygon in slots.items():
        pts = np.array(polygon, dtype=np.int32)

        # Determine color
        if colors and slot_id in colors:
            color = colors[slot_id]
        elif labels and slot_id in labels:
            color = occupied_color if labels[slot_id] == 1 else vacant_color
        else:
            color = default_color

        # Draw polygon outline
        cv2.polylines(annotated, [pts], isClosed=True,
                      color=color, thickness=thickness)

        # Draw semi-transparent fill
        if labels is not None and slot_id in labels:
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)

        # Draw slot ID
        if show_ids:
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(annotated, str(slot_id),
                        tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


def save_slots_json(filepath, slots, coordinate_system='bev'):
    """
    Save slot definitions to JSON file.

    Parameters
    ----------
    filepath : str
        Output path (e.g., 'config/slots.json').
    slots : dict
        Mapping of slot_id → np.ndarray (Nx2 polygon).
    coordinate_system : str
        'bev' or 'original' — documents which coordinate space.
    """
    data = {
        'coordinate_system': coordinate_system,
        'num_slots': len(slots),
        'slots': {}
    }

    for slot_id, polygon in slots.items():
        data['slots'][str(slot_id)] = polygon.tolist()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_slots_json(filepath):
    """
    Load slot definitions from JSON file.

    Parameters
    ----------
    filepath : str
        Path to slots.json.

    Returns
    -------
    slots : dict
        Mapping of slot_id (int) → np.ndarray (Nx2 float32 polygon).
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    slots = {}
    for slot_id_str, points_list in data['slots'].items():
        slot_id = int(slot_id_str)
        slots[slot_id] = np.array(points_list, dtype=np.float32)

    return slots


def compute_slot_areas(slots):
    """
    Compute the area of each slot polygon (in pixels²).

    Parameters
    ----------
    slots : dict
        Mapping of slot_id → polygon (Nx2 array).

    Returns
    -------
    areas : dict
        Mapping of slot_id → area (float).
    """
    areas = {}
    for slot_id, polygon in slots.items():
        pts = np.array(polygon, dtype=np.float32)
        areas[slot_id] = float(cv2.contourArea(pts))
    return areas


def assign_slot_rows(slots, angle_tolerance_deg=15.0):
    """
    Group slots into rows based on Y-coordinate proximity.
    Useful for per-row statistics.

    Parameters
    ----------
    slots : dict
        Mapping of slot_id → polygon (Nx2 array).
    angle_tolerance_deg : float
        Not used currently; reserved for angled lots.

    Returns
    -------
    rows : dict
        Mapping of row_index → list of slot_ids.
    """
    # Get centroid Y for each slot
    centroids = {}
    for slot_id, polygon in slots.items():
        centroid_y = np.mean(polygon[:, 1])
        centroids[slot_id] = centroid_y

    # Sort by Y coordinate
    sorted_ids = sorted(centroids.keys(), key=lambda sid: centroids[sid])

    # Group into rows (slots within ~30px of each other)
    rows = {}
    row_idx = 0
    if sorted_ids:
        current_row = [sorted_ids[0]]
        current_y = centroids[sorted_ids[0]]

        for sid in sorted_ids[1:]:
            if abs(centroids[sid] - current_y) < 30:
                current_row.append(sid)
            else:
                rows[row_idx] = current_row
                row_idx += 1
                current_row = [sid]
                current_y = centroids[sid]

        rows[row_idx] = current_row

    return rows
