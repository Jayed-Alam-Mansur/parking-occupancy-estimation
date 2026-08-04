# io_utils.py - I/O Utilities & PKLot Dataset Parser
"""
Handles loading frames, parsing PKLot XML annotations, quality gates,
and ground truth export.

Functions:
    parse_pklot_xml()         - Parse a single PKLot XML annotation file
    load_lot_annotations()    - Load all annotations for a parking lot
    list_frames()             - List available frames by weather/time
    quality_gate()            - Reject too-dark or too-blurry frames
    export_ground_truth_csv() - Export annotations to labels.csv
    load_ground_truth_csv()   - Load labels.csv back
    curate_samples()          - Select representative sample frames

Theory:
    PKLot XML format:
        <parking>
            <space id="1" occupied="1">
                <contour>
                    <point x="100" y="200"/>
                    <point x="120" y="200"/>
                    ...
                </contour>
            </space>
            ...
        </parking>

    Each <space> has an integer id and an occupied flag (0 or 1).
    The <contour> contains 4 corner points defining the slot polygon
    in original image coordinates.
"""

import cv2
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd


def parse_pklot_xml(xml_path):
    """
    Parse a single PKLot XML annotation file.

    Parameters
    ----------
    xml_path : str
        Path to the XML annotation file.

    Returns
    -------
    slots : list of dict
        Each dict has keys:
            'id'       : int   - slot identifier
            'occupied' : int   - 0=vacant, 1=occupied
            'points'   : np.ndarray - Nx2 array of polygon vertices (float32)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    slots = []
    for space in root.findall('space'):
        occ_str = space.get('occupied')
        if occ_str is None:
            continue
            
        slot_id = int(space.get('id'))
        occupied = int(occ_str)

        points = []
        contour = space.find('contour')
        if contour is not None:
            for point in contour.findall('point'):
                x = int(point.get('x'))
                y = int(point.get('y'))
                points.append([x, y])

        if len(points) >= 3:  # Need at least 3 points for a polygon
            slots.append({
                'id': slot_id,
                'occupied': occupied,
                'points': np.array(points, dtype=np.float32)
            })

    return slots


def list_frames(lot_dir, weather=None):
    """
    List all available image frames for a parking lot.

    Parameters
    ----------
    lot_dir : str
        Path to lot directory (e.g., data/raw/PKLot/UFPR05).
    weather : str or None
        Filter by weather: 'Sunny', 'Cloudy', 'Rainy', or None for all.

    Returns
    -------
    frames : list of dict
        Each dict has keys: 'image_path', 'xml_path', 'weather', 'date',
                            'timestamp'
    """
    frames = []
    weather_dirs = ['sunny', 'cloudy', 'rainy']
    if weather is not None:
        weather_dirs = [weather]

    for w in weather_dirs:
        weather_path = os.path.join(lot_dir, w)
        if not os.path.isdir(weather_path):
            continue

        # PKLot structure: Lot/Weather/Date/images+xml
        for date_dir in sorted(os.listdir(weather_path)):
            date_path = os.path.join(weather_path, date_dir)
            if not os.path.isdir(date_path):
                continue

            # Find all image files
            image_files = sorted(glob.glob(os.path.join(date_path, '*.jpg')))
            image_files += sorted(glob.glob(os.path.join(date_path, '*.png')))

            for img_path in image_files:
                # Corresponding XML has same name with .xml extension
                base = os.path.splitext(img_path)[0]
                xml_path = base + '.xml'

                if os.path.exists(xml_path):
                    frames.append({
                        'image_path': img_path,
                        'xml_path': xml_path,
                        'weather': w,
                        'date': date_dir,
                        'timestamp': os.path.basename(base)
                    })

    return frames


def quality_gate(image, min_brightness=30, min_laplacian_var=50.0):
    """
    Reject frames that are too dark or too blurry.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    min_brightness : int
        Minimum mean brightness (0-255).
    min_laplacian_var : float
        Minimum Laplacian variance (focus measure).

    Returns
    -------
    passes : bool
        True if image passes quality checks.
    diagnostics : dict
        Dictionary with 'brightness' and 'blur_score' values.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness check: mean intensity
    brightness = float(np.mean(gray))

    # Blur check: Laplacian variance (higher = sharper)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    passes = (brightness >= min_brightness) and (blur_score >= min_laplacian_var)

    diagnostics = {
        'brightness': brightness,
        'blur_score': blur_score,
        'too_dark': brightness < min_brightness,
        'too_blurry': blur_score < min_laplacian_var
    }

    return passes, diagnostics


def export_ground_truth_csv(frames, output_path):
    """
    Parse all XML annotations and export to a single CSV file.

    Parameters
    ----------
    frames : list of dict
        Frame listing from list_frames().
    output_path : str
        Path to output CSV file.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: frame_path, slot_id, x1,y1,...,x4,y4, occupancy
    """
    rows = []
    for frame_info in frames:
        slots = parse_pklot_xml(frame_info['xml_path'])
        for slot in slots:
            pts = slot['points']
            row = {
                'frame_path': frame_info['image_path'],
                'weather': frame_info['weather'],
                'date': frame_info['date'],
                'slot_id': slot['id'],
                'occupied': slot['occupied'],
            }
            # Add corner coordinates
            for i, (x, y) in enumerate(pts[:4]):  # Max 4 corners
                row[f'x{i+1}'] = float(x)
                row[f'y{i+1}'] = float(y)
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def load_ground_truth_csv(csv_path):
    """
    Load ground truth labels from CSV.

    Parameters
    ----------
    csv_path : str
        Path to labels.csv.

    Returns
    -------
    df : pd.DataFrame
        Ground truth DataFrame.
    """
    return pd.read_csv(csv_path)


def curate_samples(frames, n_per_weather=7, output_dir='data/samples/'):
    """
    Select representative sample frames across weather conditions.

    Parameters
    ----------
    frames : list of dict
        Full frame listing.
    n_per_weather : int
        Number of frames to select per weather condition.
    output_dir : str
        Directory to copy sample frames to.

    Returns
    -------
    selected : list of dict
        Selected frame info dictionaries.
    """
    import shutil
    os.makedirs(output_dir, exist_ok=True)

    selected = []
    for weather in ['sunny', 'cloudy', 'rainy']:
        weather_frames = [f for f in frames if f['weather'] == weather]
        if not weather_frames:
            continue

        # Evenly space the selection
        step = max(1, len(weather_frames) // n_per_weather)
        chosen = weather_frames[::step][:n_per_weather]

        for frame_info in chosen:
            # Copy image to samples dir
            basename = f"{weather}_{os.path.basename(frame_info['image_path'])}"
            dst = os.path.join(output_dir, basename)
            shutil.copy2(frame_info['image_path'], dst)

            # Copy XML too
            xml_basename = f"{weather}_{os.path.basename(frame_info['xml_path'])}"
            xml_dst = os.path.join(output_dir, xml_basename)
            shutil.copy2(frame_info['xml_path'], xml_dst)

            frame_copy = frame_info.copy()
            frame_copy['sample_path'] = dst
            frame_copy['sample_xml_path'] = xml_dst
            selected.append(frame_copy)

    return selected


def get_slot_geometry_from_xml(xml_path):
    """
    Extract just the slot geometry (no occupancy) from an XML file.
    Used to get the fixed slot layout for a parking lot.

    Parameters
    ----------
    xml_path : str
        Path to any XML file for this lot.

    Returns
    -------
    slot_polygons : dict
        Mapping of slot_id → np.ndarray of polygon points.
    """
    slots = parse_pklot_xml(xml_path)
    slot_polygons = {}
    for slot in slots:
        slot_polygons[slot['id']] = slot['points']
    return slot_polygons
