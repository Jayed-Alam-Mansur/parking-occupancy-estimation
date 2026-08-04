# pipeline.py - Pipeline Orchestrator
"""
End-to-end pipeline that processes a single frame through all stages.

Functions:
    run_pipeline()            - Process one frame → full report
    run_pipeline_batch()      - Process multiple frames
    setup_pipeline()          - Load configs, build masks (one-time)

This module ties together all other modules into a single callable
workflow: frame → undistort → warp → extract ROIs → preprocess →
segment → morphology → features → decide → stats → visualize.
"""

import cv2
import numpy as np
import os

from src.geometry import warp_perspective, load_homography, transform_points
from src.roi import (extract_slot_image, create_slot_mask,
                     create_eroded_core_mask, load_slots_json,
                     assign_slot_rows)
from src.preprocessing import preprocess_pipeline
from src.segmentation import (otsu_threshold, adaptive_threshold,
                               shadow_suppress_hsv, fuse_channels)
from src.morphology import clean_binary_mask
from src.features import extract_all_features
from src.decide import classify_all_slots, load_thresholds
from src.stats import compute_statistics, per_row_breakdown, format_report
from src.visualize import (annotate_parking_image, create_legend,
                           create_occupancy_map, create_dashboard)
from src.evaluate import Timer


class ParkingPipeline:
    """
    Encapsulates the full parking occupancy estimation pipeline.

    Usage:
        pipeline = ParkingPipeline('config/')
        result = pipeline.process_frame(image)
        print(result['report'])
    """

    def __init__(self, config_dir='config/', reference_bg=None):
        """
        Initialize the pipeline by loading all config artifacts.

        Parameters
        ----------
        config_dir : str
            Directory containing homography.npz, slots.json, thresholds.yaml.
        reference_bg : np.ndarray or None
            Reference empty-lot image (grayscale, in BEV).
        """
        self.config_dir = config_dir
        self.timer = Timer()

        # Load homography
        homography_path = os.path.join(config_dir, 'homography.npz')
        if os.path.exists(homography_path):
            hdata = load_homography(homography_path)
            self.H = hdata['H']
            self.output_size = hdata['output_size']
            self.px_per_metre = hdata.get('px_per_metre', None)
        else:
            self.H = None
            self.output_size = None
            self.px_per_metre = None

        # Load slot definitions
        slots_path = os.path.join(config_dir, 'slots.json')
        if os.path.exists(slots_path):
            self.slots = load_slots_json(slots_path)
        else:
            self.slots = {}

        # Load thresholds
        thresholds_path = os.path.join(config_dir, 'thresholds.yaml')
        self.thresholds, self.weights = load_thresholds(thresholds_path)

        # Precompute row assignments
        self.rows = assign_slot_rows(self.slots) if self.slots else {}

        # Reference background
        self.reference_bg = reference_bg

    def process_frame(self, image, return_intermediates=False):
        """
        Process a single frame through the complete pipeline.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image (original camera view).
        return_intermediates : bool
            If True, include per-slot intermediate images.

        Returns
        -------
        result : dict
            Contains:
                'labels'     : dict of slot_id → 0/1
                'confidences': dict of slot_id → float
                'scores'     : dict of slot_id → float
                'stats'      : overall statistics dict
                'row_stats'  : per-row statistics dict
                'report'     : formatted text report
                'annotated'  : annotated BEV image
                'bev'        : bird's-eye view image
                'timing'     : per-stage timing dict
        """
        # Stage 1: Warp to BEV
        with self.timer.measure('BEV Warp'):
            if self.H is not None:
                bev = warp_perspective(image, self.H, self.output_size)
            else:
                bev = image.copy()

        # Stage 2-9: Per-slot processing
        all_features = {}

        with self.timer.measure('Per-Slot Processing'):
            for slot_id, polygon in self.slots.items():
                # Extract slot ROI
                slot_img, bbox, mask = extract_slot_image(bev, polygon)

                if slot_img.size == 0:
                    continue

                # Create core mask (within bbox coordinates)
                core_mask = create_eroded_core_mask(mask, erosion_px=3)

                # Preprocess
                preprocessed = preprocess_pipeline(slot_img)

                # Segment (multi-channel)
                otsu_binary, _, _ = otsu_threshold(preprocessed)
                adapt_binary = adaptive_threshold(preprocessed)
                fused = fuse_channels(otsu_binary, adapt_binary)

                # Morphology
                cleaned = clean_binary_mask(fused)

                # Features
                features = extract_all_features(
                    preprocessed, cleaned,
                    bgr_image=slot_img,
                    mask=core_mask
                )

                all_features[slot_id] = features

        # Stage 10: Classification
        with self.timer.measure('Classification'):
            results = classify_all_slots(
                all_features, self.thresholds, self.weights
            )

        labels = {sid: r['label'] for sid, r in results.items()}
        confidences = {sid: r['confidence'] for sid, r in results.items()}
        scores = {sid: r['score'] for sid, r in results.items()}

        # Stage 11: Statistics
        with self.timer.measure('Statistics'):
            stats = compute_statistics(labels)
            row_stats = per_row_breakdown(labels, self.rows)
            report = format_report(stats, row_stats)

        # Stage 12: Visualization
        with self.timer.measure('Visualization'):
            annotated = annotate_parking_image(
                bev, self.slots, labels,
                confidences=confidences, scores=scores
            )
            annotated = create_legend(annotated, stats)

        result = {
            'labels': labels,
            'confidences': confidences,
            'scores': scores,
            'stats': stats,
            'row_stats': row_stats,
            'report': report,
            'annotated': annotated,
            'bev': bev,
            'timing': self.timer.get_mean_times(),
        }

        if return_intermediates:
            result['all_features'] = all_features

        return result


def run_pipeline(image, config_dir='config/', reference_bg=None):
    """
    Convenience function: process one frame.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    config_dir : str
        Config directory path.
    reference_bg : np.ndarray or None
        Reference background.

    Returns
    -------
    result : dict
        Pipeline results.
    """
    pipeline = ParkingPipeline(config_dir, reference_bg)
    return pipeline.process_frame(image)
