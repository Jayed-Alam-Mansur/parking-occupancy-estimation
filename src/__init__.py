# =============================================================================
# Automatic Parking Occupancy Estimation
# Classical Image Processing Project
# =============================================================================
# src/ package initializer
# =============================================================================

"""
Parking Occupancy Estimation - Source Package

This package contains modular Python functions for each stage of the
classical image processing pipeline:

Modules:
    io_utils       - Frame loading, PKLot XML parsing, quality gate
    geometry       - Camera geometry & perspective transformation
    roi            - Region of Interest extraction & slot management
    preprocessing  - Image preprocessing (grayscale, CLAHE, blur, denoise)
    segmentation   - Thresholding methods (global, adaptive, Otsu, ref-diff)
    morphology     - Morphological operations (erosion, dilation, opening, closing)
    features       - Feature extraction (8 features + Fisher ratio)
    decide         - Rule-based occupancy decision engine
    stats          - Occupancy statistics & reporting
    visualize      - Drawing, annotation, dashboard generation
    evaluate       - Performance metrics (accuracy, precision, recall, F1)
    pipeline       - End-to-end pipeline orchestrator
    utils          - Utility functions (config loading, display helpers)
"""

__version__ = "1.0.0"
__author__ = "Image Processing Course Project"
