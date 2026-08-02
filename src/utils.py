# =============================================================================
# utils.py - Utility Functions
# =============================================================================
"""
Common utility functions used across all modules.

Functions:
    load_config()             - Load config.yaml
    load_image()              - Load an image from disk
    save_image()              - Save an image to disk
    resize_image()            - Resize image maintaining aspect ratio
    display_images()          - Display multiple images in a grid
    print_separator()         - Print a formatted section separator
"""

import cv2
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt


def load_config(config_path="config/config.yaml"):
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.

    Returns
    -------
    config : dict
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_image(image_path):
    """
    Load an image from disk with error handling.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    image : np.ndarray
        Loaded image in BGR format.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If the image cannot be read.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


def save_image(image, save_path):
    """
    Save an image to disk, creating directories if needed.

    Parameters
    ----------
    image : np.ndarray
        Image to save.
    save_path : str
        Output file path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)


def display_images(images, titles, figsize=(15, 5), cmap=None, cols=None):
    """
    Display multiple images in a horizontal grid.

    Parameters
    ----------
    images : list of np.ndarray
        Images to display.
    titles : list of str
        Title for each image.
    figsize : tuple
        Figure size (width, height).
    cmap : str, optional
        Colormap for grayscale images (e.g., 'gray').
    cols : int, optional
        Number of columns. Defaults to len(images).
    """
    n = len(images)
    if cols is None:
        cols = n
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle single row/column case
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        if len(img.shape) == 2:
            # Grayscale image
            ax.imshow(img, cmap=cmap or 'gray')
        else:
            # Color image: convert BGR to RGB for matplotlib
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.show()


def resize_image(image, max_width=800):
    """
    Resize image maintaining aspect ratio.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    max_width : int
        Maximum width in pixels.

    Returns
    -------
    resized : np.ndarray
        Resized image.
    """
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()

    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def print_separator(title, char="=", width=60):
    """Print a formatted section separator."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")
