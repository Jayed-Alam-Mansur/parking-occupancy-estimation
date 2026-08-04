# visualize.py - Drawing, Annotation & Dashboard
"""
Visualization functions for presenting results.

Functions:
    annotate_parking_image()  - Draw colored overlays on parking slots
    create_legend()           - Add legend banner to annotated image
    create_occupancy_map()    - Schematic grid abstraction
    create_dashboard()        - Statistics dashboard figure
    create_pipeline_figure()  - Show processing steps side-by-side

Color Convention:
    Green (0, 255, 0) = VACANT
    Red   (0, 0, 255) = OCCUPIED (in BGR)
"""

import cv2
import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Reusable helper: display inline + save to disk
# ---------------------------------------------------------------------------

def show_and_save_fig(content, title, filename, figsize=(12, 8), dpi=150,
                      cmap=None, save_dir='outputs/screenshots'):
    """
    Display a figure or image inline in Jupyter and save to disk.

    Handles both pre-built matplotlib Figures and raw numpy arrays
    (BGR or grayscale).  BGR images are automatically converted to RGB.

    Parameters
    ----------
    content : matplotlib.figure.Figure  or  np.ndarray
        • Figure  – save and show the existing figure.
        • ndarray – create a new single-panel figure and display it.
    title : str
        Displayed as suptitle (Figure) or axis title (ndarray).
    filename : str
        File name under *save_dir*, e.g. ``'08_preprocessing.png'``.
    figsize : tuple
        Only used when *content* is an ndarray.
    dpi : int
        Resolution for the saved PNG.
    cmap : str or None
        Colour map for single-channel images (default ``'gray'``).
    save_dir : str
        Directory to save to (created if it doesn't exist).

    Returns
    -------
    None
        Nothing is returned on purpose: in Jupyter, returning the Figure
        makes it the cell's result value, so it renders a *second* time
        underneath the one already shown by ``plt.show()``.
    """
    if filename:
        save_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_path = None

    if isinstance(content, plt.Figure):
        fig = content
        if title:
            # Place the suptitle and reserve room for it in absolute inches,
            # so tall figures don't have it collide with the first row of
            # axis titles (tight_layout alone ignores the suptitle).
            h = fig.get_figheight()
            fig.suptitle(title, fontsize=14, fontweight='bold',
                         y=1.0 - 0.12 / h)
            fig.tight_layout(rect=[0, 0, 1, 1.0 - 0.45 / h])
        else:
            fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
    else:
        # numpy array (OpenCV image)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if content.ndim == 3 and content.shape[2] == 3:
            ax.imshow(cv2.cvtColor(content, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(content, cmap=cmap or 'gray')
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()

    plt.close(fig)

    if save_path:
        print(f"Saved: {save_path}")


def annotate_parking_image(image, slots, labels, confidences=None,
                           alpha=0.35, show_scores=False, scores=None):
    """
    Draw colored semi-transparent overlays on each parking slot.

    Parameters
    ----------
    image : np.ndarray
        Original parking lot image (BGR).
    slots : dict
        slot_id → polygon (Nx2 array).
    labels : dict
        slot_id → 0 (VACANT) or 1 (OCCUPIED).
    confidences : dict or None
        slot_id → confidence (float).
    alpha : float
        Overlay transparency (0.0=transparent, 1.0=opaque).
    show_scores : bool
        Whether to show numerical scores on each slot.
    scores : dict or None
        slot_id → raw score (float).

    Returns
    -------
    annotated : np.ndarray
        Image with colored slot overlays and labels.
    """
    annotated = image.copy()
    overlay = image.copy()

    vacant_color = (0, 255, 0)    # Green
    occupied_color = (0, 0, 255)  # Red

    for slot_id, polygon in slots.items():
        pts = np.array(polygon, dtype=np.int32)
        label = labels.get(slot_id, 0)
        color = occupied_color if label == 1 else vacant_color

        # Draw filled polygon on overlay
        cv2.fillPoly(overlay, [pts], color)

        # Draw outline
        cv2.polylines(annotated, [pts], isClosed=True,
                      color=color, thickness=2)

    # Blend overlay
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

    # Draw slot IDs and scores
    for slot_id, polygon in slots.items():
        pts = np.array(polygon, dtype=np.int32)
        centroid = pts.mean(axis=0).astype(int)

        if show_scores and scores and slot_id in scores:
            text = f"{slot_id}:{scores[slot_id]:.2f}"
        else:
            text = str(slot_id)

        # White text with dark outline for readability
        cv2.putText(annotated, text, tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, text, tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return annotated


def create_legend(image, stats, position='top'):
    """
    Add a legend banner to the annotated image.

    Parameters
    ----------
    image : np.ndarray
        Annotated parking image.
    stats : dict
        Statistics dictionary.
    position : str
        'top' or 'bottom'.

    Returns
    -------
    with_legend : np.ndarray
        Image with legend banner.
    """
    h, w = image.shape[:2]
    banner_height = 60

    # Create banner
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner[:] = (40, 40, 40)  # Dark grey background

    # Add text
    y_text = 20

    # Legend colours
    cv2.rectangle(banner, (10, 8), (30, 28), (0, 255, 0), -1)
    cv2.putText(banner, "VACANT", (35, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(banner, (130, 8), (150, 28), (0, 0, 255), -1)
    cv2.putText(banner, "OCCUPIED", (155, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Statistics
    stats_text = (
        f"Total: {stats['total_spaces']}  |  "
        f"Occupied: {stats['occupied']}  |  "
        f"Vacant: {stats['vacant']}  |  "
        f"Rate: {stats['occupancy_pct']}%"
    )
    cv2.putText(banner, stats_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Concatenate
    if position == 'top':
        with_legend = np.vstack([banner, image])
    else:
        with_legend = np.vstack([image, banner])

    return with_legend


def create_occupancy_map(slots, labels, figsize=(10, 8)):
    """
    Create a schematic grid abstraction of the parking lot.

    Parameters
    ----------
    slots : dict
        slot_id → polygon (Nx2 array).
    labels : dict
        slot_id → 0 or 1.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for slot_id, polygon in slots.items():
        pts = polygon[:, :2]  # Ensure 2D
        label = labels.get(slot_id, 0)
        color = '#2ecc71' if label == 0 else '#e74c3c'
        edge_color = '#27ae60' if label == 0 else '#c0392b'

        # Draw filled polygon
        poly_patch = plt.Polygon(pts, closed=True,
                                  facecolor=color, edgecolor=edge_color,
                                  linewidth=1.5, alpha=0.7)
        ax.add_patch(poly_patch)

        # Label
        centroid = pts.mean(axis=0)
        ax.text(centroid[0], centroid[1], str(slot_id),
                ha='center', va='center', fontsize=7,
                fontweight='bold', color='white')

    # Legend
    vacant_patch = mpatches.Patch(color='#2ecc71', label='Vacant')
    occupied_patch = mpatches.Patch(color='#e74c3c', label='Occupied')
    ax.legend(handles=[vacant_patch, occupied_patch],
              loc='upper right', fontsize=10)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('Parking Lot Occupancy Map', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    # Auto-scale
    all_pts = np.vstack(list(slots.values()))
    margin = 20
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].max() + margin, all_pts[:, 1].min() - margin)

    plt.tight_layout()
    return fig


def create_dashboard(stats, row_stats=None, figsize=(14, 8)):
    """
    Create a comprehensive statistics dashboard figure.

    Parameters
    ----------
    stats : dict
        Overall statistics.
    row_stats : dict or None
        Per-row breakdown.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Parking Lot Occupancy Dashboard', fontsize=16,
                 fontweight='bold', y=0.98)

    # 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # --- Pie chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = [stats['occupied'], stats['vacant']]
    colors = ['#e74c3c', '#2ecc71']
    explode = (0.05, 0)
    labels_pie = [f"Occupied\n({stats['occupied']})",
                  f"Vacant\n({stats['vacant']})"]

    ax1.pie(sizes, explode=explode, labels=labels_pie, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 10})
    ax1.set_title('Occupancy Distribution', fontsize=12)

    # --- Summary text ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    summary_lines = [
        f"Total Spaces:     {stats['total_spaces']}",
        f"Occupied:         {stats['occupied']}",
        f"Vacant:           {stats['vacant']}",
        f"Occupancy Rate:   {stats['occupancy_pct']}%",
    ]

    bar_width = 30
    filled = int(bar_width * stats['occupancy_pct'] / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    summary_lines.append(f"\n[{bar}]")

    ax2.text(0.1, 0.9, '\n'.join(summary_lines),
             transform=ax2.transAxes, fontsize=13,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray',
                       alpha=0.3))
    ax2.set_title('Summary', fontsize=12)

    # --- Bar chart (occupancy by row) ---
    ax3 = fig.add_subplot(gs[1, 0])
    if row_stats:
        rows_sorted = sorted(row_stats.keys())
        row_labels = [f"Row {r}" for r in rows_sorted]
        occ_counts = [row_stats[r]['occupied'] for r in rows_sorted]
        vac_counts = [row_stats[r]['vacant'] for r in rows_sorted]

        x = np.arange(len(row_labels))
        width = 0.35

        ax3.bar(x - width/2, occ_counts, width, label='Occupied',
                color='#e74c3c', alpha=0.8)
        ax3.bar(x + width/2, vac_counts, width, label='Vacant',
                color='#2ecc71', alpha=0.8)

        ax3.set_xlabel('Row')
        ax3.set_ylabel('Count')
        ax3.set_title('Per-Row Breakdown', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(row_labels, rotation=45)
        ax3.legend()
    else:
        ax3.bar(['Occupied', 'Vacant'],
                [stats['occupied'], stats['vacant']],
                color=['#e74c3c', '#2ecc71'], alpha=0.8)
        ax3.set_ylabel('Count')
        ax3.set_title('Overall Count', fontsize=12)

    # --- Gauge-style occupancy meter ---
    ax4 = fig.add_subplot(gs[1, 1])
    pct = stats['occupancy_pct']

    # Simple horizontal gauge
    ax4.barh([0], [pct], height=0.4, color='#e74c3c', alpha=0.8)
    ax4.barh([0], [100 - pct], left=[pct], height=0.4,
             color='#2ecc71', alpha=0.8)

    ax4.set_xlim(0, 100)
    ax4.set_yticks([])
    ax4.set_xlabel('Percentage')
    ax4.set_title(f'Occupancy Rate: {pct}%', fontsize=12, fontweight='bold')

    # Add percentage markers
    for x_val in [25, 50, 75]:
        ax4.axvline(x=x_val, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig


def create_pipeline_figure(stages, titles=None, cols=3, figsize=(15, 10)):
    """
    Create a side-by-side figure showing each processing stage.

    Parameters
    ----------
    stages : list of np.ndarray
        Images at each processing step.
    titles : list of str or None
        Title for each stage. If None, uses "Stage 1", "Stage 2", etc.
    cols : int
        Number of columns in the grid.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n = len(stages)
    if titles is None:
        titles = [f"Stage {i+1}" for i in range(n)]

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx, (img, title) in enumerate(zip(stages, titles)):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axis('off')

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    fig.suptitle('Processing Pipeline Stages', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
