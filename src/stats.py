# =============================================================================
# stats.py - Occupancy Statistics
# =============================================================================
"""
Compute and format occupancy statistics.

Functions:
    compute_statistics()      - N, N_occ, N_vac, O%
    per_row_breakdown()       - Statistics per parking row
    format_report()           - Human-readable text report
"""

import numpy as np


def compute_statistics(labels):
    """
    Compute the four required output statistics.

    Parameters
    ----------
    labels : dict
        Mapping of slot_id → label (0=vacant, 1=occupied).

    Returns
    -------
    stats : dict
        Dictionary with keys:
            total_spaces   : int  - N
            occupied       : int  - N_occ
            vacant         : int  - N_vac
            occupancy_pct  : float - O% = (N_occ / N) × 100
    """
    total = len(labels)
    occupied = sum(1 for v in labels.values() if v == 1)
    vacant = total - occupied

    occupancy_pct = (occupied / total * 100) if total > 0 else 0.0

    return {
        'total_spaces': total,
        'occupied': occupied,
        'vacant': vacant,
        'occupancy_pct': round(occupancy_pct, 1)
    }


def per_row_breakdown(labels, rows):
    """
    Compute statistics per parking row.

    Parameters
    ----------
    labels : dict
        Mapping of slot_id → label (0 or 1).
    rows : dict
        Mapping of row_index → list of slot_ids.

    Returns
    -------
    row_stats : dict
        Mapping of row_index → {total, occupied, vacant, occupancy_pct}
    """
    row_stats = {}
    for row_idx, slot_ids in rows.items():
        row_labels = {sid: labels.get(sid, 0) for sid in slot_ids}
        row_stats[row_idx] = compute_statistics(row_labels)

    return row_stats


def format_report(stats, row_stats=None, confidence_stats=None):
    """
    Format a human-readable occupancy report.

    Parameters
    ----------
    stats : dict
        Overall statistics from compute_statistics().
    row_stats : dict or None
        Per-row breakdown.
    confidence_stats : dict or None
        Confidence distribution info.

    Returns
    -------
    report : str
        Formatted text report.
    """
    lines = [
        "=" * 60,
        "  PARKING LOT OCCUPANCY REPORT",
        "=" * 60,
        "",
        f"  Total Parking Spaces:  {stats['total_spaces']}",
        f"  Occupied:              {stats['occupied']}",
        f"  Vacant:                {stats['vacant']}",
        f"  Occupancy Rate:        {stats['occupancy_pct']}%",
        "",
    ]

    # Visual bar
    if stats['total_spaces'] > 0:
        bar_width = 40
        filled = int(bar_width * stats['occupancy_pct'] / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(f"  [{bar}] {stats['occupancy_pct']}%")
        lines.append("")

    # Per-row breakdown
    if row_stats:
        lines.append("  Per-Row Breakdown:")
        lines.append("  " + "-" * 50)
        lines.append(f"  {'Row':<6} {'Total':<8} {'Occ':<6} {'Vac':<6} {'Rate':<8}")
        lines.append("  " + "-" * 50)

        for row_idx in sorted(row_stats.keys()):
            rs = row_stats[row_idx]
            lines.append(
                f"  {row_idx:<6} {rs['total_spaces']:<8} "
                f"{rs['occupied']:<6} {rs['vacant']:<6} "
                f"{rs['occupancy_pct']:<8.1f}%"
            )
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
