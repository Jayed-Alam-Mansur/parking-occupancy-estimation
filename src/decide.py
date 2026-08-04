# decide.py - Rule-Based Occupancy Decision Engine
"""
Classifies parking slots as OCCUPIED or VACANT using rule-based logic.

Functions:
    cascade_fast_path()       - Quick reject/accept on extreme features
    weighted_score()          - Compute weighted occupancy score
    classify_slot()           - Full classification for one slot
    classify_all_slots()      - Classify all slots
    apply_hysteresis()        - Temporal stability over multiple frames
    neighbour_refinement()    - Spatial consistency check

Theory:
    This is NOT machine learning. It is a transparent, interpretable
    decision rule based on handcrafted features and human-selected
    thresholds.

    Decision cascade:
        1. FAST PATH: if edge_density > τ_high → OCCUPIED (no doubt)
                     if edge_density < τ_low  → VACANT  (no doubt)
        2. WEIGHTED SCORE: for ambiguous cases, compute:
              S_i = Σ(w_k · f_k) / Σ(w_k)
           where f_k are normalised features and w_k are weights.
        3. THRESHOLD: S_i > τ → OCCUPIED, else VACANT
        4. HYSTERESIS: require m consecutive agreeing frames before
           flipping state (prevents flicker from clouds/pedestrians)
        5. NEIGHBOUR REFINEMENT: if a slot disagrees with all neighbours,
           and its confidence is low, reconsider.

    The thresholds are justified by inspecting feature histograms
    (Phase 9) and choosing values that separate the two class
    distributions. This is parameter selection, not model training.

IMPORTANT:
    No sklearn classifiers, no neural networks, no trained models.
    Pure conditional logic only.
"""

import numpy as np
import yaml
import os


# Default weights reflecting each feature's discriminative power
# Edge density gets highest weight (strongest single discriminator)
DEFAULT_WEIGHTS = {
    'edge_density':       0.30,
    'foreground_ratio':   0.15,
    'gradient_magnitude': 0.15,
    'local_variance':     0.10,
    'largest_component':  0.10,
    'intensity_std':      0.08,
    'otsu_separability':  0.07,
    'mean_saturation':    0.05,
}

# Default thresholds (will be refined in Phase 9-10)
DEFAULT_THRESHOLDS = {
    'edge_density_high': 0.15,   # Fast-path OCCUPIED
    'edge_density_low':  0.02,   # Fast-path VACANT
    'score_threshold':   0.35,   # Weighted score decision boundary
    'confidence_low':    0.2,    # Below this → uncertain
}


def cascade_fast_path(features, thresholds=None):
    """
    Quick reject/accept for obvious cases based on edge density.

    Parameters
    ----------
    features : dict
        Feature dictionary from features.extract_all_features().
    thresholds : dict or None
        Must contain 'edge_density_high' and 'edge_density_low'.

    Returns
    -------
    decision : int or None
        1 = OCCUPIED, 0 = VACANT, None = ambiguous (needs scoring).
    confidence : float
        Confidence in the fast-path decision (0.0 to 1.0).
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    ed = features['edge_density']
    t_high = thresholds['edge_density_high']
    t_low = thresholds['edge_density_low']

    if ed >= t_high:
        # Very high edge density → definitely occupied
        confidence = min(1.0, ed / t_high)
        return 1, confidence

    if ed <= t_low:
        # Very low edge density → definitely vacant
        confidence = min(1.0, (t_low - ed) / t_low + 0.5)
        return 0, confidence

    return None, 0.0  # Ambiguous → needs full scoring


def weighted_score(features, weights=None):
    """
    Compute weighted occupancy score from normalised features.

    Parameters
    ----------
    features : dict
        Feature dictionary (values already normalised to 0-1).
    weights : dict or None
        Per-feature weights. If None, uses defaults.

    Returns
    -------
    score : float
        Weighted score (0.0 to 1.0). Higher = more likely occupied.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    total_score = 0.0
    total_weight = 0.0

    for feature_name, w in weights.items():
        if feature_name in features:
            total_score += w * features[feature_name]
            total_weight += w

    if total_weight == 0:
        return 0.0

    return total_score / total_weight


def classify_slot(features, thresholds=None, weights=None):
    """
    Classify a single parking slot as OCCUPIED (1) or VACANT (0).

    Uses the cascade: fast-path → weighted score → threshold.

    Parameters
    ----------
    features : dict
        Feature dictionary from features.extract_all_features().
    thresholds : dict or None
        Decision thresholds.
    weights : dict or None
        Feature weights.

    Returns
    -------
    label : int
        1 = OCCUPIED, 0 = VACANT.
    confidence : float
        Confidence score (0.0 to 1.0).
    score : float
        Raw weighted score.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Step 1: Fast path
    fast_decision, fast_conf = cascade_fast_path(features, thresholds)
    if fast_decision is not None:
        return fast_decision, fast_conf, features['edge_density']

    # Step 2: Weighted score
    score = weighted_score(features, weights)

    # Step 3: Threshold
    tau = thresholds.get('score_threshold', 0.35)
    label = 1 if score > tau else 0

    # Confidence = distance from decision boundary
    distance = abs(score - tau) / max(tau, 1.0 - tau)
    confidence = min(1.0, distance)

    return label, confidence, score


def classify_all_slots(all_features, thresholds=None, weights=None):
    """
    Classify all parking slots.

    Parameters
    ----------
    all_features : dict
        Mapping of slot_id → feature dict.
    thresholds : dict or None
        Decision thresholds.
    weights : dict or None
        Feature weights.

    Returns
    -------
    results : dict
        Mapping of slot_id → {
            'label': int,
            'confidence': float,
            'score': float
        }
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if weights is None:
        weights = DEFAULT_WEIGHTS

    results = {}
    for slot_id, features in all_features.items():
        label, confidence, score = classify_slot(
            features, thresholds, weights
        )
        results[slot_id] = {
            'label': label,
            'confidence': confidence,
            'score': score
        }

    return results


def apply_hysteresis(current_states, previous_states, m=3):
    """
    Apply temporal hysteresis to prevent state flicker.

    A slot's state only changes if the new state persists for
    m consecutive frames.

    Parameters
    ----------
    current_states : dict
        slot_id → current frame label (0 or 1).
    previous_states : dict
        slot_id → {
            'label': int,           # confirmed state
            'pending_label': int,   # candidate new state
            'count': int            # consecutive frames with pending
        }
    m : int
        Number of consecutive frames required for state change.

    Returns
    -------
    updated_states : dict
        Updated state tracking dictionary.
    confirmed_labels : dict
        slot_id → confirmed label (stable output).
    """
    updated = {}
    confirmed = {}

    for slot_id, current_label in current_states.items():
        prev = previous_states.get(slot_id, {
            'label': current_label,
            'pending_label': current_label,
            'count': 0
        })

        if current_label == prev['label']:
            # No change — reset pending
            updated[slot_id] = {
                'label': prev['label'],
                'pending_label': prev['label'],
                'count': 0
            }
        elif current_label == prev.get('pending_label', -1):
            # Same pending label as last frame — increment
            count = prev.get('count', 0) + 1
            if count >= m:
                # Enough consecutive frames → confirm change
                updated[slot_id] = {
                    'label': current_label,
                    'pending_label': current_label,
                    'count': 0
                }
            else:
                updated[slot_id] = {
                    'label': prev['label'],
                    'pending_label': current_label,
                    'count': count
                }
        else:
            # Different pending label — reset
            updated[slot_id] = {
                'label': prev['label'],
                'pending_label': current_label,
                'count': 1
            }

        confirmed[slot_id] = updated[slot_id]['label']

    return updated, confirmed


def neighbour_refinement(labels, confidences, slots, distance_threshold=50.0):
    """
    Spatial consistency check: if a slot disagrees with ALL neighbours
    and has low confidence, flip it.

    Parameters
    ----------
    labels : dict
        slot_id → label (0 or 1).
    confidences : dict
        slot_id → confidence (float).
    slots : dict
        slot_id → polygon (Nx2 array) for computing distances.
    distance_threshold : float
        Maximum distance (in pixels) to consider as neighbour.

    Returns
    -------
    refined_labels : dict
        Potentially corrected labels.
    flipped : list
        Slot IDs that were flipped.
    """
    # Compute centroids
    centroids = {}
    for sid, poly in slots.items():
        centroids[sid] = np.mean(poly, axis=0)

    # Find neighbours for each slot
    slot_ids = list(labels.keys())
    refined = dict(labels)
    flipped = []

    for sid in slot_ids:
        if sid not in centroids:
            continue

        c = centroids[sid]
        my_label = labels[sid]
        my_conf = confidences.get(sid, 1.0)

        # Skip high-confidence decisions
        if my_conf > 0.5:
            continue

        # Find neighbours
        neighbours = []
        for other_sid in slot_ids:
            if other_sid == sid or other_sid not in centroids:
                continue
            dist = np.linalg.norm(c - centroids[other_sid])
            if dist < distance_threshold:
                neighbours.append(labels[other_sid])

        if len(neighbours) < 2:
            continue

        # If ALL neighbours disagree, flip
        neighbour_consensus = sum(neighbours) / len(neighbours)
        if (my_label == 1 and neighbour_consensus < 0.3) or \
           (my_label == 0 and neighbour_consensus > 0.7):
            refined[sid] = 1 - my_label
            flipped.append(sid)

    return refined, flipped


def load_thresholds(filepath='config/thresholds.yaml'):
    """
    Load decision thresholds from YAML file.

    Parameters
    ----------
    filepath : str
        Path to thresholds.yaml.

    Returns
    -------
    thresholds : dict
        Threshold dictionary.
    weights : dict
        Feature weights dictionary.
    """
    if not os.path.exists(filepath):
        return DEFAULT_THRESHOLDS.copy(), DEFAULT_WEIGHTS.copy()

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    thresholds = data.get('thresholds', DEFAULT_THRESHOLDS)
    weights = data.get('weights', DEFAULT_WEIGHTS)
    return thresholds, weights


def save_thresholds(filepath, thresholds, weights):
    """
    Save decision thresholds and weights to YAML file.

    Parameters
    ----------
    filepath : str
        Output path.
    thresholds : dict
        Threshold dictionary.
    weights : dict
        Feature weights dictionary.
    """
    def _to_python(d):
        """Convert numpy scalars to plain Python types for YAML."""
        return {k: float(v) if hasattr(v, 'item') else v
                for k, v in d.items()}

    data = {
        'thresholds': _to_python(thresholds),
        'weights': _to_python(weights)
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
