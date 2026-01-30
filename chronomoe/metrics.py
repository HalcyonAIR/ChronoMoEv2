# chronomoe/metrics.py
"""
Deterministic metric functions for ChronoMoE Phase 1.

All functions are pure - no state, deterministic, easy to test.
Given same inputs, produce identical outputs across runs.
"""

import numpy as np
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .events import RoutingEvent


def compute_utilization_shares(
    events: List['RoutingEvent'],
    layer_id: int
) -> np.ndarray:
    """
    Compute expert utilization shares for a layer across event window.

    Args:
        events: List of RoutingEvents to aggregate
        layer_id: Which layer to compute shares for

    Returns:
        Array of length n_experts, sums to 1.0 (or empty if no events)
    """
    layer_events = [e for e in events if e.layer_id == layer_id]
    if not layer_events:
        return np.array([])

    n_experts = layer_events[0].n_experts
    tokens_per_expert = np.zeros(n_experts)

    for event in layer_events:
        tokens_per_expert += np.array(event.expert_token_counts)

    total_tokens = tokens_per_expert.sum()
    if total_tokens == 0:
        return np.zeros(n_experts)

    shares = tokens_per_expert / total_tokens
    return shares


def compute_entropy(shares: np.ndarray, eps: float = 1e-10) -> float:
    """
    Shannon entropy over expert utilization.

    Args:
        shares: Utilization shares (should sum to 1.0)
        eps: Small constant to avoid log(0)

    Returns:
        Entropy value. Higher = more uniform distribution.
        For n experts with uniform distribution, H = log(n).
    """
    # Filter out zeros to avoid log(0)
    nonzero_shares = shares[shares > eps]
    if len(nonzero_shares) == 0:
        return 0.0
    return float(-np.sum(nonzero_shares * np.log(nonzero_shares)))


def compute_n_effective(entropy: float) -> float:
    """
    Effective number of experts from entropy.

    This is the exponential of Shannon entropy, giving an
    intuitive measure of "how many experts are really being used."

    Args:
        entropy: Shannon entropy value

    Returns:
        Effective expert count. For n balanced experts, returns n.
        For complete collapse to 1 expert, returns 1.
    """
    return float(np.exp(entropy))


def compute_top2_share(shares: np.ndarray) -> float:
    """
    Sum of two largest utilization shares.

    Early warning signal: if Top2Share approaches 1.0,
    the layer is collapsing to effectively 2 experts.

    Args:
        shares: Utilization shares

    Returns:
        Sum of top 2 shares. Values > 0.75 indicate pre-collapse.
    """
    if len(shares) < 2:
        return 1.0
    sorted_shares = np.sort(shares)[::-1]  # Descending
    return float(sorted_shares[0] + sorted_shares[1])


def count_dead_experts(shares: np.ndarray) -> int:
    """
    Count experts with ZERO utilization over the window.

    STRICT DEFINITION: share == 0, not share < threshold.
    This avoids mislabeling in small layers or short windows.
    For "nearly dead" tracking, use a separate metric.

    Args:
        shares: Utilization shares

    Returns:
        Count of experts with exactly zero share
    """
    return int(np.sum(shares == 0))


def count_nearly_dead_experts(shares: np.ndarray, threshold: float = 0.01) -> int:
    """
    Count experts with negligible (but non-zero) utilization.

    Separate from strict dead count for clarity.
    Use for early warning, not for collapse confirmation.

    Args:
        shares: Utilization shares
        threshold: Share below which expert is "nearly dead"

    Returns:
        Count of experts with 0 < share < threshold
    """
    return int(np.sum((shares > 0) & (shares < threshold)))


def compute_layer_metrics(
    events: List['RoutingEvent'],
    layer_id: int
) -> Tuple[np.ndarray, float, float, float, int, int]:
    """
    Compute all metrics for a layer from event window.

    Args:
        events: List of RoutingEvents
        layer_id: Which layer to compute metrics for

    Returns:
        Tuple of (shares, entropy, n_effective, top2_share, dead_count, nearly_dead_count)
    """
    shares = compute_utilization_shares(events, layer_id)

    if len(shares) == 0:
        return np.array([]), 0.0, 0.0, 0.0, 0, 0

    entropy = compute_entropy(shares)
    n_eff = compute_n_effective(entropy)
    top2 = compute_top2_share(shares)
    dead = count_dead_experts(shares)
    nearly_dead = count_nearly_dead_experts(shares)

    return shares, entropy, n_eff, top2, dead, nearly_dead


def compute_gini_coefficient(shares: np.ndarray) -> float:
    """
    Gini coefficient for utilization inequality.

    0 = perfect equality (all experts equal)
    1 = perfect inequality (one expert does everything)

    Optional metric for additional insight into imbalance.

    Args:
        shares: Utilization shares

    Returns:
        Gini coefficient between 0 and 1
    """
    if len(shares) == 0 or shares.sum() == 0:
        return 0.0

    # Sort shares
    sorted_shares = np.sort(shares)
    n = len(sorted_shares)
    cumulative = np.cumsum(sorted_shares)

    # Gini formula
    return float((2 * np.sum((np.arange(1, n + 1) * sorted_shares)) - (n + 1) * cumulative[-1]) / (n * cumulative[-1] + 1e-10))
