# csp.py — Wheelchair accessibility routing via Constraint Satisfaction
# INFO-H410 — Wheelchair Routing CSP — ULB
#
# Approach: hard constraints.
# Streets that violate accessibility thresholds are removed from the graph
# before any path search. Dijkstra then runs on the filtered graph.
# If no path exists, the system retries with relaxed constraints.
# This contrasts with the soft-constraint approach in astar.py, where
# streets are penalised but never fully removed.

import time
import copy
import networkx as nx
import osmnx as ox

from config import (
    ABSOLUTE_CONSTRAINTS,
    STRICT_CONSTRAINTS,
    RELAXED_CONSTRAINTS,
    WEIGHTS,
    CONFIDENCE_LEVELS,
    DEFAULTS,
)


def build_feasible_graph(G, strict=True):
    """
    Removes edges that violate accessibility constraints and returns
    a filtered copy of the graph. This is the CSP filtering step.

    With strict=True, applies STRICT_CONSTRAINTS (slope <= 8%, width >= 1.2m).
    With strict=False, applies RELAXED_CONSTRAINTS (slope <= 15%, width >= 1.0m).
    Absolute constraints (kerb != unknown, wheelchair != no) are always applied.

    Returns the filtered graph and a dict with removal statistics.
    """
    constraints = STRICT_CONSTRAINTS if strict else RELAXED_CONSTRAINTS

    edges_total   = G.number_of_edges()
    edges_removed = 0

    H = copy.deepcopy(G)

    edges_to_remove = []
    for u, v, key, data in H.edges(keys=True, data=True):

        # Absolute constraints — never relaxed
        wheelchair = data.get('wheelchair', DEFAULTS['wheelchair'])

        if not ABSOLUTE_CONSTRAINTS['wheelchair'](wheelchair):
            edges_to_remove.append((u, v, key))
            continue

        # Slope and width constraints
        slope = data.get('slope', DEFAULTS['slope'])
        width = data.get('width', DEFAULTS['width'])

        if slope > constraints['slope']:
            edges_to_remove.append((u, v, key))
            continue

        if width < constraints['width']:
            edges_to_remove.append((u, v, key))
            continue

    for edge in edges_to_remove:
        H.remove_edge(*edge)
        edges_removed += 1

    stats = {
        'edges_total':   edges_total,
        'edges_removed': edges_removed,
        'edges_kept':    edges_total - edges_removed,
        'removal_pct':   round(edges_removed / edges_total * 100, 1) if edges_total > 0 else 0,
        'strict':        strict,
    }

    return H, stats


def find_path(G, point_A, point_B):
    """
    Finds the shortest path between two GPS points on a filtered graph.
    Maps coordinates to the nearest graph nodes, then runs Dijkstra
    weighted by street length.

    Returns a dict with path, cost, and runtime. If no path exists,
    path is None and cost is infinity.
    """
    origin = ox.distance.nearest_nodes(G, X=point_A[1], Y=point_A[0])
    goal   = ox.distance.nearest_nodes(G, X=point_B[1], Y=point_B[0])

    t0 = time.perf_counter()

    try:
        path = nx.shortest_path(G, origin, goal, weight='length')
        cost = nx.shortest_path_length(G, origin, goal, weight='length')
    except nx.NetworkXNoPath:
        path = None
        cost = float('inf')
    except nx.NodeNotFound:
        path = None
        cost = float('inf')

    return {
        'path':      path,
        'cost':      cost,
        'origin':    origin,
        'goal':      goal,
        'runtime_s': round(time.perf_counter() - t0, 4),
    }


def compute_confidence(path, G):
    """
    Estimates how reliable the found path is, based on how many
    accessibility attributes are real OSM data vs default estimates.

    Each attribute has a weight defined in config.WEIGHTS.
    An attribute counts as "known" if its value differs from the default.
    The score is the weighted average of known attributes across all edges.

    Returns a float between 0 and 100, and a human-readable label.
    """
    if path is None or len(path) < 2:
        return 0.0, _confidence_label(0.0)

    total_weight  = sum(WEIGHTS.values())
    score_sum     = 0.0
    edge_count    = 0

    for u, v in zip(path[:-1], path[1:]):
        edge_data = _best_edge(G, u, v)
        if edge_data is None:
            continue

        edge_score = 0.0
        for attr, weight in WEIGHTS.items():
            value   = edge_data.get(attr)
            default = DEFAULTS.get(attr)
            if value is not None and value != default:
                edge_score += weight

        score_sum  += edge_score / total_weight * 100
        edge_count += 1

    if edge_count == 0:
        return 0.0, _confidence_label(0.0)

    score = round(score_sum / edge_count, 1)
    return score, _confidence_label(score)


def solve(G, point_A, point_B):
    """
    Main entry point for the CSP routing.

    Attempts to find a path in this order:
      1. Filtered graph with strict constraints
      2. Filtered graph with relaxed constraints

    Returns a dict with all relevant information for display and evaluation.
    Keys:
      path         — list of node ids, or None
      cost         — total length in metres
      confidence   — score 0-100
      label        — human-readable confidence string
      mode         — 'strict', 'relaxed', or 'no_path'
      csp_stats    — removal statistics from build_feasible_graph
      runtime_s    — total time in seconds
    """
    t_start = time.perf_counter()

    for strict in [True, False]:
        mode = 'strict' if strict else 'relaxed'

        H, csp_stats = build_feasible_graph(G, strict=strict)
        result       = find_path(H, point_A, point_B)

        if result['path'] is not None:
            confidence, label = compute_confidence(result['path'], H)
            return {
                'path':       result['path'],
                'cost':       result['cost'],
                'origin':     result['origin'],
                'goal':       result['goal'],
                'confidence': confidence,
                'label':      label,
                'mode':       mode,
                'csp_stats':  csp_stats,
                'runtime_s':  round(time.perf_counter() - t_start, 4),
            }

    return {
        'path':       None,
        'cost':       float('inf'),
        'origin':     None,
        'goal':       None,
        'confidence': 0.0,
        'label':      'No accessible path found',
        'mode':       'no_path',
        'csp_stats':  csp_stats,
        'runtime_s':  round(time.perf_counter() - t_start, 4),
    }


def path_metrics(path, G):
    """
    Computes summary statistics for a path. Used by evaluate.py
    to fill the comparison table in the report.

    Returns a dict with length_m, n_edges, max_slope, confidence.
    """
    if path is None or len(path) < 2:
        return {'length_m': 0.0, 'n_edges': 0, 'max_slope': 0.0, 'confidence': 0.0}

    length_m  = 0.0
    max_slope = 0.0

    for u, v in zip(path[:-1], path[1:]):
        data = _best_edge(G, u, v)
        if data is None:
            continue
        length_m  += float(data.get('length', 0.0))
        max_slope  = max(max_slope, float(data.get('slope', 0.0)))

    confidence, _ = compute_confidence(path, G)

    return {
        'length_m':   round(length_m, 1),
        'n_edges':    len(path) - 1,
        'max_slope':  round(max_slope, 2),
        'confidence': confidence,
    }


def _best_edge(G, u, v):
    """
    Returns the attribute dict of the shortest edge between u and v.
    Handles both simple Graph and MultiDiGraph (osmnx default).
    """
    edges = G.get_edge_data(u, v)
    if edges is None:
        return None

    if 'length' in edges:
        return edges

    return min(edges.values(), key=lambda d: d.get('length', float('inf')))


def _confidence_label(score):
    """Maps a numeric confidence score to a human-readable string."""
    for threshold, _, label in sorted(CONFIDENCE_LEVELS, reverse=True):
        if score >= threshold:
            return label
    return CONFIDENCE_LEVELS[-1][2]