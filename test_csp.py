# test_csp.py — Tests for csp.py
# INFO-H410 — Wheelchair Routing CSP — ULB
# Run with: python test_csp.py

import pickle
import os

from config import DATA_DIR
from csp import build_feasible_graph, find_path, compute_confidence, solve, path_metrics


# Fixed coordinates — same as test_graph_builder.py so the cache hits immediately
A = (50.8116064, 4.380511)   # ULB
B = (50.8364862, 4.337896)   # Gare du Midi


def load_graph():
    filename = f"graph_{A[0]:.4f}_{A[1]:.4f}_{B[0]:.4f}_{B[1]:.4f}.pkl"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"Cache not found at {filepath}")
        print("Run test_graph_builder.py first to generate the graph.")
        exit(1)

    print(f"Loading graph from cache: {filepath}")
    with open(filepath, 'rb') as f:
        G = pickle.load(f)
    print(f"  nodes: {len(G.nodes)}, edges: {len(G.edges)}")
    return G


def test_build_feasible_graph(G):
    print("\nTEST 1 - build_feasible_graph()")

    print("\n  strict mode:")
    H_strict, stats_strict = build_feasible_graph(G, strict=True)
    print(f"    edges before: {stats_strict['edges_total']}")
    print(f"    edges removed: {stats_strict['edges_removed']} ({stats_strict['removal_pct']}%)")
    print(f"    edges kept: {stats_strict['edges_kept']}")

    print("\n  relaxed mode:")
    H_relaxed, stats_relaxed = build_feasible_graph(G, strict=False)
    print(f"    edges before: {stats_relaxed['edges_total']}")
    print(f"    edges removed: {stats_relaxed['edges_removed']} ({stats_relaxed['removal_pct']}%)")
    print(f"    edges kept: {stats_relaxed['edges_kept']}")

    assert stats_relaxed['edges_kept'] >= stats_strict['edges_kept'], \
        "relaxed mode should keep at least as many edges as strict"
    print("\n  sanity check passed: relaxed keeps >= edges than strict")

    return H_strict, H_relaxed


def test_find_path(H_strict, H_relaxed):
    print("\nTEST 2 - find_path()")

    print("\n  strict graph:")
    result_strict = find_path(H_strict, A, B)
    if result_strict['path']:
        print(f"    path found: {len(result_strict['path'])} nodes, {result_strict['cost']:.1f}m, {result_strict['runtime_s']}s")
    else:
        print("    no path found")

    print("\n  relaxed graph:")
    result_relaxed = find_path(H_relaxed, A, B)
    if result_relaxed['path']:
        print(f"    path found: {len(result_relaxed['path'])} nodes, {result_relaxed['cost']:.1f}m, {result_relaxed['runtime_s']}s")
    else:
        print("    no path found")

    return result_strict, result_relaxed


def test_compute_confidence(G, result_strict, result_relaxed):
    print("\nTEST 3 - compute_confidence()")

    if result_strict['path']:
        score, label = compute_confidence(result_strict['path'], G)
        print(f"\n  strict path: {score}% — {label}")

    if result_relaxed['path']:
        score, label = compute_confidence(result_relaxed['path'], G)
        print(f"  relaxed path: {score}% — {label}")


def test_solve(G):
    print("\nTEST 4 - solve()")

    result = solve(G, A, B)

    print(f"\n  mode: {result['mode']}")
    print(f"  path found: {result['path'] is not None}")

    if result['path']:
        print(f"  cost: {result['cost']:.1f}m")
        print(f"  confidence: {result['confidence']}% — {result['label']}")
        print(f"  runtime: {result['runtime_s']}s")

        m = path_metrics(result['path'], G)
        print(f"\n  path metrics:")
        print(f"    length:     {m['length_m']}m")
        print(f"    edges:      {m['n_edges']}")
        print(f"    max slope:  {m['max_slope']}%")
        print(f"    confidence: {m['confidence']}%")

    print(f"\n  csp stats:")
    for k, v in result['csp_stats'].items():
        print(f"    {k}: {v}")

    return result


if __name__ == "__main__":
    G = load_graph()

    H_strict, H_relaxed        = test_build_feasible_graph(G)
    result_strict, result_relaxed = test_find_path(H_strict, H_relaxed)
    test_compute_confidence(G, result_strict, result_relaxed)
    test_solve(G)

    print("\nall tests completed")