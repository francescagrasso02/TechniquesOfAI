# benchmark.py - Multi-pair experimental evaluation for the report
# INFO-H410 - Wheelchair Routing - ULB
#
# Runs A* (manual) and Dijkstra on a fixed set of origin-destination pairs
# across Brussels. Writes results to benchmark_results.csv and produces a
# scatter plot showing the expansion-count gap between the two algorithms.
#
# Usage:
#   python benchmark.py
#
# Requires: astar.py and graph_builder.py in the same directory, and a
# working Python environment (see README in repo root).

import csv
import time
from pathlib import Path

import osmnx as ox
import matplotlib.pyplot as plt

from graph_builder import load_or_download
from astar import astar_manual, dijkstra_manual, path_metrics


# ---------------------------------------------------------------------------
# 1. ORIGIN-DESTINATION PAIRS
# ---------------------------------------------------------------------------
# Eight pairs spanning short, medium and long walking distances in Brussels.
# Coordinates are pre-resolved (lat, lon) so the benchmark is fully non-
# interactive - no geocoder, no manual selection of "Found N places".
#
# These should be the same OD pairs Francesca uses for the CSP, so the final
# report can show all three approaches on identical inputs.

OD_PAIRS = [
    # name                      origin (lat, lon)           destination (lat, lon)        approx straight-line
    ("ULB -> Gare du Midi",      (50.8116064, 4.380511),    (50.8364862, 4.337896)),    # ~4.5 km
    ("Grand Place -> Bourse",    (50.8466,    4.3528),      (50.8483,    4.3500)),       # ~250 m  - very short
    ("Flagey -> Sainte-Catherine", (50.8276,  4.3722),      (50.8500,    4.3471)),       # ~3.2 km
    ("Madou -> Botanique",       (50.8517,    4.3681),      (50.8541,    4.3650)),       # ~400 m  - short
    ("Schuman -> Cinquantenaire",(50.8430,    4.3811),      (50.8404,    4.3927)),       # ~900 m
    ("Louise -> Trone",          (50.8260,    4.3617),      (50.8378,    4.3675)),       # ~1.5 km
    ("Tour & Taxis -> Yser",     (50.8666,    4.3450),      (50.8533,    4.3500)),       # ~1.6 km
    ("ULB -> Bois de la Cambre", (50.8116064, 4.380511),    (50.8055,    4.3870)),       # ~700 m  - within campus area
]


# ---------------------------------------------------------------------------
# 2. RUN ONE PAIR
# ---------------------------------------------------------------------------

def run_pair(name: str, pt_A: tuple, pt_B: tuple, margin: int = 500) -> dict:
    """
    Load the graph for one OD pair, run A* and Dijkstra, return a row of
    metrics suitable for CSV output.
    """
    print(f"\n--- {name} ---")
    t_load_start = time.perf_counter()
    G = load_or_download(pt_A, pt_B, margin=margin)
    load_time = time.perf_counter() - t_load_start

    # Map GPS coords to nearest graph node (osmnx version-tolerant)
    try:
        origin = ox.distance.nearest_nodes(G, X=pt_A[1], Y=pt_A[0])
        goal   = ox.distance.nearest_nodes(G, X=pt_B[1], Y=pt_B[0])
    except (TypeError, AttributeError):
        origin = ox.nearest_nodes(G, pt_A[1], pt_A[0])
        goal   = ox.nearest_nodes(G, pt_B[1], pt_B[0])

    if origin == goal:
        print("  origin == goal (graph too small or points coincide), skipping")
        return None

    print(f"  graph: {len(G.nodes)} nodes, {len(G.edges)} edges, "
          f"load time {load_time:.1f}s")

    # Run both algorithms
    res_astar = astar_manual(G,    origin, goal)
    res_dij   = dijkstra_manual(G, origin, goal)

    if res_astar["path"] is None or res_dij["path"] is None:
        print("  no path found, skipping")
        return None

    m_astar = path_metrics(G, res_astar["path"])
    m_dij   = path_metrics(G, res_dij["path"])

    # Sanity: both algorithms should find the same optimum
    assert abs(res_astar["cost"] - res_dij["cost"]) < 1e-6, (
        f"A* and Dijkstra disagree on optimum: "
        f"{res_astar['cost']:.4f} vs {res_dij['cost']:.4f}"
    )

    speedup = (res_dij["runtime_s"] / res_astar["runtime_s"]
               if res_astar["runtime_s"] > 0 else float("inf"))
    expansion_ratio = (res_astar["expanded"] / res_dij["expanded"]
                       if res_dij["expanded"] > 0 else float("nan"))

    print(f"  cost          : {res_astar['cost']:.1f}")
    print(f"  length        : {m_astar['length_m']:.1f} m  "
          f"({m_astar['n_edges']} edges)")
    print(f"  A*  expanded  : {res_astar['expanded']:>6}  in "
          f"{res_astar['runtime_s']*1000:.1f} ms")
    print(f"  Dij expanded  : {res_dij  ['expanded']:>6}  in "
          f"{res_dij  ['runtime_s']*1000:.1f} ms")
    print(f"  speedup (time): {speedup:.2f}x   "
          f"expansion ratio: {expansion_ratio:.2%}")

    return {
        "pair_name"          : name,
        "graph_nodes"        : len(G.nodes),
        "graph_edges"        : len(G.edges),
        "path_length_m"      : round(m_astar["length_m"], 1),
        "path_cost"          : round(res_astar["cost"], 2),
        "path_edges"         : m_astar["n_edges"],
        "max_slope_pct"      : round(m_astar["max_slope"], 2),
        "astar_expanded"     : res_astar["expanded"],
        "dijkstra_expanded"  : res_dij["expanded"],
        "expansion_ratio"    : round(expansion_ratio, 4),
        "astar_time_ms"      : round(res_astar["runtime_s"] * 1000, 1),
        "dijkstra_time_ms"   : round(res_dij  ["runtime_s"] * 1000, 1),
        "speedup_time"       : round(speedup, 2),
    }


# ---------------------------------------------------------------------------
# 3. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print(f"Benchmarking A* vs Dijkstra on {len(OD_PAIRS)} OD pairs")
    print("=" * 78)

    rows = []
    for name, pt_A, pt_B in OD_PAIRS:
        try:
            row = run_pair(name, pt_A, pt_B, margin=500)
            if row is not None:
                rows.append(row)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    if not rows:
        print("\nNo successful runs.")
        raise SystemExit(1)

    # ---------- CSV output ----------
    csv_path = Path("benchmark_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path.resolve()}")

    # ---------- Summary table to stdout ----------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'OD pair':<28} {'len(m)':>7} {'A* exp':>8} "
          f"{'Dij exp':>8} {'ratio':>7} {'speedup':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['pair_name']:<28} {r['path_length_m']:>7.0f} "
              f"{r['astar_expanded']:>8} {r['dijkstra_expanded']:>8} "
              f"{r['expansion_ratio']:>7.2%} {r['speedup_time']:>7.2f}x")

    # Aggregate stats - the numbers to cite in the report's prose
    n = len(rows)
    mean_ratio = sum(r["expansion_ratio"] for r in rows) / n
    mean_speedup = sum(r["speedup_time"]   for r in rows) / n
    print("-" * 78)
    print(f"{'mean over ' + str(n) + ' pairs':<28} "
          f"{'':>7} {'':>8} {'':>8} {mean_ratio:>7.2%} {mean_speedup:>7.2f}x")

    # ---------- Plot: A* vs Dijkstra expansions ----------
    fig, ax = plt.subplots(figsize=(7, 6))
    dij_x = [r["dijkstra_expanded"] for r in rows]
    ast_y = [r["astar_expanded"]    for r in rows]
    labels = [r["pair_name"]        for r in rows]

    ax.scatter(dij_x, ast_y, s=80, c="#0066CC", edgecolors="black",
               linewidths=0.8, zorder=3)

    # Label each point with a short name
    short = {
        "ULB -> Gare du Midi"           : "ULB→Midi",
        "Grand Place -> Bourse"          : "GPlace→Bourse",
        "Flagey -> Sainte-Catherine"     : "Flagey→S-Cath",
        "Madou -> Botanique"             : "Madou→Bot",
        "Schuman -> Cinquantenaire"      : "Schuman→Cinq",
        "Louise -> Trone"                : "Louise→Trone",
        "Tour & Taxis -> Yser"           : "T&T→Yser",
        "ULB -> Bois de la Cambre"       : "ULB→Bois",
    }
    for x, y, lbl in zip(dij_x, ast_y, labels):
        ax.annotate(short.get(lbl, lbl), (x, y),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color="#333333")

    lim = max(max(dij_x), max(ast_y)) * 1.08
    ax.plot([0, lim], [0, lim],   "k--", lw=1, alpha=0.5, label="y = x (no speedup)")
    ax.plot([0, lim], [0, lim/2], color="gray", ls=":", lw=1, alpha=0.6, label="y = x/2")

    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Dijkstra — nodes expanded")
    ax.set_ylabel("A* — nodes expanded")
    ax.set_title(f"A* vs Dijkstra on {len(rows)} Brussels OD pairs\n"
                 f"A* expands {mean_ratio:.0%} of Dijkstra's nodes on average "
                 f"({mean_speedup:.1f}× faster)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plot_path = Path("benchmark_expansions.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nExpansion-count plot saved to {plot_path.resolve()}")