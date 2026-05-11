# astar.py - A* search for wheelchair-accessible routing
# INFO-H410 - Wheelchair Routing - ULB
#
# This module implements A* search on the enriched OSM graph produced by
# graph_builder.py (Matteo). The same cost function is used by A* manual,
# A* networkx, and the Dijkstra baseline, so the comparison is fair: only
# the algorithm changes, not the objective.
#
# Soft vs hard constraints (key design choice for the report):
#   - The CSP approach prunes edges that violate
#     STRICT_CONSTRAINTS (slope > 8%, width < 1.2 m) before searching.
#   - A* (this module) keeps all edges in the graph and adds an
#     additive penalty proportional to how much an edge violates the
#     soft thresholds defined below. A high-penalty edge can still be
#     used if no alternative exists.

import math
import time
import heapq
from typing import Callable, Hashable

import networkx as nx

# Read the team-wide constraint thresholds so we stay aligned with the CSP.
# We deliberately start penalising earlier than the hard CSP cutoffs:
# A* "complains" sooner than CSP "forbids", which is what makes the two
# methods interestingly different at evaluation time.
from config import STRICT_CONSTRAINTS  # slope=8.0, width=1.2


# ---------------------------------------------------------------------------
# 1. ACCESSIBILITY COST FUNCTION
# ---------------------------------------------------------------------------
#
# Additive model: cost = length + sum(penalties).
# Penalties are expressed in "equivalent metres": a +8 penalty means the
# edge is as costly as taking an 8 m detour. This keeps the haversine
# heuristic admissible (see section 2) and makes the weights interpretable.

# Soft thresholds for A* (strictly easier than the CSP hard thresholds)
SLOPE_SOFT_THRESHOLD = 5.0                              # %  - start penalising
WIDTH_SOFT_THRESHOLD = 1.5                              # m  - "comfortable" min

# Hard thresholds (imported, kept for cross-checking against CSP)
SLOPE_HARD_THRESHOLD = STRICT_CONSTRAINTS["slope"]       # 8.0
WIDTH_HARD_THRESHOLD = STRICT_CONSTRAINTS["width"]       # 1.2

# Penalty weights
SLOPE_PENALTY_PER_PCT      = 10.0   # +metres per % above soft threshold
SLOPE_HARD_BONUS           = 20.0   # extra flat penalty above hard threshold
WIDTH_PENALTY              = 8.0    # flat penalty if below soft threshold
WIDTH_HARD_BONUS           = 12.0   # extra flat penalty if below hard threshold
KERB_PENALTY               = 5.0    # raised/yes kerb
KERB_UNKNOWN_PENALTY       = 3.0    # unknown kerb - CSP forbids, A* discourages
WHEELCHAIR_UNKNOWN_PENALTY = 2.0    # unknown wheelchair tag - mild caution

SURFACE_PENALTY = {
    "cobblestone": 8.0,
    "sett":        8.0,
    "gravel":     10.0,
    "ground":      6.0,
    "dirt":        6.0,
    "grass":       6.0,
    "sand":       10.0,
    "unknown":     2.0,
}

KERB_BAD = {"raised", "yes"}

# A wheelchair=no edge is effectively blocked (huge penalty rather than removal,
# so A* remains complete: if the only path goes through a forbidden edge, the
# user is at least told a route exists, with a very high cost flag).
BLOCKED = 9999.0


def accessibility_cost(u: Hashable, v: Hashable, data: dict) -> float:
    """
    Compute the wheelchair-accessibility cost of a single edge.

    If the edge already has a precomputed 'cost' attribute (e.g. produced by
    grid.CityRouting.get_cost()), we use it as-is - this lets the team swap
    in a shared cost function without touching this file.

    Otherwise we compute the cost from raw accessibility attributes.

    Parameters
    ----------
    u, v : node ids - unused but required by the networkx weight signature
    data : edge attribute dict from graph_builder.enrich_graph()

    Returns
    -------
    float : cost in "equivalent metres". Always >= data['length'].
    """
    # Opt-in to a shared cost field if it exists. This lets the team agree on
    # one cost function later without rewriting this module.
    if "cost" in data and isinstance(data["cost"], (int, float)):
        # We assume a precomputed cost is already in metre-equivalent units.
        # If the team adopts grid.py's normalised [0,1] cost instead, multiply
        # here by data['length'] so the haversine heuristic stays admissible.
        return float(data["cost"])

    length = float(data.get("length", 1.0))

    # Hard blocker - early return
    if str(data.get("wheelchair", "unknown")).lower() == "no":
        return BLOCKED

    cost = length

    # Slope: progressive penalty above soft threshold, extra bonus above hard
    slope = float(data.get("slope", 0.0))
    if slope > SLOPE_SOFT_THRESHOLD:
        cost += (slope - SLOPE_SOFT_THRESHOLD) * SLOPE_PENALTY_PER_PCT
    if slope > SLOPE_HARD_THRESHOLD:
        cost += SLOPE_HARD_BONUS

    # Width: flat penalty below soft, extra bonus below hard
    width = float(data.get("width", 1.5))
    if width < WIDTH_SOFT_THRESHOLD:
        cost += WIDTH_PENALTY
    if width < WIDTH_HARD_THRESHOLD:
        cost += WIDTH_HARD_BONUS

    # Surface
    surface = str(data.get("surface", "unknown")).lower()
    cost += SURFACE_PENALTY.get(surface, 0.0)

    # Kerb: CSP forbids 'unknown' outright (config.ABSOLUTE_CONSTRAINTS).
    # A* keeps the edge but discourages it - showing the soft/hard contrast.
    kerb = str(data.get("kerb", "unknown")).lower()
    if kerb in KERB_BAD:
        cost += KERB_PENALTY
    elif kerb == "unknown":
        cost += KERB_UNKNOWN_PENALTY

    # Wheelchair tag: 'limited' or 'unknown' is mildly discouraged
    wheelchair = str(data.get("wheelchair", "unknown")).lower()
    if wheelchair in ("unknown", "limited"):
        cost += WHEELCHAIR_UNKNOWN_PENALTY

    return cost


# ---------------------------------------------------------------------------
# 2. HAVERSINE HEURISTIC
# ---------------------------------------------------------------------------
#
# Great-circle distance between two GPS points, in metres.
#
# Admissibility argument (to put in the report):
#   haversine(n, goal) <= true_road_length(n -> goal)    [geodesic <= path]
#                      <= accessibility_cost(n -> goal)  [penalties >= 0]
#                      = h*(n)
# Therefore h is admissible w.r.t. our cost function, and A* is optimal.

_EARTH_RADIUS_M = 6_371_000.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def make_heuristic(G: nx.Graph, goal: Hashable) -> Callable[[Hashable, Hashable], float]:
    """
    Build a heuristic function h(u, v) suitable for nx.astar_path.

    networkx passes (current_node, goal_node) to the heuristic, but we always
    measure distance to the same goal - we capture goal coordinates in a
    closure for speed.
    """
    goal_lat = G.nodes[goal]["y"]
    goal_lon = G.nodes[goal]["x"]

    def h(u: Hashable, _v: Hashable) -> float:
        return haversine(G.nodes[u]["y"], G.nodes[u]["x"], goal_lat, goal_lon)

    return h


# ---------------------------------------------------------------------------
# 3. A* - NETWORKX VERSION (sanity check)
# ---------------------------------------------------------------------------

def astar_networkx(G: nx.Graph, origin: Hashable, goal: Hashable) -> dict:
    """
    Run A* using networkx's built-in implementation.
    Used as a reference to cross-check our manual version.

    networkx passes a *dict of parallel edges* to the weight function when G
    is a MultiDiGraph (which OSMnx produces). We wrap accessibility_cost in
    a multigraph-aware adapter that picks the cheapest parallel edge - same
    convention as our manual A* uses via _best_edge().

    Returns a dict with keys: path, cost, runtime_s, expanded (None for nx).
    """
    h = make_heuristic(G, goal)

    def weight_multigraph(u, v, edge_dict):
        # In a MultiDiGraph, networkx passes {key: data_dict, ...}.
        # In a simple Graph it passes the data dict directly.
        if "length" in edge_dict:                       # simple graph path
            return accessibility_cost(u, v, edge_dict)
        return min(                                     # multigraph path
            accessibility_cost(u, v, d) for d in edge_dict.values()
        )

    t0 = time.perf_counter()
    path = nx.astar_path(G, origin, goal, heuristic=h, weight=weight_multigraph)
    runtime = time.perf_counter() - t0
    return {
        "path": path,
        "cost": _path_cost(G, path),
        "runtime_s": runtime,
        "expanded": None,   # networkx does not expose this
    }

# ---------------------------------------------------------------------------
# 4. A* - MANUAL IMPLEMENTATION (the one we discuss in the report)
# ---------------------------------------------------------------------------

def astar_manual(G: nx.Graph, origin: Hashable, goal: Hashable) -> dict:
    """
    Hand-rolled A* on a networkx graph using a binary min-heap.

    We implement A* from scratch so we can:
      - count the number of nodes expanded (impossible with nx.astar_path)
      - demonstrate full understanding (project rubric: high-level packages
        require a thorough understanding of their underlying mechanisms)
    """
    t0 = time.perf_counter()
    h = make_heuristic(G, goal)

    # g_score[n] = best known cost from origin to n
    g_score: dict[Hashable, float] = {origin: 0.0}
    # came_from[n] = predecessor of n on the best known path
    came_from: dict[Hashable, Hashable] = {}

    # Open set: heap of (f, counter, node).
    # counter is a tiebreaker for deterministic ordering and to avoid heapq
    # trying to compare node IDs when two entries share the same f.
    counter = 0
    open_heap: list[tuple[float, int, Hashable]] = [(h(origin, goal), counter, origin)]
    # Best known f for each open node - used to detect stale heap entries
    # (lazy deletion: simpler than decrease-key with heapq).
    open_best_f: dict[Hashable, float] = {origin: h(origin, goal)}

    expanded = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        # Skip stale entries: this f is worse than what we already know
        if f > open_best_f.get(current, math.inf):
            continue

        if current == goal:
            path = _reconstruct(came_from, current)
            return {
                "path": path,
                "cost": g_score[current],
                "runtime_s": time.perf_counter() - t0,
                "expanded": expanded,
            }

        expanded += 1
        # Mark current as "closed": any future entry for it must beat g_score
        open_best_f.pop(current, None)

        for neighbour in G.neighbors(current):
            # Multigraph-safe: pick the cheapest parallel edge.
            # OSMnx returns MultiDiGraph; two carriageways may exist between
            # the same pair of nodes.
            edge_data = _best_edge(G, current, neighbour)
            step_cost = accessibility_cost(current, neighbour, edge_data)
            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbour, math.inf):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g
                f_new = tentative_g + h(neighbour, goal)
                if f_new < open_best_f.get(neighbour, math.inf):
                    open_best_f[neighbour] = f_new
                    counter += 1
                    heapq.heappush(open_heap, (f_new, counter, neighbour))

    # Open set exhausted, goal unreachable
    return {
        "path": None,
        "cost": math.inf,
        "runtime_s": time.perf_counter() - t0,
        "expanded": expanded,
    }


# ---------------------------------------------------------------------------
# 5. DIJKSTRA BASELINE
# ---------------------------------------------------------------------------

def dijkstra_manual(G: nx.Graph, origin: Hashable, goal: Hashable) -> dict:
    """
    Hand-rolled Dijkstra. Equivalent to astar_manual with h(n) = 0.

    Since the only difference is the heuristic, any gap in 'expanded' count
    between Dijkstra and A* is attributable to the heuristic alone -
    exactly the experimental claim we make in the report.
    """
    t0 = time.perf_counter()

    g_score: dict[Hashable, float] = {origin: 0.0}
    came_from: dict[Hashable, Hashable] = {}

    counter = 0
    open_heap: list[tuple[float, int, Hashable]] = [(0.0, counter, origin)]
    open_best_f: dict[Hashable, float] = {origin: 0.0}

    expanded = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)
        if f > open_best_f.get(current, math.inf):
            continue

        if current == goal:
            path = _reconstruct(came_from, current)
            return {
                "path": path,
                "cost": g_score[current],
                "runtime_s": time.perf_counter() - t0,
                "expanded": expanded,
            }

        expanded += 1
        open_best_f.pop(current, None)

        for neighbour in G.neighbors(current):
            edge_data = _best_edge(G, current, neighbour)
            tentative_g = g_score[current] + accessibility_cost(current, neighbour, edge_data)

            if tentative_g < g_score.get(neighbour, math.inf):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g
                if tentative_g < open_best_f.get(neighbour, math.inf):
                    open_best_f[neighbour] = tentative_g
                    counter += 1
                    heapq.heappush(open_heap, (tentative_g, counter, neighbour))

    return {
        "path": None,
        "cost": math.inf,
        "runtime_s": time.perf_counter() - t0,
        "expanded": expanded,
    }


# ---------------------------------------------------------------------------
# 6. HELPERS
# ---------------------------------------------------------------------------

def _best_edge(G: nx.Graph, u: Hashable, v: Hashable) -> dict:
    """
    Return the attribute dict of the cheapest edge between u and v.
    Handles both simple Graph and MultiDiGraph (OSMnx default).
    """
    edges = G.get_edge_data(u, v)
    if edges is None:
        raise KeyError(f"No edge between {u} and {v}")

    # Simple graph case: edges is the attribute dict itself
    if "length" in edges:
        return edges

    # MultiGraph case: edges is {key: data, ...} - pick the cheapest
    return min(edges.values(), key=lambda d: accessibility_cost(u, v, d))


def _reconstruct(came_from: dict, current: Hashable) -> list:
    """Walk back through came_from to build the path origin -> goal."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _path_cost(G: nx.Graph, path: list) -> float:
    """Sum accessibility_cost over a path. Used by astar_networkx."""
    if path is None:
        return math.inf
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += accessibility_cost(u, v, _best_edge(G, u, v))
    return total


# ---------------------------------------------------------------------------
# 7. PATH METRICS (for the experimental table in the report)
# ---------------------------------------------------------------------------

def path_metrics(G: nx.Graph, path: list) -> dict:
    """
    Compute summary statistics over a found path. These are the columns of
    the comparison table in the report.
    """
    if path is None or len(path) < 2:
        return {"length_m": 0.0, "cost": 0.0, "max_slope": 0.0,
                "n_edges": 0, "n_blocked": 0}

    length_m = 0.0
    cost = 0.0
    max_slope = 0.0
    n_blocked = 0

    for u, v in zip(path[:-1], path[1:]):
        data = _best_edge(G, u, v)
        length_m += float(data.get("length", 0.0))
        c = accessibility_cost(u, v, data)
        cost += c
        if c >= BLOCKED:
            n_blocked += 1
        max_slope = max(max_slope, float(data.get("slope", 0.0)))

    return {
        "length_m": length_m,
        "cost": cost,
        "max_slope": max_slope,
        "n_edges": len(path) - 1,
        "n_blocked": n_blocked,
    }


# ---------------------------------------------------------------------------
# 8. DEMO / SMOKE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # End-to-end demo:
    # 1. load the cached Brussels graph (run test_graph_builder.py first
    #    to populate the cache - this script is non-interactive)
    # 2. run A* (networkx), A* (manual), Dijkstra on the same OD pair
    # 3. print the comparison table and save a PNG of the chosen route
    import osmnx as ox
    from graph_builder import load_or_download

    print("=" * 78)
    print("A* wheelchair routing - demo")
    print("=" * 78)

    # Same coordinates as test_graph_builder.py so the cache hits immediately
    pt_A = (50.8116064, 4.380511)    # ULB
    pt_B = (50.8364862, 4.337896)    # Gare du Midi
    print(f"\nOrigin (ULB)             : {pt_A}")
    print(f"Destination (Gare du Midi): {pt_B}\n")

    G = load_or_download(pt_A, pt_B, margin=500)

    # Map GPS coords -> nearest graph node. osmnx >=1.3 expects X (lon) and
    # Y (lat) as separate kwargs; older versions used a tuple.
    try:
        origin = ox.distance.nearest_nodes(G, X=pt_A[1], Y=pt_A[0])
        goal   = ox.distance.nearest_nodes(G, X=pt_B[1], Y=pt_B[0])
    except (TypeError, AttributeError):
        origin = ox.nearest_nodes(G, pt_A[1], pt_A[0])
        goal   = ox.nearest_nodes(G, pt_B[1], pt_B[0])

    print(f"\nNearest graph nodes: origin={origin}, goal={goal}")
    print(f"Graph size: {len(G.nodes)} nodes, {len(G.edges)} edges\n")

    results = {
        "A* (networkx)": astar_networkx(G, origin, goal),
        "A* (manual)"  : astar_manual  (G, origin, goal),
        "Dijkstra"     : dijkstra_manual(G, origin, goal),
    }

    # Print comparison table
    print(f"\n{'Algorithm':<16} {'cost':>10} {'length(m)':>11} "
          f"{'max slope %':>12} {'edges':>7} {'expanded':>10} {'time(ms)':>10}")
    print("-" * 78)
    for name, r in results.items():
        m = path_metrics(G, r["path"])
        exp = str(r["expanded"]) if r["expanded"] is not None else "-"
        print(f"{name:<16} {m['cost']:>10.1f} {m['length_m']:>11.1f} "
              f"{m['max_slope']:>12.2f} {m['n_edges']:>7} "
              f"{exp:>10} {r['runtime_s']*1000:>10.1f}")

    # Sanity check: A* networkx and A* manual must find the same cost.
    # (They may return different paths if multiple optima exist, but the
    # cost must match to the last decimal.)
    nx_cost  = results["A* (networkx)"]["cost"]
    man_cost = results["A* (manual)"]["cost"]
    dij_cost = results["Dijkstra"]["cost"]

    print()
    if abs(nx_cost - man_cost) < 1e-6:
        print("[OK] A* manual matches networkx (both optimal).")
    else:
        print(f"[!!] A* costs differ: nx={nx_cost:.4f}, manual={man_cost:.4f}")

    if abs(dij_cost - man_cost) < 1e-6:
        print("[OK] Dijkstra finds the same optimum (expected).")
    else:
        print(f"[!!] Dijkstra differs from A*: this should not happen.")

    # Save a PNG of the chosen route for the report
    try:
        import matplotlib.pyplot as plt

        fig, ax = ox.plot_graph_route(
            G, results["A* (manual)"]["path"],
            route_color="#0000FF", route_linewidth=3, route_alpha=0.9,
            node_size=0,
            bgcolor="white", edge_color="#DDDDDD", edge_linewidth=0.5,
            show=False, close=False,
            figsize=(8, 8),
        )

        # Overlay origin (green) and destination (red) as separate scatter points,
        # so they are guaranteed to be drawn on top of the route in the right colours.
        ax.scatter(
            G.nodes[origin]["x"], G.nodes[origin]["y"],
            c="#00AA00", s=120, zorder=5, edgecolors="black", linewidths=1.0,
            label="Origin (ULB)",
        )
        ax.scatter(
            G.nodes[goal]["x"], G.nodes[goal]["y"],
            c="#FF4400", s=120, zorder=5, edgecolors="black", linewidths=1.0,
            label="Destination (Gare du Midi)",
        )
        ax.legend(loc="lower left", fontsize=9, frameon=True)

        fig.savefig("astar_route.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("\nRoute plot saved to astar_route.png")
    except Exception as e:
        print(f"\nPlot skipped: {e}")
