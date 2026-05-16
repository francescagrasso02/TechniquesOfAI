# astar.py — A* search for wheelchair-accessible routing
# INFO-H410 — Wheelchair Routing — ULB
#
# Finds the best accessible path between two points in Brussels.
# Uses A* with a haversine heuristic and an accessibility cost function
# that penalises (but never removes) inaccessible edges.
#
# Key difference from CSP: edges are never removed, just made more expensive.
# So A* always returns a path, even if it's not perfectly accessible.

import math
import time
import heapq

import networkx as nx
import osmnx as ox

from utils.config import STRICT_CONSTRAINTS


# ---------------------------------------------------------------------------
# COST FUNCTION
# ---------------------------------------------------------------------------
# Penalties are in "equivalent metres": a +20 penalty means the edge
# feels like taking a 20m detour. This keeps haversine admissible.

SLOPE_SOFT    = 5.0     # start penalising above this %
SLOPE_HARD    = STRICT_CONSTRAINTS['slope']   # 8.0% — same as CSP cutoff
WIDTH_SOFT    = 1.5     # start penalising below this (m)
WIDTH_HARD    = STRICT_CONSTRAINTS['width']   # 1.2m — same as CSP cutoff

SURFACE_PENALTIES = {
    'cobblestone': 8.0,
    'sett':        8.0,
    'gravel':     10.0,
    'ground':      6.0,
    'dirt':        6.0,
    'grass':       6.0,
    'sand':       10.0,
    'unknown':     2.0,
}

BLOCKED = 9999.0   # wheelchair=no: huge penalty but edge stays in graph


def accessibility_cost(u, v, data):
    """
    Cost of one edge. Returns length + penalties for accessibility issues.
    Higher cost = harder to traverse in a wheelchair.
    """
    # if cost was already computed by grid.py, use it directly
    if 'cost' in data and isinstance(data['cost'], (int, float)):
        return float(data['cost'])

    length = float(data.get('length', 1.0))

    # wheelchair=no → effectively blocked, but path still exists
    if str(data.get('wheelchair', 'unknown')).lower() == 'no':
        return BLOCKED

    cost = length

    # slope penalty: progressive above 5%, extra flat penalty above 8%
    slope = float(data.get('slope', 0.0))
    if slope > SLOPE_SOFT:
        cost += (slope - SLOPE_SOFT) * 10
    if slope > SLOPE_HARD:
        cost += 20

    # width penalty: flat penalty below 1.5m, extra below 1.2m
    width = float(data.get('width', 1.5))
    if width < WIDTH_SOFT:
        cost += 8
    if width < WIDTH_HARD:
        cost += 12

    # surface penalty
    surface = str(data.get('surface', 'unknown')).lower()
    cost += SURFACE_PENALTIES.get(surface, 0.0)

    # kerb: CSP removes edges with unknown kerb, A* just discourages them
    kerb = str(data.get('kerb', 'unknown')).lower()
    if kerb in ('raised', 'yes'):
        cost += 5
    elif kerb == 'unknown':
        cost += 3

    # wheelchair=unknown or limited: mild penalty
    wheelchair = str(data.get('wheelchair', 'unknown')).lower()
    if wheelchair in ('unknown', 'limited'):
        cost += 2

    return cost


# ---------------------------------------------------------------------------
# HAVERSINE HEURISTIC
# ---------------------------------------------------------------------------
# Straight-line distance between two GPS points (in metres).
# Always <= real road distance, so the heuristic is admissible → A* is optimal.

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two (lat, lon) points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return 2 * R * math.asin(math.sqrt(a))


def make_heuristic(G, goal):
    """Returns h(n) = haversine distance from node n to the goal."""
    goal_lat = G.nodes[goal]['y']
    goal_lon = G.nodes[goal]['x']

    def h(node, _):
        return haversine(G.nodes[node]['y'], G.nodes[node]['x'], goal_lat, goal_lon)

    return h


# ---------------------------------------------------------------------------
# A* — MANUAL IMPLEMENTATION
# ---------------------------------------------------------------------------

def astar_manual(G, origin, goal):
    """
    A* on a networkx graph using a binary min-heap.

    Written by hand (instead of nx.astar_path) so we can count
    how many nodes are expanded — needed for the benchmark comparison.

    Returns a dict with: path, cost, runtime_s, expanded.
    """
    t0 = time.perf_counter()
    h  = make_heuristic(G, goal)

    g_score   = {origin: 0.0}   # best known cost from origin to each node
    came_from = {}               # predecessor on the best known path

    counter   = 0
    open_heap = [(h(origin, goal), counter, origin)]
    open_best = {origin: h(origin, goal)}   # best f seen for each open node

    expanded = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        # stale entry — we already found a better path to this node
        if f > open_best.get(current, math.inf):
            continue

        if current == goal:
            path = _reconstruct(came_from, current)
            return {
                'path':      path,
                'cost':      g_score[current],
                'runtime_s': round(time.perf_counter() - t0, 4),
                'expanded':  expanded,
            }

        expanded += 1
        open_best.pop(current, None)

        for neighbour in G.neighbors(current):
            edge_data    = _best_edge(G, current, neighbour)
            step_cost    = accessibility_cost(current, neighbour, edge_data)
            tentative_g  = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbour, math.inf):
                came_from[neighbour] = current
                g_score[neighbour]   = tentative_g
                f_new = tentative_g + h(neighbour, goal)

                if f_new < open_best.get(neighbour, math.inf):
                    open_best[neighbour] = f_new
                    counter += 1
                    heapq.heappush(open_heap, (f_new, counter, neighbour))

    # no path found
    return {
        'path':      None,
        'cost':      math.inf,
        'runtime_s': round(time.perf_counter() - t0, 4),
        'expanded':  expanded,
    }


# ---------------------------------------------------------------------------
# DIJKSTRA BASELINE
# ---------------------------------------------------------------------------

def dijkstra_manual(G, origin, goal):
    """
    Same as astar_manual but with h(n) = 0 (no heuristic).
    Used as baseline to show how much A* saves in node expansions.
    """
    t0 = time.perf_counter()

    g_score   = {origin: 0.0}
    came_from = {}

    counter   = 0
    open_heap = [(0.0, counter, origin)]
    open_best = {origin: 0.0}

    expanded = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        if f > open_best.get(current, math.inf):
            continue

        if current == goal:
            path = _reconstruct(came_from, current)
            return {
                'path':      path,
                'cost':      g_score[current],
                'runtime_s': round(time.perf_counter() - t0, 4),
                'expanded':  expanded,
            }

        expanded += 1
        open_best.pop(current, None)

        for neighbour in G.neighbors(current):
            edge_data   = _best_edge(G, current, neighbour)
            tentative_g = g_score[current] + accessibility_cost(current, neighbour, edge_data)

            if tentative_g < g_score.get(neighbour, math.inf):
                came_from[neighbour] = current
                g_score[neighbour]   = tentative_g

                if tentative_g < open_best.get(neighbour, math.inf):
                    open_best[neighbour] = tentative_g
                    counter += 1
                    heapq.heappush(open_heap, (tentative_g, counter, neighbour))

    return {
        'path':      None,
        'cost':      math.inf,
        'runtime_s': round(time.perf_counter() - t0, 4),
        'expanded':  expanded,
    }


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _best_edge(G, u, v):
    """
    Returns the cheapest edge between u and v.
    Needed because OSMnx uses MultiDiGraph — two nodes can have
    multiple parallel edges (e.g. two lanes of the same street).
    """
    edges = G.get_edge_data(u, v)
    if edges is None:
        raise KeyError(f'No edge between {u} and {v}')
    if 'length' in edges:
        return edges   # simple graph
    return min(edges.values(), key=lambda d: accessibility_cost(u, v, d))


def _reconstruct(came_from, current):
    """Walks back through came_from to rebuild the path origin → goal."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_metrics(G, path):
    """
    Summary stats for a path — used by benchmark.py for the report table.
    Returns: length_m, cost, max_slope, n_edges, n_blocked.
    """
    if path is None or len(path) < 2:
        return {'length_m': 0.0, 'cost': 0.0, 'max_slope': 0.0,
                'n_edges': 0, 'n_blocked': 0}

    length_m  = 0.0
    cost      = 0.0
    max_slope = 0.0
    n_blocked = 0

    for u, v in zip(path[:-1], path[1:]):
        data = _best_edge(G, u, v)
        length_m  += float(data.get('length', 0.0))
        c          = accessibility_cost(u, v, data)
        cost      += c
        max_slope  = max(max_slope, float(data.get('slope', 0.0)))
        if c >= BLOCKED:
            n_blocked += 1

    return {
        'length_m':  round(length_m, 1),
        'cost':      round(cost, 2),
        'max_slope': round(max_slope, 2),
        'n_edges':   len(path) - 1,
        'n_blocked': n_blocked,
    }


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from graph_builder import load_or_download

    A = (50.8116064, 4.380511)   # ULB
    B = (50.8364862, 4.337896)   # Gare du Midi

    print('Loading graph...')
    G = load_or_download(A, B, margin=500)

    origin = ox.distance.nearest_nodes(G, X=A[1], Y=A[0])
    goal   = ox.distance.nearest_nodes(G, X=B[1], Y=B[0])

    print(f'Graph: {len(G.nodes)} nodes, {len(G.edges)} edges\n')

    res_astar = astar_manual(G, origin, goal)
    res_dijk  = dijkstra_manual(G, origin, goal)

    for name, res in [('A*', res_astar), ('Dijkstra', res_dijk)]:
        m = path_metrics(G, res['path'])
        print(f'{name}: cost={m["cost"]:.1f}m  length={m["length_m"]:.1f}m  '
              f'expanded={res["expanded"]}  time={res["runtime_s"]*1000:.1f}ms')

    # sanity check: both should find the same optimal cost
    if abs(res_astar['cost'] - res_dijk['cost']) < 1e-6:
        print('\n[OK] A* and Dijkstra agree on the optimal cost.')
    else:
        print('\n[!!] Costs differ — something is wrong.')
