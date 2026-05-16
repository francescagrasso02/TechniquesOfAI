# main.py — Wheelchair Routing Brussels — INFO-H410
#
# Runs three AI approaches on the same routing problem and compares them.
#
# Scenario A — long route, deterministic world (ULB → Gare du Midi)
#   CSP  : hard constraints, Dijkstra on filtered graph
#   A*   : soft constraints, heuristic search on full graph
#   MDP  : skipped (value iteration does not scale to graphs this large)
#
# Scenario B — short route, stochastic world (ULB → Cimetière d'Ixelles)
#   CSP  : same as above but on a smaller graph
#   A*   : same as above but on a smaller graph
#   MDP  : value iteration + policy extraction on CityRouting graph
#
# Output: two interactive HTML maps (one per scenario) + a comparison table.
#
# Usage:
#   python main.py

import time
import numpy as np
import osmnx as ox
import folium

from graph_builder import load_or_download
from csp import solve as csp_solve, path_metrics as csp_path_metrics
from astar import astar_manual, path_metrics as astar_path_metrics
from mdp import value_iteration, get_policy, apply_policy, N_ITERATIONS, GAMMA
from grid import CityRouting, WEIGHTS


# Scenario A: long, deterministic
SCENARIO_A_ORIGIN = (50.8116064, 4.380511)   # ULB
SCENARIO_A_DEST   = (50.8364862, 4.337896)   # Gare du Midi
SCENARIO_A_MARGIN = 500

# Scenario B: short, stochastic
SCENARIO_B_ORIGIN = (50.8116064, 4.380511)   # ULB
SCENARIO_B_DEST   = (50.8164000, 4.382400)   # Cimetière d'Ixelles
SCENARIO_B_MARGIN = 200

ROUTE_COLORS = {
    'CSP':  '#0055ff',
    'A*':   '#ff6600',
    'MDP':  '#00aa44',
}


def nearest_node(G, point):
    """Maps a (lat, lon) coordinate to the nearest graph node id."""
    try:
        return ox.distance.nearest_nodes(G, X=point[1], Y=point[0])
    except (TypeError, AttributeError):
        return ox.nearest_nodes(G, point[1], point[0])


def path_to_latlon(G, path):
    """Converts a list of node ids to a list of (lat, lon) tuples for Folium."""
    return [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]


def safe_apply_policy(city, start_node, target_node, max_steps=1000):
    """
    Follows the MDP policy from start_node to target_node.
    Returns the path as a list of node ids, or None if the policy
    leads to a cycle or a dead end before reaching the target.
    """
    state  = start_node
    path   = []
    visited = set()

    while state != target_node:
        if state in visited or len(path) >= max_steps:
            return None
        visited.add(state)
        path.append(state)
        action = city.policy.get(state)
        if action is None:
            return None
        state = action

    path.append(target_node)
    return path


def build_folium_map(G, paths, center):
    """
    Builds a Folium map with one PolyLine per approach.
    Each approach is in its own FeatureGroup so the user can toggle them.

    paths: dict of {approach_name: list_of_node_ids or None}
    center: (lat, lon) tuple for the initial map view
    """
    m = folium.Map(location=center, zoom_start=15, tiles='OpenStreetMap')

    for name, path in paths.items():
        if path is None:
            continue
        coords = path_to_latlon(G, path)
        fg = folium.FeatureGroup(name=name, show=True)
        folium.PolyLine(
            coords,
            color=ROUTE_COLORS[name],
            weight=5,
            opacity=0.85,
            tooltip=name,
        ).add_to(fg)
        folium.Marker(
            coords[0],
            tooltip=f'{name} — origin',
            icon=folium.Icon(color='green', icon='circle', prefix='fa'),
        ).add_to(fg)
        folium.Marker(
            coords[-1],
            tooltip=f'{name} — destination',
            icon=folium.Icon(color='red', icon='circle', prefix='fa'),
        ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def print_table(scenario_name, rows):
    """Prints a comparison table for one scenario."""
    print(f"\n{scenario_name}")
    header = f"  {'approach':<10} {'found':<8} {'length (m)':<13} {'time (ms)':<12} {'expanded':<11} {'confidence':<12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        found      = "yes"  if r['found']      else "no"
        length     = f"{r['length_m']:.0f}"   if r['found'] else "-"
        time_ms    = f"{r['time_ms']:.1f}"    if r['found'] else "-"
        expanded   = str(r['expanded'])        if r.get('expanded') is not None else "-"
        confidence = f"{r['confidence']:.1f}%" if r.get('confidence') is not None else "-"
        print(f"  {r['name']:<10} {found:<8} {length:<13} {time_ms:<12} {expanded:<11} {confidence:<12}")


def run_scenario_a():
    """
    Scenario A: ULB → Gare du Midi, margin=500m.
    Runs CSP and A*. MDP is skipped — value iteration does not scale
    to graphs of this size (several thousand nodes).
    """
    print("\nScenario A — ULB → Gare du Midi (deterministic, margin=500m)")
    print("Loading graph...")
    G = load_or_download(SCENARIO_A_ORIGIN, SCENARIO_A_DEST, margin=SCENARIO_A_MARGIN)
    print(f"  {len(G.nodes)} nodes, {len(G.edges)} edges")

    rows  = []
    paths = {}

    # CSP
    t0         = time.perf_counter()
    csp_result = csp_solve(G, SCENARIO_A_ORIGIN, SCENARIO_A_DEST)
    csp_time   = (time.perf_counter() - t0) * 1000

    if csp_result['path']:
        m = csp_path_metrics(csp_result['path'], G)
        rows.append({
            'name':       'CSP',
            'found':      True,
            'length_m':   m['length_m'],
            'time_ms':    csp_time,
            'expanded':   None,
            'confidence': csp_result['confidence'],
        })
        paths['CSP'] = csp_result['path']
    else:
        rows.append({'name': 'CSP', 'found': False})

    # A*
    origin     = nearest_node(G, SCENARIO_A_ORIGIN)
    goal       = nearest_node(G, SCENARIO_A_DEST)
    t0         = time.perf_counter()
    ast_result = astar_manual(G, origin, goal)
    ast_time   = (time.perf_counter() - t0) * 1000

    if ast_result['path']:
        m = astar_path_metrics(G, ast_result['path'])
        rows.append({
            'name':       'A*',
            'found':      True,
            'length_m':   m['length_m'],
            'time_ms':    ast_time,
            'expanded':   ast_result['expanded'],
            'confidence': None,
        })
        paths['A*'] = ast_result['path']
    else:
        rows.append({'name': 'A*', 'found': False})

    # MDP 
    rows.append({
        'name':       'MDP',
        'found':      False,
        'length_m':   None,
        'time_ms':    None,
        'expanded':   None,
        'confidence': None,
    })

    print_table("Scenario A", rows)

    center = (
        (SCENARIO_A_ORIGIN[0] + SCENARIO_A_DEST[0]) / 2,
        (SCENARIO_A_ORIGIN[1] + SCENARIO_A_DEST[1]) / 2,
    )
    m = build_folium_map(G, paths, center)
    m.save("map_scenario_a.html")
    print("\n  map saved to map_scenario_a.html")


def run_scenario_b():
    """
    Scenario B: ULB → Cimetière d'Ixelles, margin=200m.
    Runs all three approaches on the same small graph enriched
    with stochastic attributes (crowd, weather) via CityRouting.
    """
    print("\nScenario B — ULB → Cimetière d'Ixelles (stochastic, margin=200m)")
    print("Loading graph...")
    G = load_or_download(SCENARIO_B_ORIGIN, SCENARIO_B_DEST, margin=SCENARIO_B_MARGIN)
    print(f"  {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Enrich the graph with stochastic attributes for MDP and A* cost
    weather = np.random.randint(0, 5)
    city    = CityRouting(G)
    city.inject_missing_attributes(weather)
    city.get_cost(WEIGHTS)
    G_enriched = city.graph

    rows  = []
    paths = {}

    # CSP — runs on the enriched graph (OSM attributes unchanged)
    t0         = time.perf_counter()
    csp_result = csp_solve(G_enriched, SCENARIO_B_ORIGIN, SCENARIO_B_DEST)
    csp_time   = (time.perf_counter() - t0) * 1000

    if csp_result['path']:
        m = csp_path_metrics(csp_result['path'], G_enriched)
        rows.append({
            'name':       'CSP',
            'found':      True,
            'length_m':   m['length_m'],
            'time_ms':    csp_time,
            'expanded':   None,
            'confidence': csp_result['confidence'],
        })
        paths['CSP'] = csp_result['path']
    else:
        rows.append({'name': 'CSP', 'found': False})

    # A*
    origin     = nearest_node(G_enriched, SCENARIO_B_ORIGIN)
    goal       = nearest_node(G_enriched, SCENARIO_B_DEST)
    t0         = time.perf_counter()
    ast_result = astar_manual(G_enriched, origin, goal)
    ast_time   = (time.perf_counter() - t0) * 1000

    if ast_result['path']:
        m = astar_path_metrics(G_enriched, ast_result['path'])
        rows.append({
            'name':       'A*',
            'found':      True,
            'length_m':   m['length_m'],
            'time_ms':    ast_time,
            'expanded':   ast_result['expanded'],
            'confidence': None,
        })
        paths['A*'] = ast_result['path']
    else:
        rows.append({'name': 'A*', 'found': False})

    # MDP
    t0 = time.perf_counter()
    value_iteration(city, N_ITERATIONS, goal, GAMMA)
    get_policy(city, GAMMA)
    mdp_path = safe_apply_policy(city, origin, goal)
    mdp_time = (time.perf_counter() - t0) * 1000

    if mdp_path:
        m = astar_path_metrics(G_enriched, mdp_path)
        rows.append({
            'name':       'MDP',
            'found':      True,
            'length_m':   m['length_m'],
            'time_ms':    mdp_time,
            'expanded':   None,
            'confidence': None,
        })
        paths['MDP'] = mdp_path
    else:
        rows.append({'name': 'MDP', 'found': False})

    print_table("Scenario B", rows)

    center = (
        (SCENARIO_B_ORIGIN[0] + SCENARIO_B_DEST[0]) / 2,
        (SCENARIO_B_ORIGIN[1] + SCENARIO_B_DEST[1]) / 2,
    )
    m = build_folium_map(G_enriched, paths, center)
    m.save("map_scenario_b.html")
    print("\n  map saved to map_scenario_b.html")


if __name__ == '__main__':
    run_scenario_a()
    run_scenario_b()

