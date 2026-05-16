import time
import numpy as np
import networkx as nx
import osmnx as ox
import folium
import pandas as pd

from utils.graph_builder import load_or_download
from method_csp.csp import solve as csp_solve, path_metrics as csp_path_metrics
from method_astar.astar import astar_manual, path_metrics as astar_path_metrics
from method_mdp.mdp import value_iteration, get_policy, apply_policy, N_ITERATIONS, GAMMA, compute_path_cost
from utils.grid import CityRouting, WEIGHTS
from utils.config import inject_shared_random_attributes


FINAL_BENCHMARK_PAIRS = [
    # Short paths
    ("De Brouckère -> Monnaie",       (50.8510, 4.3526), (50.8498, 4.3529)),  # 150m
    ("Mont des Arts -> Sablon",       (50.8438, 4.3565), (50.8400, 4.3550)),  # 450m

    # Intermediate paths
    ("Porte de Hal -> Jeu de Balle",  (50.8331, 4.3444), (50.8373, 4.3458)),  # 500m
    ("Georges Henri -> Montgomery",   (50.8427, 4.4057), (50.8379, 4.4075)),  # 600m
    ("Palais Royal -> Pl. Luxembourg",(50.8423, 4.3626), (50.8392, 4.3725)),  # 850m
    ("VUB -> Delta",                  (50.8226, 4.3946), (50.8170, 4.4045)),  # 1.0km

    # Long paths
    ("Rogier -> Basilique Koekelberg",(50.8557, 4.3582), (50.8665, 4.3175)),  # 3.5km
    ("Atomium -> Gare du Nord",       (50.8949, 4.3415), (50.8606, 4.3608)),  # 4.5km
]


ROUTE_COLORS = {
    'Baseline': '#888888',
    'CSP':      '#0055ff',
    'A*':       '#ff6600',
    'MDP':      '#00aa44',
}

CSP_ESCALATION_MULTIPLIER = 3  # larger margin = default_margin * this


def nearest_node(G, point):
    try:
        return ox.distance.nearest_nodes(G, X=point[1], Y=point[0])
    except (TypeError, AttributeError):
        return ox.nearest_nodes(G, point[1], point[0])


def path_to_latlon(G, path):
    return [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]


def csp_with_escalation(origin, destination, default_margin, G_default):
    """
    Runs the CSP with the 4-step escalation:
      1. strict   + default-area graph
      2. strict   + larger-area  graph
      3. relaxed  + default-area graph
      4. relaxed  + larger-area  graph

    Other approaches (A*, MDP) keep the stricter default graph.
    Returns (result_dict, graph_used_for_csp, attempts_log).
    """
    larger_margin = default_margin * CSP_ESCALATION_MULTIPLIER
    G_larger = None
    attempts = []

    plan = [
        ('strict_only',  'default', G_default,  default_margin),
        ('strict_only',  'larger',  None,       larger_margin),
        ('relaxed_only', 'default', G_default,  default_margin),
        ('relaxed_only', 'larger',  None,       larger_margin),
    ]

    last_result = None
    for csp_mode, area_label, G, m in plan:
        if G is None:
            if G_larger is None:
                G_larger = load_or_download(origin, destination, m)
            G = G_larger

        result = csp_solve(G, origin, destination, mode=csp_mode)
        attempts.append({
            'constraints': csp_mode.replace('_only', ''),
            'area':        area_label,
            'margin':      m,
            'found':       result['path'] is not None,
        })
        last_result = result

        if result['path'] is not None:
            return result, G, attempts

    return last_result, G_default, attempts


def build_folium_map(G, paths, center):
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
    print(f"\n{scenario_name}")
    header = f"  {'approach':<10} {'found':<8} {'length (m)':<13} {'time (ms)':<12} {'expanded':<11} {'confidence':<12} {'cost':<12} "
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        found      = "yes"  if r['found']      else "no"
        length     = f"{r['length_m']:.0f}"   if r['found'] else "-"
        time_ms    = f"{r['time_ms']:.1f}"    if r['found'] else "-"
        expanded   = str(r['expanded'])        if r.get('expanded') is not None else "-"
        confidence = f"{r['confidence']:.1f}%" if r.get('confidence') is not None else "-"
        cost       = f"{r['cost']:.1f}"        if r.get('cost') is not None else "-"
        print(f"  {r['name']:<10} {found:<8} {length:<13} {time_ms:<12} {expanded:<11} {confidence:<12} {cost:<12}")


def run_scenario(scenario, weather, margin, n):
    name        = scenario[0]
    origin      = scenario[1]
    destination = scenario[2]

    print(f"\n{name}")
    print("Loading graph...")

    G = load_or_download(origin, destination, margin)
    G = inject_shared_random_attributes(G)
    print(f"  {len(G.nodes)} nodes, {len(G.edges)} edges")

    rows  = []
    paths = {}

    city = CityRouting(G)
    city.get_cost(WEIGHTS,weather)

    origin_node = nearest_node(G, origin)
    goal_node   = nearest_node(G, destination)

    # Baseline — unconstrained shortest path on length
    t0 = time.perf_counter()
    try:
        sp_path   = nx.shortest_path(G, origin_node, goal_node, weight='length')
        sp_length = nx.shortest_path_length(G, origin_node, goal_node, weight='length')
        sp_found  = True
    except nx.NetworkXNoPath:
        sp_path, sp_length, sp_found = None, None, False
    sp_time = (time.perf_counter() - t0) * 1000

    if sp_found:
        rows.append({
            'name':       'Baseline',
            'found':      True,
            'length_m':   sp_length,
            'time_ms':    sp_time,
            'expanded':   None,
            'confidence': None,
            'cost':       None,
        })
        paths['Baseline'] = sp_path
    else:
        rows.append({'name': 'Baseline', 'found': False})

    # CSP — with 4-step escalation
    t0 = time.perf_counter()
    csp_result, G_csp, csp_attempts = csp_with_escalation(
        origin, destination, margin, G
    )
    csp_time = (time.perf_counter() - t0) * 1000

    csp_attempt_str = ", ".join(
        f"{a['constraints']}/{a['area']}" + ("(ok)" if a['found'] else "(no)")
        for a in csp_attempts
    )
    print(f"  CSP attempts: {csp_attempt_str}")

    if csp_result['path']:
        m = csp_path_metrics(csp_result['path'], G_csp)
        rows.append({
            'name':       'CSP',
            'found':      True,
            'length_m':   m['length_m'],
            'time_ms':    csp_time,
            'expanded':   None,
            'confidence': csp_result['confidence'],
            'cost':       None,
            'csp_mode':   csp_result['mode'],
            'csp_area':   csp_attempts[-1]['area'],
        })
        paths['CSP'] = csp_result['path']
    else:
        rows.append({'name': 'CSP', 'found': False})

    # A*
    t0         = time.perf_counter()
    ast_result = astar_manual(G, origin_node, goal_node)
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
            'cost':       None,
        })
        paths['A*'] = ast_result['path']
    else:
        rows.append({'name': 'A*', 'found': False})

    # MDP
    t0 = time.perf_counter()
    value_iteration(city, N_ITERATIONS, goal_node, GAMMA)
    get_policy(city, GAMMA)
    mdp_path = apply_policy(city, origin_node, goal_node)
    mdp_time = (time.perf_counter() - t0) * 1000

    if mdp_path:
        m = astar_path_metrics(G, mdp_path)
        rows.append({
            'name':       'MDP',
            'found':      True,
            'length_m':   m['length_m'],
            'time_ms':    mdp_time,
            'expanded':   None,
            'confidence': None,
            'cost':       compute_path_cost(city, mdp_path),
        })
        paths['MDP'] = mdp_path
    else:
        rows.append({'name': 'MDP', 'found': False})

    print_table(f"{name}", rows)

    center = (
        (scenario[1][0] + scenario[2][0]) / 2,
        (scenario[1][1] + scenario[2][1]) / 2,
    )
    # Use G_csp because if CSP escalated to a larger area, it contains all
    # nodes of the default G, so other path nodes are still lookupable.
    folium_map = build_folium_map(G_csp, paths, center)
    folium_map.save(f"maps_sunny/map_scenario_{n}.html")
    print(f"\n  map saved to map_scenario_{n}.html")

    for r in rows:
        r['scenario'] = name
    return rows


if __name__ == '__main__':
    all_results = []
    weather = 3
    print("weather",weather)
    for n, scenario in enumerate(FINAL_BENCHMARK_PAIRS):
        scenario_rows = run_scenario(scenario, weather=weather, margin=200, n=n)
        all_results.extend(scenario_rows)

    df = pd.DataFrame(all_results)
    cols_order = ['scenario', 'name', 'found', 'length_m', 'time_ms', 'cost', 'confidence', 'expanded']
    df = df[cols_order]

    df.to_csv("final_benchmark_results_sunny.csv", index=False)
    print("Benchmark results saved in: 'final_benchmark_results_sunny.csv'")
    print(df.head(10))
