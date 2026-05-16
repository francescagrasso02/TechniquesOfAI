import time
import numpy as np
import osmnx as ox
import folium
import pandas as pd

from graph_builder import load_or_download
from csp import solve as csp_solve, path_metrics as csp_path_metrics
from astar import astar_manual, path_metrics as astar_path_metrics
from mdp import value_iteration, get_policy, apply_policy, N_ITERATIONS, GAMMA, EPSILON,compute_path_cost
from grid import CityRouting, WEIGHTS


FINAL_BENCHMARK_PAIRS = [
    #Short paths
    ("De Brouckère -> Monnaie",(50.8510, 4.3526), (50.8498, 4.3529)),  #150m 
    ("Mont des Arts -> Sablon",(50.8438, 4.3565), (50.8400, 4.3550)),  #450m 
    
    #Intermediate paths
    ("Porte de Hal -> Jeu de Balle", (50.8331, 4.3444), (50.8373, 4.3458)),  #500m 
    ("Georges Henri -> Montgomery", (50.8427, 4.4057), (50.8379, 4.4075)),  #600m 
    ("Palais Royal -> Pl. Luxembourg",(50.8423, 4.3626), (50.8392, 4.3725)),  #850m 
    ("VUB -> Delta",(50.8226, 4.3946), (50.8170, 4.4045)),  #1.0km 

    #Long paths
    ("Rogier -> Basilique Koekelberg",(50.8557, 4.3582), (50.8665, 4.3175)),  #3.5km 
    ("Atomium -> Gare du Nord",(50.8949, 4.3415), (50.8606, 4.3608)),  #4.5km 
]


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
    header = f"  {'approach':<10} {'found':<8} {'length (m)':<13} {'time (ms)':<12} {'expanded':<11} {'confidence':<12} {'cost':<12} "
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        found      = "yes"  if r['found']      else "no"
        length     = f"{r['length_m']:.0f}"   if r['found'] else "-"
        time_ms    = f"{r['time_ms']:.1f}"    if r['found'] else "-"
        expanded   = str(r['expanded'])        if r.get('expanded') is not None else "-"
        confidence = f"{r['confidence']:.1f}%" if r.get('confidence') is not None else "-"
        cost = f"{r['cost']:.1f}" if r.get('cost') is not None else "-"
        print(f"  {r['name']:<10} {found:<8} {length:<13} {time_ms:<12} {expanded:<11} {confidence:<12} {cost:<12}")

def run_scenario(scenario,weather,margin,n):

    name = scenario[0]

    print(f"\n{name}")
    print("Loading graph...")

    origin = scenario[1]
    destination = scenario[2]

    G = load_or_download(origin, destination, margin)
    print(f"  {len(G.nodes)} nodes, {len(G.edges)} edges")

    rows  = []
    paths = {}

    city    = CityRouting(G)
    city.inject_missing_attributes(weather)
    city.get_cost(WEIGHTS)

    # CSP
    t0         = time.perf_counter()
    csp_result = csp_solve(G, origin, destination)
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
            'cost' : None
        })
        paths['CSP'] = csp_result['path']
    else:
        rows.append({'name': 'CSP', 'found': False})

    # A*
    origin     = nearest_node(G, origin)
    goal       = nearest_node(G, destination)
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
            'cost' : None
        })
        paths['A*'] = ast_result['path']
    else:
        rows.append({'name': 'A*', 'found': False})

    # MDP 
    t0 = time.perf_counter()
    value_iteration(city, N_ITERATIONS, goal, GAMMA, EPSILON)
    get_policy(city, GAMMA)
    mdp_path = apply_policy(city,origin,goal)
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
            'cost': compute_path_cost(city,mdp_path)
        })
        paths['MDP'] = mdp_path
    else:
        rows.append({'name': 'MDP', 'found': False})

    print_table(f"{name}", rows)

    center = (
        (scenario[1][0] + scenario[2][0])/2,
        (scenario[1][1] + scenario[2][1])/2,
    )
    m = build_folium_map(G, paths, center)
    m.save(f"maps/map_scenario_{n}.html")
    print(f"\n  map saved to map_scenario_{n}.html")

    for r in rows:
        r['scenario'] = name
    return rows



if __name__ == '__main__':
    
    all_results = []
    
    for n, scenario in enumerate(FINAL_BENCHMARK_PAIRS):
        scenario_rows = run_scenario(scenario, weather=4, margin=200, n=n)
        all_results.extend(scenario_rows)  

    df = pd.DataFrame(all_results)
    
    cols_order = ['scenario', 'name', 'found', 'length_m', 'time_ms', 'cost', 'confidence', 'expanded']
    df = df[cols_order]
    
    df.to_csv("final_benchmark_results.csv", index=False)
    
    print("Bechmark results saved in: 'final_benchmark_results.csv'")
    print(df.head(10))

