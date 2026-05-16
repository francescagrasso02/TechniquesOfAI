# graph_builder.py — Downloads and enriches the map from OpenStreetMap
# INFO-H410 — Wheelchair Routing CSP — ULB

import os
import math
import pickle
import requests
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from config import DATA_DIR, DEFAULT_MARGIN, DEFAULTS


_geolocator = Nominatim(user_agent="wheelchair_routing_ulb", timeout=10)


def geocode(name: str) -> tuple[float, float]:
    """
    Converts a place name to GPS coordinates (lat, lon).
    If multiple results are found, shows up to 3 and asks the user to choose.

    Example:
        geocode("ULB")          -> (50.8122, 4.3822)
        geocode("Gare du Midi") -> (50.8354, 4.3360)
    """

    query = name.strip()
    if "brussels" not in query.lower() and "bruxelles" not in query.lower():
        query += ", Brussels, Belgium"

    try:
        results = _geolocator.geocode(query, exactly_one=False, limit=3)
    except GeocoderTimedOut:
        raise ValueError("Request timed out. Check your connection and try again.")

    if not results:
        raise ValueError(f"Place not found: '{name}'. Try a more specific name.")

    if len(results) == 1:
        return (results[0].latitude, results[0].longitude)

    print(f"\nFound {len(results)} places with this name:")
    for i, r in enumerate(results, start=1):
        print(f"  {i}. {r.address}")

    while True:
        choice = input(f"\nWhich one did you mean? (1/{'/'.join(str(i) for i in range(2, len(results)+1))}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(results):
            return (results[int(choice) - 1].latitude, results[int(choice) - 1].longitude)
        print("  Invalid choice, try again.")


def calculate_bbox(point_A: tuple, point_B: tuple, margin: int = DEFAULT_MARGIN) -> tuple:
    """
    Calculates the minimum bounding box containing A and B, with a margin in metres.
    Returns (south, west, north, east).

    Example:
        calculate_bbox((50.81, 4.38), (50.84, 4.40), margin=500)
        -> (50.805, 4.373, 50.845, 4.407)
    """

    lat_min = min(point_A[0], point_B[0])
    lat_max = max(point_A[0], point_B[0])
    lon_min = min(point_A[1], point_B[1])
    lon_max = max(point_A[1], point_B[1])

    lat_center = (lat_min + lat_max) / 2
    delta_lat  = margin / 111_000
    delta_lon  = margin / (111_000 * math.cos(math.radians(lat_center)))

    return (
        lat_min - delta_lat,   # south
        lon_min - delta_lon,   # west
        lat_max + delta_lat,   # north
        lon_max + delta_lon,   # east
    )


def download_graph(point_A: tuple, point_B: tuple, margin: int = DEFAULT_MARGIN):
    """
    Downloads from OpenStreetMap only the area needed between A and B.
    Returns the raw graph (osmnx MultiDiGraph).

    Example:
        G = download_graph((50.81, 4.38), (50.84, 4.40))
    """

    south, west, north, east = calculate_bbox(point_A, point_B, margin)
    ox.settings.timeout = 180

    center_lat = (south + north) / 2
    center_lon = (west  + east)  / 2

    # Radius = half the bbox diagonal + margin.
    # Using the full diagonal would download roughly twice the necessary area.
    diagonal = ox.distance.great_circle(south, west, north, east)
    dist = diagonal / 2 + margin

    print(f"  Downloading map from OpenStreetMap...")
    print(f"  Centre: ({center_lat:.4f}, {center_lon:.4f}), radius: {int(dist)}m")

    G = ox.graph_from_point(
        (center_lat, center_lon),
        dist=dist,
        network_type='walk',
        simplify=True,
    )

    print(f"  Map downloaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


def enrich_graph(G):
    """
    Adds missing accessibility attributes to every edge in the graph.
    - slope     : calculated from the elevation difference between nodes
    - width     : width in metres (from OSM or default)
    - kerb      : kerb type (from OSM or default)
    - wheelchair: accessibility tag (from OSM or default)
    - surface   : surface type (from OSM or default)

    Returns the enriched graph.
    """

    print("  Downloading node elevations...")
    G = _add_elevations(G)
    print("  Elevations downloaded.")

    for u, v, data in G.edges(data=True):
        data['width']      = _parse_float(data.get('width'),      DEFAULTS['width'])
        data['kerb']       = data.get('kerb',       DEFAULTS['kerb'])
        data['wheelchair'] = data.get('wheelchair', DEFAULTS['wheelchair'])
        data['surface']    = data.get('surface',    DEFAULTS['surface'])

        elev_u = G.nodes[u].get('elevation', 0)
        elev_v = G.nodes[v].get('elevation', 0)
        length = data.get('length', 1)

        data['slope'] = abs(elev_v - elev_u) / length * 100 if length > 0 else DEFAULTS['slope']

    print(f"  Graph enriched: {len(G.edges)} edges processed")
    return G


def _add_elevations(G):
    """
    Downloads elevations for all nodes using the Open-Elevation API (free, no key needed).
    Sends requests in batches of 100 to avoid overloading the server.
    If a batch fails, elevation defaults to 0 for those nodes.
    """

    nodes = list(G.nodes(data=True))
    batch_size = 100

    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]
        locations = [{"latitude": data['y'], "longitude": data['x']} for _, data in batch]

        try:
            response = requests.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": locations},
                timeout=30
            )
            results = response.json().get("results", [])
            for j, (node_id, _) in enumerate(batch):
                G.nodes[node_id]['elevation'] = results[j].get('elevation', 0) if j < len(results) else 0
        except Exception:
            for node_id, _ in batch:
                G.nodes[node_id]['elevation'] = 0

    return G


def _parse_float(value, default: float) -> float:
    """
    Converts an OSM value to float.
    OSM sometimes returns strings like '1.5' or '1.5;2.0' (multiple measurements).
    Falls back to the default on any error.
    """

    if value is None:
        return default
    try:
        return float(str(value).split(';')[0].strip())
    except (ValueError, TypeError):
        return default


def save_graph(G, point_A: tuple, point_B: tuple) -> str:
    """
    Saves the graph to cache as a .pkl file in the data/ folder.
    The filename is based on the coordinates of A and B.

    Example:
        save_graph(G, (50.81, 4.38), (50.84, 4.40))
        -> data/graph_50.8100_4.3800_50.8400_4.4000.pkl
    """

    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"graph_{point_A[0]:.4f}_{point_A[1]:.4f}_{point_B[0]:.4f}_{point_B[1]:.4f}.pkl"
    filepath = os.path.join(DATA_DIR, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(G, f)

    print(f"  Graph saved to cache: {filepath}")
    return filepath


def load_or_download(point_A: tuple, point_B: tuple, margin: int = DEFAULT_MARGIN):
    """
    Main entry point. Checks whether the graph is already cached:
    - If yes: loads from .pkl (instant)
    - If no:  downloads, enriches, and saves it

    Example:
        G = load_or_download((50.81, 4.38), (50.84, 4.40))
    """

    filename = f"graph_{point_A[0]:.4f}_{point_A[1]:.4f}_{point_B[0]:.4f}_{point_B[1]:.4f}.pkl"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print(f"  Map loaded from cache: {filepath}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    print(f"  No cache found, downloading from OpenStreetMap...")
    G = download_graph(point_A, point_B, margin)
    G = enrich_graph(G)
    save_graph(G, point_A, point_B)
    return G