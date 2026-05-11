# test_graph_builder.py — Tests for graph_builder.py
# INFO-H410 — Wheelchair Routing CSP — ULB
import os
import pickle
import time

from config import DATA_DIR
from graph_builder import geocode, calculate_bbox, download_graph, enrich_graph, load_or_download, save_graph


def test_geocode():
    print("TEST 1 - geocode()")

    locations = [
        "ULB",               # should append Brussels automatically
        "Gare du Midi",      # known location in Brussels
        "Central Station",   # ambiguous name, should suggest options
        "ajsdklajsdlkajsd",  # non-existent place, should raise error
    ]

    coords = {}
    for name in locations:
        print(f"\nSearching: '{name}'")
        try:
            c = geocode(name)
            coords[name] = c
            print(f"  found: {c}")
        except ValueError as e:
            print(f"  error: {e}")

    return coords


def test_calculate_bbox(coords):
    print("\nTEST 2 - calculate_bbox()")

    if "ULB" not in coords or "Gare du Midi" not in coords:
        print("  skipped: geocode did not find ULB or Gare du Midi")
        return None

    A = coords["ULB"]
    B = coords["Gare du Midi"]
    south, west, north, east = calculate_bbox(A, B, margin=500)

    print(f"\n  A     = {A}")
    print(f"  B     = {B}")
    print(f"  south = {south:.5f}")
    print(f"  west  = {west:.5f}")
    print(f"  north = {north:.5f}")
    print(f"  east  = {east:.5f}")

    assert north > south, "north must be greater than south"
    assert east  > west,  "east must be greater than west"
    print("\n  sanity check passed")

    return (A, B)


def test_download_graph(points):
    print("\nTEST 3 - download_graph()")

    if points is None:
        print("  skipped: no points available")
        return None

    A, B = points
    print("\nDownloading map between ULB and Gare du Midi (margin 500m)...")
    G = download_graph(A, B, margin=500)

    assert len(G.nodes) > 0, "graph has no nodes"
    assert len(G.edges) > 0, "graph has no edges"
    print(f"  sanity check passed")
    print(f"  nodes:  {len(G.nodes)}")
    print(f"  edges:  {len(G.edges)}")

    return G


def test_enrich_graph(G, points):
    print("\nTEST 4 - enrich_graph()")

    if G is None:
        print("  skipped: no graph available")
        return None

    G = enrich_graph(G)

    edges_without_slope = sum(1 for u, v, data in G.edges(data=True) if 'slope' not in data)
    print(f"\n  edges without slope: {edges_without_slope} / {len(G.edges)}")
    assert edges_without_slope == 0, "some edges are missing the slope attribute"

    print("\n  sample of 3 enriched edges:")
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        if i >= 3:
            break
        print(f"    edge {i+1}: slope={data.get('slope', '?'):.1f}%  "
              f"width={data.get('width', '?')}m  "
              f"kerb={data.get('kerb', '?')}  "
              f"wheelchair={data.get('wheelchair', '?')}  "
              f"surface={data.get('surface', '?')}")

    print("\n  enrich_graph() completed")
    return G, points


def test_load_or_download(enrich_result):
    print("\nTEST 5 - load_or_download() [cache]")

    if enrich_result is None:
        print("  skipped: no graph available")
        return

    G, (A, B) = enrich_result

    print("\n  saving to cache...")
    save_graph(G, A, B)

    print("\n  loading from cache...")
    start = time.time()
    G2 = load_or_download(A, B)
    elapsed = round(time.time() - start, 2)

    assert len(G2.nodes) == len(G.nodes), "cached graph has a different number of nodes"
    print(f"  cache works, loaded in {elapsed}s")
    print(f"  nodes: {len(G2.nodes)}, edges: {len(G2.edges)}")


if __name__ == "__main__":

    # Fixed coordinates to avoid re-geocoding every run
    A = (50.8116064, 4.380511)   # ULB
    B = (50.8364862, 4.337896)   # Gare du Midi

    test_geocode()
    test_calculate_bbox({"ULB": A, "Gare du Midi": B})

    filename = f"graph_{A[0]:.4f}_{A[1]:.4f}_{B[0]:.4f}_{B[1]:.4f}.pkl"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print("\ncache found, skipping download and enrich")
        with open(filepath, 'rb') as f:
            G = pickle.load(f)
        print(f"  graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
        result = (G, (A, B))
    else:
        G      = test_download_graph((A, B))
        result = test_enrich_graph(G, (A, B))
        if result is not None:
            save_graph(result[0], A, B)

    test_load_or_download(result)

    print("\nall tests completed")