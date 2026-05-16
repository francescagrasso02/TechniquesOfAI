# config.py — Global parameters of the project
# INFO-H410 — Wheelchair Routing CSP — ULB

# Cache
DATA_DIR       = "data"   # folder where graphs are saved
DEFAULT_MARGIN = 500      # search area margin in metres around the A-B route

# Absolute constraints — always active, regardless of mode
ABSOLUTE_CONSTRAINTS = {
    "kerb":       lambda v: v != "unknown",  # kerb info must be available
    "wheelchair": lambda v: v != "no",       # street must not be explicitly forbidden
}

# Strict constraints — normal mode
STRICT_CONSTRAINTS = {
    "slope": 8.0,   # maximum slope in %
    "width": 1.2,   # minimum width in metres
}

# Relaxed constraints — fallback if no path is found
RELAXED_CONSTRAINTS = {
    "slope": 15.0,  # maximum slope in %
    "width": 1.0,   # minimum width in metres
}

# Confidence score weights
WEIGHTS = {
    "kerb":       0.30,  # critical: without a dropped kerb, passage is impossible
    "wheelchair": 0.30,  # critical: official accessibility tag
    "slope":      0.20,  # important: wrong slope estimate can be dangerous
    "width":      0.15,  # medium: uncomfortable but not a blocker
    "surface":    0.05,  # low: inconvenient but manageable
}

# Confidence thresholds — (minimum_score, colour, user_message)
CONFIDENCE_LEVELS = [
    (90, "green",  "Route is reliable"),
    (70, "yellow", "Route is fairly reliable"),
    (50, "orange", "Some data estimated, verification recommended"),
    ( 0, "red",    "Insufficient data, verify before setting off"),
]

# Default values — used when OSM data is missing
DEFAULTS = {
    "slope":      0.0,
    "width":      1.5,
    "kerb":       "unknown",
    "wheelchair": "unknown",
    "surface":    "unknown",
}

# Colours for map visualisation
COLORS = {
    "accessible":  "#CCCCCC",  # light grey — streets kept by the CSP
    "blocked":     "#FFAAAA",  # soft red   — streets removed by the CSP
    "path":        "#0000FF",  # blue       — chosen route
    "origin":      "#00AA00",  # green      — starting point
    "destination": "#FF4400",  # red        — destination
}

import numpy as np 

def inject_shared_random_attributes(G, seed=42):
    """
    Injects random attributes into the graph edges.
    Using a fixed seed ensures that A*, CSP, and MDP all solve 
    the exact same problem.
    """
    np.random.seed(seed)

    surface_choices = ['asphalt', 'paving_stones', 'compacted', 'sett', 'cobblestone', 'gravel']

    for u, v, key, data in G.edges(keys=True, data=True):
        if 'slope' not in data or data['slope'] == 0:
            data['slope'] = float(np.random.uniform(0, 10))
            
        if 'width' not in data:
            data['width'] = float(np.random.uniform(0.8, 2.5))

        if 'surface' not in data or data.get('surface') == 'unknown':
            data['surface'] = np.random.choice(surface_choices)
        
        data['surface_type'] = data['surface']

        if 'wheelchair' not in data or data.get('wheelchair') == 'unknown':
            data['wheelchair'] = np.random.choice(['yes', 'limited', 'no'], p=[0.80, 0.15, 0.05])
            
        if 'kerb' not in data or data.get('kerb') == 'unknown':
            data['kerb'] = np.random.choice(['lowered', 'raised'], p=[0.70, 0.30])
            
        if 'cross_slope' not in data:
            data['cross_slope'] = float(np.random.uniform(0, 2.5))
        if 'traversability' not in data:
            data['traversability'] = int(np.random.choice([0, 1], p=[0.9, 0.1]))
        if 'p_crowd' not in data:
            data['p_crowd'] = float(np.random.uniform(0, 0.4))
            
        if 'length' not in data: 
            data['length'] = 1.0

    return G
