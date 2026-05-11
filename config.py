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
