# Wheelchair Routing — Team Guide

Written by: Matteo
INFO-H410 — ULB 2024/2025

---

## What we built

We wrote the shared part of the project: everything related to the map.
Our code downloads the Brussels map, enriches it with accessibility data,
and makes it available for each member's CSP implementation.

The two files we are responsible for are: config.py and graph_builder.py.

---

## How the map logic works

### We do not download all of Brussels

When the user says "I want to go from ULB to Gare du Midi", there is no point
in downloading the entire city map. Instead, we identify the minimum area
needed between the two points — a rectangle that contains both of them with
a 500-metre margin on all sides — and download only that.

The 500-metre margin matters: it is what allows the CSP to find alternative
routes when the direct path is blocked by an inaccessible street.

### The map is cached

The first time a pair of locations is searched, the program downloads the map
from OpenStreetMap and saves it as a file in the data/ folder. The next time
anyone searches the same route, the file is loaded directly without touching
the internet — in under a second.

Each pair of locations has its own cache file. Searching a new pair creates a
new file without overwriting the existing ones.

### Every street is enriched with accessibility data

After the download, the system adds the following attributes to every street:

- slope: gradient in %, calculated from the elevation difference between the
  two endpoints of the street
- width: width in metres, taken from OSM tags
- kerb: kerb type (lowered, flush, raised...)
- wheelchair: official accessibility tag (yes / no / limited / unknown)
- surface: surface type (asphalt, cobblestone, gravel...)

When a value is missing — which is common in OSM — a default value defined
in config.py is used. The defaults are conservative: we always assume the
worst case so as not to mislead the user.

### If no path is found, the search area is expanded

When the CSP cannot find any accessible path within the initial area,
the solution is to expand the search radius. You can call load_or_download()
with a larger margin (e.g. 1000m, 1500m) and the system downloads a wider
area. Each different margin generates a separate cache file.

---

## Accessibility constraints

The constraints are defined in config.py and split into three levels.

### Absolute constraints (always active)

These are never relaxed, no matter what:

- kerb = "unknown": the street is removed. If we do not know whether there
  is a dropped kerb, we cannot guarantee accessibility.
- wheelchair = "no": the street is removed. It is explicitly forbidden.

### Strict constraints (normal mode)

- Maximum slope: 8%
- Minimum width: 1.2 metres

### Relaxed constraints (fallback mode)

If no path is found with strict constraints, the system retries with:

- Maximum slope: 15%
- Minimum width: 1.0 metre

The full fallback sequence is:

  Attempt 1: normal area + strict constraints
  Attempt 2: normal area + relaxed constraints
  Attempt 3: expanded area + strict constraints
  Attempt 4: expanded area + relaxed constraints
  Attempt 5: ask the user whether to expand further

---

## How to use our code

### Basic case — getting the graph

```python
from graph_builder import geocode, load_or_download

point_A = geocode("ULB")
point_B = geocode("Gare du Midi")
G = load_or_download(point_A, point_B, margin=500)
```

The graph G you receive is already enriched with all accessibility attributes
and ready to be used by your CSP.

### If you need to expand the search area

```python
G = load_or_download(point_A, point_B, margin=1000)
```

### What the graph G contains

For every edge (u, v, data) in the graph, data contains:

  data['length']      street length in metres
  data['slope']       gradient in %
  data['width']       width in metres
  data['kerb']        kerb type
  data['wheelchair']  yes / no / limited / unknown
  data['surface']     surface type

### How to read the constraints from config.py

```python
from config import STRICT_CONSTRAINTS, RELAXED_CONSTRAINTS, ABSOLUTE_CONSTRAINTS

# Strict constraints
slope_max = STRICT_CONSTRAINTS['slope']   # 8.0
width_min = STRICT_CONSTRAINTS['width']   # 1.2

# Relaxed constraints
slope_max = RELAXED_CONSTRAINTS['slope']  # 15.0
width_min = RELAXED_CONSTRAINTS['width']  # 1.0
```

Do not hardcode constraint values in your code — always import them from
config.py. That way, if we change a value, it updates for everyone.

---

## The files

**config.py** — all parameters: constraints, defaults, confidence score
weights, visualisation colours. If you want to change a constraint value,
change it here.

**graph_builder.py** — the full pipeline. You should never need to modify
this file. Just call load_or_download().

**test_graph_builder.py** — run this to verify everything works on your
machine.

```bash
python test_graph_builder.py
```

**data/** — auto-generated folder containing cached graphs.
Do not commit this folder to GitHub.
