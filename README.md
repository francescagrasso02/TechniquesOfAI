# Wheelchair-Accessible Routing in Brussels — INFO-H410

Route planning for wheelchair users in Brussels, comparing three AI techniques
against an unconstrained shortest-path baseline on real OpenStreetMap data.

Université Libre de Bruxelles — 2024/2025  
[Code](https://github.com/francescagrasso02/TechniquesOfAI)

---

## What it does

Given an origin and a destination in Brussels, the system finds the most
wheelchair-accessible walking route between them. Four approaches are compared
on the same street graph:

- **Baseline** — unconstrained Dijkstra on physical length. No accessibility
  filtering. Represents what a non-disabled pedestrian would take and serves as
  a lower bound on path length.
- **CSP** — hard constraint filtering: inaccessible edges are removed before
  running Dijkstra on the residual graph. Falls back through a four-step
  escalation if no path is found.
- **A\*** — haversine heuristic search with soft penalty costs. Accessibility
  violations increase edge cost rather than removing the edge, so a path is
  always returned.
- **MDP** — value iteration over the full graph computes a global accessibility
  policy, accounting for stochastic crowds and weather conditions.

---

## Repository structure

```
.
├── main.py                       # entry point: runs all benchmark scenarios
├── utils/
│   ├── config.py                 # global parameters, constraints, inject_shared_random_attributes
│   ├── graph_builder.py          # OSM download, elevation enrichment, cache
│   └── grid.py                   # CityRouting class and edge cost computation for MDP
├── method_csp/
│   └── csp.py                    # build_feasible_graph, find_path, confidence_score, solve
├── method_astar/
│   ├── astar.py                  # A* and Dijkstra, accessibility cost function
│   └── benchmark.py              # standalone A* vs Dijkstra benchmark
├── method_mdp/
│   └── mdp.py                    # value_iteration, get_policy, apply_policy, grid search
├── data/                         # cached graph pickle files (created on first run)
├── maps/                         # HTML interactive maps, one per scenario
├── final_benchmark_results.csv   # output of main.py
└── benchmark_results.csv         # output of method_astar/benchmark.py
```

---

## Installation

Python 3.10 or later is required.

```bash
pip install osmnx networkx geopy folium numpy pandas
```

osmnx handles most geospatial dependencies. Elevation data is fetched from
SRTM at runtime and cached automatically by osmnx.

---

## Reproducing the results

### Main benchmark (all four methods, 8 scenarios)

```bash
python main.py
```

Runs all scenarios defined in `FINAL_BENCHMARK_PAIRS`, writes one HTML map per
scenario to `maps/`, and saves a summary to `final_benchmark_results.csv`.

The random seed is fixed at 42 inside `inject_shared_random_attributes`.
Any machine with the same dependencies produces identical results.

Expected output per scenario:

```
Rogier -> Basilique Koekelberg
Loading graph...
  11575 nodes, 33566 edges
  CSP attempts: strict/default(no), strict/larger(no), relaxed/default(ok)
Value iteration converged!

  approach   found    length (m)    time (ms)    expanded    confidence   cost
  -------------------------------------------------------------------------------
  Baseline   yes      3407          31.7         -           -            -
  CSP        yes      3600          2599.6       -           85.0%        -
  A*         yes      3810          26.5         4303        -            -
  MDP        yes      4148          9453.2       -           -            30046.5

  map saved to maps/map_scenario_6.html
```

### A\* vs Dijkstra benchmark

```bash
python method_astar/benchmark.py
```

Runs A\* and Dijkstra on 8 OD pairs and saves `benchmark_results.csv` and
`benchmark_expansions.png`. On average, A\* expands 19% of the nodes Dijkstra
explores (4.4× faster) while finding the same optimal path.

### MDP hyperparameter grid search

```bash
python method_mdp/mdp.py
```

Runs a grid search over 4 crowd penalties × 3 convergence thresholds.
Prints a summary table and saves `mdp_tradeoff_time_cost.png`. The
configuration used in the main benchmark: penalty = −1, ε = 0.1.

---

## How the graph is built

### Bounding box download

Only the minimum area between origin and destination is downloaded from
OpenStreetMap, extended by a configurable margin (default 200 m in the
benchmark). This avoids loading all of Brussels and reduces bandwidth and
energy consumption.

### Edge enrichment

After download, every edge receives five accessibility attributes:

| Attribute | Source | Default when missing |
|-----------|--------|----------------------|
| slope (%) | Computed from SRTM node elevations | 0.0 |
| width (m) | OSM tag | 1.5 |
| kerb | OSM tag | unknown |
| wheelchair | OSM tag | unknown |
| surface | OSM tag | unknown |

Defaults are conservative: the system never assumes accessibility where
data is missing.

### Shared random attributes

OSM accessibility tags cover fewer than 15% of Brussels edges. To enable a
meaningful benchmark where constraints actually bind, `inject_shared_random_
attributes(G, seed=42)` fills missing attributes with reproducible random
values before any method runs. All four methods receive the same graph,
guaranteeing a fair comparison.

### Cache

After the first download for a given area, the enriched graph is saved to
`data/` as a pickle file. Subsequent runs load from cache in under a second
without touching the network.

---

## Accessibility constraints

Defined in `utils/config.py`.

**Absolute — never relaxed:**

| Tag | Condition | Effect |
|-----|-----------|--------|
| `wheelchair` | `= no` | Edge removed |

**Strict — default mode:**

| Attribute | Threshold |
|-----------|-----------|
| slope | ≤ 8 % |
| width | ≥ 1.2 m |

**Relaxed — fallback:**

| Attribute | Threshold |
|-----------|-----------|
| slope | ≤ 15 % |
| width | ≥ 1.0 m |

---

## CSP escalation (4 steps)

When the initial attempt fails, the CSP escalates along two axes —
constraint severity and search area — in this order:

1. Strict constraints, default bounding box
2. Strict constraints, larger bounding box (margin × 3)
3. Relaxed constraints, default bounding box
4. Relaxed constraints, larger bounding box

The ordering puts safety first: the system searches a wider area before
loosening accessibility requirements. If all four steps fail, the system
reports no accessible route rather than returning an unsafe path.

---

## A\* cost function

Edges are never removed. Accessibility violations add equivalent-metre penalties:

| Condition | Penalty |
|-----------|---------|
| slope > 5 % | (slope − 5) × 10 m, progressive |
| slope > 8 % | additional flat +20 m |
| width < 1.5 m | +8 m |
| width < 1.2 m | additional +12 m |
| cobblestone / sett surface | +8 m |
| gravel / sand | +10 m |
| ground / dirt / grass | +6 m |
| surface unknown | +2 m |
| kerb raised | +5 m |
| kerb unknown | +3 m |
| wheelchair unknown / limited | +2 m |
| wheelchair = no | +9999 m (effectively blocked) |

The haversine heuristic is admissible, so A\* returns the cost-optimal path.

---

## MDP setup

Each graph node is a state; each directed edge is an action. Edge costs are
computed by `CityRouting.get_cost(WEIGHTS, weather)`:

- `WEIGHTS = {"w": 0.1, "l": 0.4, "s": 0.3, "sf": 0.2}` (wheelchair, length, slope, surface)
- `weather` is an integer 0–5 scaling slope and surface penalties

Stochastic transitions model crowd density via `p_crowd` per edge: the agent
reaches the next node with probability `1 − p_crowd` or stays in place with
probability `p_crowd`.

Value iteration parameters (selected by grid search):

| Parameter | Value |
|-----------|-------|
| γ (discount factor) | 1.0 |
| ε (convergence threshold) | 0.1 |
| crowd wait penalty | −1 |
| max iterations | 10 000 |

---

## Confidence score

Every CSP solution reports a confidence score: the weighted fraction of edge
attributes on the route that are real OSM values rather than defaults.

| Attribute | Weight |
|-----------|--------|
| kerb | 30 % |
| wheelchair | 30 % |
| slope | 20 % |
| width | 15 % |
| surface | 5 % |

| Score | Meaning |
|-------|---------|
| 90–100 % | Route is reliable |
| 70–89 % | Route is fairly reliable |
| 50–69 % | Some data estimated — verify before setting off |
| 0–49 % | Insufficient data — verify before setting off |

---

## A note on OSM data quality

Results may look worse than expected. This reflects the real state of
accessibility data in Brussels, not a flaw in the algorithms.

Fewer than 15% of Brussels edges carry explicit kerb or wheelchair tags.
The remaining edges receive conservative defaults, which the CSP treats as
potentially inaccessible — causing detours and occasional failure even in
areas that are physically accessible. The algorithm is being honest about
what the data says.

Slope values are derived from 30 m SRTM elevation data; on very short
segments this can introduce noise. Width and wheelchair tags are maintained
by volunteers and may be outdated.

The confidence score quantifies this uncertainty directly. The ~85% score
observed across all benchmark scenarios reflects the shared injected
substrate rather than raw OSM coverage, which is far sparser.
