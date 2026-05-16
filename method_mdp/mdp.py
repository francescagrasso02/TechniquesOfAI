import numpy as np 
from utils.grid import CityRouting
from utils.graph_builder import load_or_download
import osmnx as ox
from method_astar.benchmark import OD_PAIRS
from time import time
from utils.config import inject_shared_random_attributes
import random

SEED = 42

#Global variables (determined with grid search on the benchmark)
N_ITERATIONS = 10000
GAMMA = 1
EPSILON = 0.1
WEATHER = 4
WEIGHTS = {"w": 0.1, "l": 0.4, "s": 0.3, "sf": 0.2}
PENALTY = -1

def transition(city:CityRouting,st1,st2,a):

    if not city.graph.has_edge(st1,a): #No edge between st1 and a 
        return 0 
    
    #We assume that only the first edge between st1 and a exists 
    edge_data = city.graph[st1][a][0]
    p_crowd = edge_data.get('p_crowd',0)
    
    #if the target is st2 we can move but the probability 
    #depends on the crowd 
    if st2 == a : 
        return 1 - p_crowd
    #st2 is st2 (no move) only happens 
    if st2 == st1 :
        return p_crowd
    
    return 0 


def reward(city:CityRouting,st1,st2):
    #if edge doesn't exist : 
    if not city.graph.has_edge(st1,st2):
        return -10000
    #we take the first edge between st1 and st2, we assume there is only this one
    edge_data = city.graph[st1][st2][0] 
    cost = edge_data.get('cost')
    return -cost


def value_iteration(city:CityRouting,n_iterations,target_node,penalty,epsilon):

    states = list(city.graph.nodes())
    city.v_s = {state:-100000 for state in states}
    city.v_s_next = {state:-100000 for state in states}

    city.v_s[target_node] = 0
    city.v_s_next[target_node] = 0

    for i in range(n_iterations):
        delta = 0
        for st1 in states : 
            if st1 == target_node:
                continue

            v_max = -float('inf')
            successors = list(city.graph.successors(st1))

            if not successors:
                city.v_s_next[st1] = -10000
                continue

            for a in successors :
                v_temp = 0
                r = reward(city,st1,a)
                for st2 in [a,st1] :
                    t = transition(city,st1,st2,a)
                    #Penalty when st1==st2
                    if st1 == st2 :
                        current_r = r+penalty
                    else :
                        current_r = r
                    v_temp += t*(current_r+GAMMA*city.v_s[st2])
                if v_temp > v_max : 
                    v_max = v_temp
            city.v_s_next[st1] = v_max
            if v_max > -50000:
                delta = max(delta, abs(city.v_s[st1] - v_max))
        city.v_s = city.v_s_next.copy()

        if delta < epsilon : 
            print("Value iteration converged!")
            break

def get_policy(city:CityRouting,penalty):
    states = list(city.graph.nodes())
    city.policy = {state:None for state in states}
    for st1 in states : 
        v_max = -float('inf')
        a_max = None
        for a in city.graph.successors(st1) :
            v_temp = 0
            r = reward(city,st1,a)
            for st2 in [a,st1] :
                #Penalty when st1==st2
                if st1 == st2 :
                    current_r = r+penalty
                else :
                    current_r = r
                t = transition(city,st1,st2,a)
                v_temp += t*(current_r+GAMMA*city.v_s[st2])
            if v_temp > v_max : 
                v_max = v_temp
                a_max = a
        city.policy[st1] = a_max

def apply_policy(city:CityRouting,start_node,target_node):
    state = start_node
    path = []
    visited = set()

    while (state != target_node):
        if state is None or state in visited:
            print(f"Error, the policy leads to a deadlock.")
            return None
        
        visited.add(state)
        path.append(state)
        action = city.policy[state]
        state = action

    if state == target_node:
        path.append(target_node)
    return path

def compute_path_cost(city,path):

    if path and len(path) > 1:
        edge_costs = []
        
        for i in range(len(path) - 1):
            u = path[i]      
            v = path[i+1]    
            
            cost = city.graph[u][v][0].get('cost',0)
            edge_costs.append(cost)
                        
        return sum(edge_costs)
    else : 
        return float('inf')
    

def compute_objective_distance(city, path):

    if path and len(path) > 1:
        total_length = 0
        
        for i in range(len(path) - 1):
            u = path[i]      
            v = path[i+1]    
            
            length = city.graph[u][v][0].get('length', 0) 
            total_length += length
            
        return total_length
    else: 
        return float('inf')

def mdp_grid_search(od_pairs, max_iter=10000):
    
    WEATHER = 4

    penalties = [-1,-5,-10,-20]
    epsilons = [0.1,0.01,0.001]

    configs = {}
    n = 0
    for pen in penalties:
        for eps in epsilons:
            configs[f"cfg_{n}"] = {"penalty": pen, "epsilon": eps}
            n += 1

    raw_results = {
        cfg_id: {"times": [], "costs": [], "distances": [], "solved": []} 
        for cfg_id in configs.keys()
    }

    for i, OD in enumerate(od_pairs):
        print(f"\nPath: {i+1}/{len(od_pairs)}")
        A = OD[1]
        B = OD[2] 

        real_graph = load_or_download(A, B, margin=50)
        real_graph = inject_shared_random_attributes(real_graph)
        
        start_node = ox.distance.nearest_nodes(real_graph, X=A[1], Y=A[0])
        target_node = ox.distance.nearest_nodes(real_graph, X=B[1], Y=B[0])

        city = CityRouting(real_graph)

        for cfg_id, params in configs.items():
            t0 = time()
            
            city.get_cost(WEIGHTS, WEATHER)
            
            value_iteration(
                city, 
                max_iter, 
                target_node, 
                penalty=params["penalty"],
                epsilon=params["epsilon"] 
            )
            
            get_policy(city, penalty=params["penalty"])
            path = apply_policy(city, start_node, target_node)
            tf = time() - t0

            if path and len(path) > 0:
                cost = compute_path_cost(city, path)
                distance = compute_objective_distance(city, path) 
                solved = True
            else:
                cost = float('inf')
                distance = float('inf')
                solved = False

            raw_results[cfg_id]["times"].append(tf)
            raw_results[cfg_id]["costs"].append(cost)
            raw_results[cfg_id]["distances"].append(distance)
            raw_results[cfg_id]["solved"].append(solved)

    print("\nFinal results")
    print(f"Weights Profile: {WEIGHTS}\n")
    
    configs_results = {}
    for cfg_id, data in raw_results.items():
        mean_time = np.mean(data["times"])
        all_solved = all(data["solved"])
        
        valid_costs = [c for c in data["costs"] if c != float('inf')]
        mean_cost = np.mean(valid_costs) if valid_costs else float('inf')
        
        valid_distances = [d for d in data["distances"] if d != float('inf')]
        mean_dist = np.mean(valid_distances) if valid_distances else float('inf')
        
        configs_results[cfg_id] = (mean_time, mean_cost, mean_dist, all_solved)
        
        pen_str = f"Pen:{configs[cfg_id]['penalty']:>3}"
        eps_str = f"Eps:{configs[cfg_id]['epsilon']:>5}"
        print(f"{cfg_id} [{pen_str} | {eps_str}]: Time={mean_time:.3f}s, Cost={mean_cost:.3f}, Dist={mean_dist:.1f}m, Solved={all_solved}")

    return configs, configs_results


if __name__ == "__main__":
    configs,configs_results = mdp_grid_search(OD_PAIRS)
    print(configs)
    print(configs_results)