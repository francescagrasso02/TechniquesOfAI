import numpy as np 
from grid import SIZE,WEIGHTS,WEATHER,CityRouting
from graph_builder import load_or_download
import osmnx as ox

SEED = 42

#Global variables
STATES = [k for k in range(SIZE**2)]
N_ITERATIONS = 1000
GAMMA = 0.99
#Transition function settings 

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


def value_iteration(city:CityRouting,n_iterations,target_node,gamma):

    delta = 0
    epsilon = 1e-3
    states = list(city.graph.nodes())
    city.v_s = {state:0 for state in states}
    city.v_s_next = {state:0 for state in states}
    for i in range(n_iterations):
        for st1 in states : 

            if st1 == target_node:
                city.v_s_next[st1] = 0
                continue

            v_max = -float('inf')
            successors = city.graph.successors(st1)

            if not successors:
                city.v_s_next[st1] = -10000
                continue

            for a in successors :
                v_temp = 0
                for st2 in [a,st1] :
                    t = transition(city,st1,st2,a)
                    r = reward(city,st1,st2)
                    v_temp += t*(r+gamma*city.v_s[st2])
                if v_temp > v_max : 
                    v_max = v_temp
            city.v_s_next[st1] = v_max
            delta = max(delta,abs(city.v_s[st1] - v_max))
        city.v_s = city.v_s_next.copy()

        if delta < epsilon : 
            print("Value iteration converged!")
            break

def get_policy(city:CityRouting,gamma):
    states = list(city.graph.nodes())
    city.policy = {state:None for state in states}
    for st1 in states : 
        v_max = -float('inf')
        a_max = None
        for a in city.graph.successors(st1) :
            v_temp = 0
            for st2 in [a,st1] :
                t = transition(city,st1,st2,a)
                r = reward(city,st1,st2)
                v_temp += t*(r+gamma*city.v_s[st2])
            if v_temp > v_max : 
                v_max = v_temp
                a_max = a
        city.policy[st1] = a_max

def apply_policy(city:CityRouting,start_node,target_node):
    state = start_node
    path = []
    while (state != target_node):
        path.append(state)
        action = city.policy[state]
        state = action
    return path

if __name__ == "__main__":
    A = (50.8116, 4.3805) #ULB
    B = (50.8164000, 4.382400) #Cimetière d'Ixelles
    
    real_graph = load_or_download(A, B,margin=50)

    start_node = ox.distance.nearest_nodes(real_graph,X=A[1],Y=A[0])
    target_node = ox.distance.nearest_nodes(real_graph,X=B[1],Y=B[0])
    
    city = CityRouting(real_graph)
    
    WEATHER = np.random.randint(0,5)
    WEIGHTS = {"w":0.5, "l":0.1, "s":0.3, "sf":0.2}
    
    city.inject_missing_attributes(WEATHER)
    city.get_cost(WEIGHTS)
    
    print("------------------Value Iteration--------------------------")
    print("Target node:",target_node)
    value_iteration(city,N_ITERATIONS,target_node,GAMMA)
    print("------------------Getting Policy---------------------------")
    get_policy(city,GAMMA)
    print("------------------Applying Policy---------------------------")
    path = apply_policy(city,start_node,target_node)
    print("Found path :",path)
    print("------------------Plotting graph---------------------------")
    city.plot_city_grid(path=path,filename='real_map.html')
