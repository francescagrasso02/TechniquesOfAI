import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from pyvis.network import Network
import networkx as nx 
from graph_builder import load_or_download

SIZE = 5

#Only for test 
np.random.seed(42)
WEIGHTS = {"w":0.5,"l":0.1,"s":0.3,"sf":0.2}

surface_map = {
            'asphalt': 0, 'concrete': 0, 'paving_stones:smooth': 0,
            'paving_stones': 1, 'slabs': 1,
            'compacted': 2, 'fine_gravel': 2,
            'sett': 3, 'cobblestone:flattened': 3,
            'cobblestone': 4,
            'gravel': 5, 'sand': 5, 'earth': 5
        }


class CityRouting():
    def __init__(self,nx_graph):
        """
        Initializes the City with the loaded graph
        """
        self.graph = nx_graph
        self.v_s = None
        self.v_s_next = None


    def get_cost(self, w, weather):

        edges_list = list(self.graph.edges(keys=True, data=True))
        surface_map = {
            'asphalt': 0, 'concrete': 0, 'paving_stones:smooth': 0,
            'paving_stones': 1, 'slabs': 1, 'unknown': 1,
            'compacted': 2, 'fine_gravel': 2,
            'sett': 3, 'cobblestone:flattened': 3,
            'cobblestone': 4,
            'gravel': 5, 'sand': 5, 'earth': 5
        }

        attributes = []
        for u, v, k, data in edges_list:
            raw_width = data.get('width', 1.5)
            try:
                if isinstance(raw_width, list):
                    width = float(raw_width[0])
                else:
                    width = float(raw_width)
            except (ValueError, TypeError):
                width = 1.5

            try:
                length = float(data.get('length', 1))
            except (ValueError, TypeError):
                length = 1

            try:
                slope = float(data.get('slope', 0))
            except (ValueError, TypeError):
                slope = 0.0

            raw_surface = data.get('surface_type', 'unknown')
            if isinstance(raw_surface, str):
                surface_type = float(surface_map.get(raw_surface, 1))
            else:
                try:
                    surface_type = float(raw_surface)
                except (ValueError, TypeError):
                    surface_type = 1

            try:
                edge_weather = float(data.get('weather', weather)) 
            except (ValueError, TypeError):
                edge_weather = weather

            try:
                traversability = float(data.get('traversability', 0))
            except (ValueError, TypeError):
                traversability = 0

            attributes.append([width, length, slope, surface_type, edge_weather, traversability])
        
        arr_attributes = np.array(attributes, dtype=float)
        features = arr_attributes[:, :5] 
        s = np.zeros_like(features, dtype=float)

        phys_min = features[:, :3].min(axis=0)
        phys_max = features[:, :3].max(axis=0)
        s[:, :3] = (features[:, :3] - phys_min)/(phys_max - phys_min + 1e-9)
        
        s[:, 0] = 1.0 - s[:, 0]

        MAX_SURFACE_TYPE = 5.0
        s[:, 3] = features[:, 3]/MAX_SURFACE_TYPE

        MAX_WEATHER = 4.0
        s[:, 4] = features[:, 4]/MAX_WEATHER

        self.score = s
        base_cost = (w["w"]*s[:,0]+w["l"]*s[:,1]+ w["s"]*s[:,2]*s[:,4] + w["sf"]*s[:,3]*s[:,4])
        
        traversability_vector = arr_attributes[:, 5]
        penalty_value = 10000
        
        for i, (u, v, k, data) in enumerate(edges_list):
            final_edge_cost = base_cost[i] + (traversability_vector[i] * penalty_value)
            data['cost'] = max(1e-4, final_edge_cost)
    

    def plot_city_grid(self, filename="map.html",path=None):
        """
        Generates an html file displaying the graph 
        :param self: CityRouting object
        :filename: file name (.html)
        """
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True, notebook=False)
        graph_vis = nx.DiGraph()

        if path: 
            path_edges = set(zip(path[:-1],path[1:]))

        for u,v,data in self.graph.edges(data=True):
            val= data.get('cost',0)
            c_norm = max(0,min(1,val))
            edge_color = f"rgba({int(255*c_norm)},{int(255*(1-c_norm))},0, 1)"
            node_color_u = "#97c2fc"
            node_color_v = "#97c2fc"
            node_size = 10
            edge_width= 2 + (val*5)
        

            if path and (u,v) in path_edges:
                edge_color = "#00FFFF"
                edge_width = 40


            if path and u in path:
                node_color_u = "#00FFFF"
                node_size = 15
            
            if path and v in path:
                node_color_v = "#00FFFF"
                node_size = 15
            
            graph_vis.add_node(u,title=str(u),color=node_color_u,size=node_size)
            graph_vis.add_node(v,title=str(v),color=node_color_v,size=node_size)
            graph_vis.add_edge(u,v, color=edge_color, width=edge_width, title=f"Cost: {val:.2f}")
        
        net.from_nx(graph_vis)
        net.write_html(filename)


