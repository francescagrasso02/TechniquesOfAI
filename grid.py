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
WEATHER = np.random.randint(0,5) #Weather condition ["Sunny","Moderate Rain","Moderate Snow","Extreme Rain","Extreme Snow"]

class CityRouting():
    def __init__(self,nx_graph):
        """
        Initializes the City with the loaded graph
        """
        self.graph = nx_graph
        self.v_s = None
        self.v_s_next = None

    def inject_missing_attributes(self,weather):
        """
        Injects missing attributes. By default a uniform 
        distribution was chosen, this should be reconsidered 
        when we'll do experiments
        
        :param self: CityRouting object
        :param seed: seed value 
        """

        for u,v,key,data in self.graph.edges(keys=True,data=True):
            data['weather'] = weather
            data['cross_slope'] = np.random.uniform(0,2.5)
            data['traversability'] = np.random.choice([0,1])
            data['tactile'] = np.random.choice([0,1])
            data['p_crowd'] = np.random.uniform(0,0.4)

            if 'length' not in data : 
                data['length'] = 1


    def get_cost(self,w):
        """
        Computes cost of each edge depending on the chosen weights
        
        :param self: CityRouting object
        :param w: weights dictionary : w = {"w":w0,"l":w1,"s":w2,"sf":w3}
        :param weather: weather state (initialized at the begining)
        """

        edges_list = list(self.graph.edges(keys=True,data=True))

        attributes = [
            [data.get('width',1), data.get('length',1),data.get('slope',0),data.get('cross_slope',1)
             , data.get('weather',0)] for u,v,k,data in edges_list
        ]
        
        arr_attributes = np.array(attributes)

        s = (arr_attributes - arr_attributes.min(axis=0))/(arr_attributes.max(axis=0) - arr_attributes.min(axis=0)+1e-9)
        
        self.score = s
        cost = (w["w"]*s[:,0] + w["l"]*s[:,1] + w["s"]*s[:,2]*s[:,4] + w["sf"]*s[:,3]*s[:,4])
        
        for i,(u,v,k,data) in enumerate(edges_list):
            data['cost'] = cost[i]
    
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


