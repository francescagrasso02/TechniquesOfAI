import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from pyvis.network import Network


#Only for test 
np.random.seed(42)
weights = {"w":0.5,"l":0.1,"s":0.3,"sf":0.2}
weather = np.random.randint(0,5) #Weather condition ["Sunny","Moderate Rain","Moderate Snow","Extreme Rain","Extreme Snow"]

class CityGrid():
    def __init__(self,size):
        """
        Initializes a grid of size : size^2 
        representing the city
        
        :param self: CityGrid object
        :param size: Grid size value
        """
        self.size = size
        num_nodes = size**2
        graph = {}

        for i in range(num_nodes):
            row = i // size
            col = i % size
            neighbors = []

            #Check Up
            if row > 0:
                neighbors.append(i - size)
            
            #Check Down
            if row < size - 1:
                neighbors.append(i + size)
            
            #Check Left 
            if col > 0:
                neighbors.append(i - 1)
            
            #Check Right 
            if col < size - 1:
                neighbors.append(i + 1)

            graph[i] = neighbors
        
        self.graph = graph
        self.attributes = None

    def generate_attributes(self,seed):
        """
        Generates random attributes for each 
        edge on the grid. By default a uniform 
        distribution was chosen, this should be reconsidered 
        when we'll do experiments
        
        :param self: CityGrid object
        :param seed: seed value 
        """
        keys = list(self.graph.keys())
        dict_attributes = {}
        try : 
            for node in keys:
                for neighbor in self.graph[node]:
                    edge_attributes = []            
                    edge_attributes.append(np.random.uniform(4,18.5)) #width in m
                    edge_attributes.append(np.random.uniform(200,2000)) #lenght in m
                    edge_attributes.append(np.random.uniform(0.7,3.9)) #Horizontal slope in °
                    edge_attributes.append(np.random.uniform(0,2.5)) #Cross slope in °
                    edge_attributes.append(np.random.choice([0,1])) #Kerb (accessible or not)
                    edge_attributes.append(np.random.randint(0,5)) #Surface type ["Concrete","Asphalt","Brick","Cobblestone","Gravel"]
                    edge_attributes.append(weather) #Weather condition ["Sunny","Moderate Rain","Moderate Snow","Extreme Rain","Extreme Snow"]
                    edge_attributes.append(np.random.choice([0,1])) #Traversability depends on crowd value that should change during the experiment for MPD, constant for search
                    edge_attributes.append(np.random.choice([0,1])) #Tactile pavement
                    dict_attributes[(node,neighbor)] = edge_attributes
        except Exception as e : 
            print(f"Error, {e}")

        self.attributes = dict_attributes

        return dict_attributes
    
    def get_cost(self,w,weather):
        """
        Computes cost of each node depending on the chosen weights
        
        :param self: CityGrid object
        :param w: weights dictionary : w = {"w":w0,"l":w1,"s":w2,"sf":w3}
        :param weather: weather state (initialized at the begining)
        """
        dict_cost = {}  
        keys_mapping = list(self.attributes.keys())
        
        arr_attributes = np.array([self.attributes[key] for key in keys_mapping])
        
        s = (arr_attributes - arr_attributes.min(axis=0))/(arr_attributes.max(axis=0) - arr_attributes.min(axis=0)+1e-9)
        self.score = s
        cost = (w["w"] * s[:,0] + w["l"] * s[:,1] + w["s"] * s[:,2] * s[:,6] + w["sf"] * s[:,3] * s[:,6])
        for k in range(len(keys_mapping)):
            dict_cost[keys_mapping[k]] = cost[k]
        self.cost = dict_cost
    
    def plot_city_grid(self, filename="map.html"):
        """
        Generates an html file displaying the graph 
        :param self: CityGrid object
        :filename: file name (.html)
        """
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True, notebook=False)
        
        for i in range(self.size**2):
            row, col = i // self.size, i % self.size
            net.add_node(i, label=str(i), x=col*200, y=row*200, physics=False)

        for (u, v), val in self.cost.items():
            c_norm = max(0, min(1, val)) 
            color = f"rgba({int(255*c_norm)}, {int(255*(1-c_norm))}, 0, 1)"
            title = f"Cost: {val:.2f}"
            net.add_edge(u, v, color=color, width=2 + (val * 5), title=title)

        net.write_html(filename)


city = CityGrid(size=10)
city.generate_attributes(42)
city.get_cost(weights,weather)
city.plot_city_grid(filename="map.html")

