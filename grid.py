import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Only for test 
weights = {"w":0.5,"l":0.1,"s":0.3,"sf":0.2}

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
        node on the grid. By default a uniform 
        distribution was chosen, this should be reconsidered 
        when we'll do experiments
        
        :param self: CityGrid object
        :param seed: seed value 
        """
        np.random.seed(seed)
        attributes = np.zeros((self.size**2,9))

        attributes[:,0]=np.random.uniform(4,18.5,size=self.size**2) #width in m
        attributes[:,1]=np.random.uniform(200,2000,size=self.size**2) #lenght in m
        attributes[:,2]=np.random.uniform(0.7,3.9,size=self.size**2) #Horizontal slope in °
        attributes[:,3]=np.random.uniform(0,2.5,size=self.size**2) #Cross slope in °
        attributes[:,4]=np.random.choice([0,1],size=self.size**2) #Kerb (accessible or not)
        attributes[:,5]=np.random.randint(0,5,size=self.size**2) #Surface type ["Concrete","Asphalt","Brick","Cobblestone","Gravel"]
        attributes[:,6]=np.random.randint(0,5,size=self.size**2) #Weather condition ["Sunny","Moderate Rain","Moderate Snow","Extreme Rain","Extreme Snow"]
        attributes[:,7]=np.random.choice([0,1],size=self.size**2) #Traversability depends on crowd value that should change during the experiment for MPD, constant for search
        attributes[:,8]=np.random.choice([0,1],size=self.size**2) #Tactile pavement

        self.attributes = attributes

        return attributes
    
    def get_cost(self,w):
        """
        Computes cost of each node depending on the chosen weights
        
        :param self: CityGrid object
        :param w: weights dictionary : w = {"w":w0,"l":w1,"s":w2,"sf":w3}
        """
        s = (self.attributes - self.attributes.min(axis=0))/(self.attributes.max(axis=0) - self.attributes.min(axis=0)+1e-9)
        self.score = s
        cost = (w["w"]*s[:,[0]] + w["l"]*s[:,[1]] + w["s"]*(s[:,[2]]+s[:,[3]])*s[:,[6]] + w["sf"]*s[:,[6]])
        self.cost = cost

        return cost
    
city = CityGrid(size=8)
city.generate_attributes(42)
city.get_cost(weights)
city.plot_city_grid()

