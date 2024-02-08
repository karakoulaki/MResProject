class ModuleNumber:

    def __init__(self):
        self.nlayers = 4 #number of layers 
        self.n = 2 #number to multiply the inital number of modules in each layer, and the length of each module is going to be divided with it
        self.distradialposition = 0.5 # distance between the layers 
        self.firstposition = 1.0 #radial position of the first layer
    
    def changedetector(self):
        modules=[]
        length=[]
        radialposition=[]
        for i in range(self.nlayers):
            modules.append(3*self.n) #inital number of modules = 3 
            
            length.append(0.63/self.n) #initial length of module = 0.63

            radialposition.append(self.firstposition + i*self.distradialposition)
            
        return self.nlayers, modules, length,  radialposition
        