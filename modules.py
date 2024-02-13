import numpy as np

class ModuleNumber:

    def __init__(self):
        self.nlayers = 4 #number of layers 
        self.initiallength=0.63
        self.n = 2 #number to multiply the inital number of modules in each layer, and the length of each module is going to be divided with it
        self.distradialposition = 0.5 # distance between the layers 
        self.firstposition = 1.0 #radial position of the first layer
        self.phi0_range = [np.pi/2,np.pi/2] 
        self.curvature_range = [16,1] #using random.normal mean=18 sd=1
        
    def changedetector(self):
        modules=[3,3,3,3]
        length = [0.63,0.63,0.63,0.63]
        #length=[]
        radialposition=[1.0,1.5,2.0,2.5]
        #modules=[]
        
        #radialposition=[]
        #for i in range(self.nlayers):
        #    modules.append(3*self.n) #inital number of modules = 3 
            
        #    length.append(0.63/self.n) #initial length of module = 0.63
        
        #    radialposition.append(self.firstposition + i*self.distradialposition)
        
        """for i in range(self.nlayers):
            length.append(self.initiallength)
            for i in range(self.n):
                length.append(self.initiallength/self.n)
            length.append(self.initiallength)
        """    
        return self.nlayers, modules, length,  radialposition, self.phi0_range, self.curvature_range

