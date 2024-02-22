
import numpy as np
import pandas as pd
class ModuleNumber:

    def __init__(self):
        self.nlayers = 4 #number of layers 
        self.initiallength=0.63
        self.phi0_range = [0,np.pi] #np.pi/2 to np.pi/2
        self.curvature_range = [20,18] #using random.normal mean=18 sd=1
        self.sep=0 #0 if uniform, 1 if normal
    def changedetector(self):
        modules=[4,4,4,4]
        length=[]
        radialposition=[1.0,1.5,2.0,2.5]

        if modules[0]>3:
            for j in range(self.nlayers):
                for i in range(modules[j]):
                    if i==0 or i==modules[j]-1:
                        length.append(self.initiallength)
                    else:
                        length.append(self.initiallength/(modules[0]/2))
        else:
            for i in range(sum(modules)):
                length.append(self.initiallength)
            
        
                
       
        return self.nlayers, modules, length,  radialposition, self.phi0_range, self.curvature_range, self.sep

