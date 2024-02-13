from math import *
import numpy as np
import pickle

class DetectorGenerator():
    def __init__(self):   
        self.NumberOfLayers  = 0
        self.ModuleLength    = []
        self.RadialPosition  = []
        self.NumberOfModules = []
        self.xrange = [-1,1]
        self.Modules = {}
        self.modulesx = [] 

    def Generate(self):
        #print("Generating detector geometry")
        
        ModuleLengthList = []
        if(len(self.ModuleLength) == 1):
            for layer in range(self.NumberOfLayers):
                ModuleLengthList.append(self.ModuleLength[0])
        else:
            ModuleLengthList = self.ModuleLength

        radius    = 0
        elementID = 0
        
        modulesx=[]
        for layer in range(self.NumberOfLayers):
           
            radius = self.RadialPosition[layer]
            xpositions = (self.xrange[1] - self.xrange[0]) /(self.NumberOfModules[layer])
            for segment in range(self.NumberOfModules[layer]):

                x0        = self.xrange[0] + xpositions*(segment) + xpositions/2 - self.ModuleLength[layer]/2
                y0        = radius
                x1        = x0 + self.ModuleLength[segment]
                y1        = radius 

                self.Modules[elementID] = [x0,x1,y0,y1]
                self.modulesx.append([x0,x1])
                modulesx.append([x0,x1])
                elementID += 1
        #return self.NumberOfLayers
        
    def Generate2(self):
        return self.modulesx

    def SavePatterns(self,filename):
        with open(str(filename)+'.pickle', 'wb') as handle:
            pickle.dump(self.modulesx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

    