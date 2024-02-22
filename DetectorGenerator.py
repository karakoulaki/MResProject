from math import *
import numpy as np
import pickle

class DetectorGenerator():
    def __init__(self):   
        self.NumberOfLayers  = 0
        
        self.ModuleLength    = [] #changed to a list of the length of each module 
        
        self.RadialPosition  = []
        self.NumberOfModules = []
        self.xrange = [-1,1]
        self.Modules = {}

    def Generate(self):
        print("Generating detector geometry")
        
        ModuleLengthList = []
        if(len(self.ModuleLength) == 1):
            for layer in range(self.NumberOfLayers):
                ModuleLengthList.append(self.ModuleLength[0])
        else:
            ModuleLengthList = self.ModuleLength

        radius    = 0
        elementID = 0
        
        for layer in range(self.NumberOfLayers):
            radius = self.RadialPosition[layer]
             
          
                
            for segment in range(self.NumberOfModules[layer]):
           
                
                   
               
                    if segment==0 :
                        x0 =  self.xrange[0]
                       
                        y0 = radius
                        x1 = x0 + self.ModuleLength[segment]
                        y1 = radius #0 
                        self.Modules[elementID] = [x0,x1,y0,y1]
                        elementID += 1
                    else:
                        
                        x0 =  x1 + 0.055
                        y0 = radius
                        x1 = x0 + self.ModuleLength[segment]
                        y1 = radius 
                        
                        self.Modules[elementID] = [x0,x1,y0,y1]
                        elementID += 1
                
                
            
                
            
                
    

    #def SavePatterns(self,filename):
     #   with open(str(filename)+'.pickle', 'wb') as handle:
     #       pickle.dump(self.modulesx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

    