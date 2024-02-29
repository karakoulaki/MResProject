
import numpy as np
import pandas as pd
#import FindingRanges2



#F = FindingRanges2.FindingRanges()
#f = F.Generate()

class ModuleNumber:

    def __init__(self):
        self.nlayers = 4 #number of layers 
        self.initiallength=0.63
        self.phi0_range = [np.pi/2,np.pi/2] #np.pi/2 to np.pi/2

        #f[1][0]#[2,38]
        #f[1][0]#f[1][0]] #using random.normal mean=18 sd=1
        self.sep=0#0 if uniform, 1 if normal
    def changedetector(self):
        modules=[]
        length=[]
        radialposition=[1.0,1.5,2.0,2.5]
        curvature_range=[]
        
        file_path = "ranges.csv"
        df = pd.read_csv(file_path)
        df = pd.DataFrame(df)
        
        if self.sep==0:
            curvature_range.append(df.at[0,'cmin'])
            curvature_range.append(df.at[0,'cmax'])
        if self.sep==1:
            curvature_range.append(df.at[0,'mean'])
            curvature_range.append(df.at[0,'standard deviation'])
       
        for i in range(self.nlayers):
            modules.append(4)

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
            
        
                
       
        return self.nlayers, modules, length,  radialposition, self.phi0_range, curvature_range, self.sep

