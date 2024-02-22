import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import modules
import FindingRanges2
#run this if the distribution is normal
modules=modules.ModuleNumber()
F=FindingRanges2.FindingRanges()
class Separation:

    def __init__(self,fig,ax):
        self.myFig=fig
        self.myAx=ax
    
                
    def plot(self):
        file_path = "alltracks.pickle"
        df = pd.read_pickle(file_path)
        df = pd.DataFrame(df)
        
        df_pos=df['Curvature'][df['Charge']==1] 
        df_neg=df['Curvature'][df['Charge']==-1]
        n,xe=np.histogram(df_pos,bins=50)
        cx = 0.5 * (xe[1:] + xe[:-1])
        n2,xe2=np.histogram(df_neg,bins=50)
        cx2 = 0.5 * (xe2[1:] + xe2[:-1])

        self.myAx.plot(cx,n,'g.-',label='positive particles')  
        self.myAx.plot(cx2,n2,'r.-',label='negative particles')
        #self.myAx.xlabel('Curvature')
        #self.myAx.ylabel('Bins')
        #self.myAx.legend()
        #self.myAx.title('Truth Level phi0=pi/2')
        self.myFig.savefig("trial.png")
        
      