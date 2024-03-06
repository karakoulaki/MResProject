import matplotlib.pyplot as plt
import pandas as pd#
import FindingRanges2
import numpy as np

F = FindingRanges2.FindingRanges()
f = F.Generate()

class Separation:

    def __init__(self,fig,ax):
        self.myFig=fig
        self.myAx=ax
    
                
    def plot(self):
        file_path = "patternsfull.csv"
        df = pd.read_csv(file_path)
        df = pd.DataFrame(df)
        #df['cmax'] = [f[0][0]]
        
        
        a=f[1]
        for i in range(len(a)):
            df.loc[df['ID']==a[i],'cmax']=f[0][i][f"{a[i]}"]['cmax']
            df.loc[df['ID']==a[i],'cmin']=f[0][i][f"{a[i]}"]['cmin']
        
            mean = (df['cmax'][df['ID']==a[i]]+df['cmin'][df['ID']==a[i]])/2
            self.myAx.bar(mean,df['frequency'][df['ID']==a[i]],width=df['cmax'][df['ID']==a[i]]-df['cmin'][df['ID']==a[i]],label=f"{a[i]}")
        self.myAx.set_title("Curvature Range")
        self.myAx.set_xlabel("Curvature Ranges")
        self.myAx.set_ylabel("Frequency")
        self.myAx.legend()
        self.myFig.savefig("frequency_curvature.png")
        
       