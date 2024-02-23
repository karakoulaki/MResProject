import matplotlib.pyplot as plt
import pandas as pd
import FindingRanges2

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
        
        
        a=[17476,8738]
        for i in range(len(a)):
            df.loc[df['ID']==a[i],'cmax']=f[0][i][f"{a[i]}"]['cmax']
            df.loc[df['ID']==a[i],'cmin']=f[0][i][f"{a[i]}"]['cmin']
        
        
        
        self.myAx.plot(df['cmax'],df['frequency'],"g.")  
        self.myAx.plot(df['cmin'],df['frequency'],"g.") 
        self.myFig.savefig("frequency_curvature.png")