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
        def set_value(row,str):
            count=0
            for i in a:
                count+=1
                if row['ID']==i:
                    return f[0][count-1][f"{i}"][str]
                else:
                    return None

        # Apply the function to create the new column
        df['cmax'] = df.apply(set_value, axis=1,str="cmax")
        df['cmin'] = df.apply(set_value, axis=1,str="cmin")
        #not these ranges
        self.myAx.plot(df['cmax'],df['frequency'],"g.")  
        self.myAx.plot(df['cmin'],df['frequency'],"g.") 
        self.myFig.savefig("frequency_curvature.png")