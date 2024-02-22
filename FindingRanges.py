import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

class FindingRanges:

    def __init__(self,fig,ax):
        self.myFig=fig
        self.myAx=ax
        self.ranges=[]
    
                
    def plot(self):
        file_path = "patternsfull.csv"
        df = pd.read_csv(file_path)
        df = pd.DataFrame(df)
        
        
        
        self.myAx.plot(df['cmax+'][df['ID']==1170] ,df['ID'][df['ID']==1170],"r.")  
        self.myAx.plot(df['cmax-'][df['ID']==1170] ,df['ID'][df['ID']==1170],"r.") 
        self.myAx.plot(df['cmin+'][df['ID']==1170] ,df['ID'][df['ID']==1170],"g.") 
        self.myAx.plot(df['cmin-'][df['ID']==1170] ,df['ID'][df['ID']==1170],"g.") 
        self.myFig.savefig("Ranges.png")
    