#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:19:57 2024

@author: kk423
"""
import pandas as pd
import FindingRanges2

F = FindingRanges2.FindingRanges()
f = F.Generate()
class FindingRanges:

    
    def __init__(self,fig,ax):
        self.myFig=fig
        self.myAx=ax
    
                
    def plot(self):
        file_path = "patternsfull.csv"
        df = pd.read_csv(file_path)
        df = pd.DataFrame(df)
        
    
        a=f[1]
        for i in range(len(a)):
            df.loc[df['ID']==a[i],'cmax']=f[0][i][f"{a[i]}"]['cmax']
            df.loc[df['ID']==a[i],'cmin']=f[0][i][f"{a[i]}"]['cmin']
    
            mean = (df['cmax'][df['ID']==a[i]]+df['cmin'][df['ID']==a[i]])/2
            self.myAx.bar(mean,df['ID'][df['ID']==a[i]],width=df['cmax'][df['ID']==a[i]]-df['cmin'][df['ID']==a[i]],label=f"{a[i]}")
        
        #self.myAx.plot(df['cmax+'][df['ID']==1170] ,df['ID'][df['ID']==a],"r.")  
        #self.myAx.plot(df['cmax-'][df['ID']==1170] ,df['ID'][df['ID']==a],"g.") 
        #self.myAx.plot(df['cmin+'][df['ID']==1170] ,df['ID'][df['ID']==a],"r.") 
        #self.myAx.plot(df['cmin-'][df['ID']==1170] ,df['ID'][df['ID']==a],"g.") 
        self.myAx.set_title("Curvature Ranges")
        self.myAx.set_xlabel("Curvature Ranges")
        self.myAx.set_ylabel("ID")
        self.myAx.legend()
        self.myFig.savefig("Ranges.png")