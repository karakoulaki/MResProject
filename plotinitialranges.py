#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:19:57 2024

@author: kk423
"""
import pandas as pd


class FindingRanges:

    
    def __init__(self,fig,ax):
        self.myFig=fig
        self.myAx=ax
    
                
    def plot(self):
        file_path = "patternsfull.csv"
        df = pd.read_csv(file_path)
        df = pd.DataFrame(df)
        
        a=1170
    
     
            
        
        
        self.myAx.plot(df['cmax+'][df['ID']==1170] ,df['ID'][df['ID']==a],"r.")  
        self.myAx.plot(df['cmax-'][df['ID']==1170] ,df['ID'][df['ID']==a],"g.") 
        self.myAx.plot(df['cmin+'][df['ID']==1170] ,df['ID'][df['ID']==a],"r.") 
        self.myAx.plot(df['cmin-'][df['ID']==1170] ,df['ID'][df['ID']==a],"g.") 
        self.myAx.set_title("Curvature Ranges")
        self.myAx.set_xlabel("Curvature Ranges")
        self.myAx.set_ylabel("ID")
        self.myFig.savefig("Ranges.png")