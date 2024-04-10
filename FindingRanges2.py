#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:17:31 2024

@author: kk423
"""

import pandas as pd
import numpy as np
import pickle

class FindingRanges:
    def __init__(self):
        
        self.ranges=[]  
                
    def Generate(self):
        file_path = "patternsfull.csv" #hitsfull.csv to work with curvature ranges of modules instead of tracks 
        df = pd.read_csv(file_path)
        df = pd.DataFrame(df)
        #ranges=[]
        a=[135300]
        #17476,8738,8740,8772,9284
        #17476,8738,8740,17474,9284,8772,16930,17442,8466,18564
        for i in a:
            
            cmax =  np.max((df['cmax+'][df['ID']==i],-df['cmin-'][df['ID']==i]))
            cmin =  np.min((df['cmin+'][df['ID']==i],-df['cmax-'][df['ID']==i]))
            
            if df.loc[df['ID'] == i, 'cmax-'].iloc[0]==0 and df.loc[df['ID'] == i, 'cmin-'].iloc[0]==0:
                cmax = np.max((df['cmin+'][df['ID']==i],df['cmax+'][df['ID']==i]))
                cmin = np.min((df['cmin+'][df['ID']==i],df['cmax+'][df['ID']==i]))
                r={f"{i}":{"cmax": cmax,
                        "cmin":cmin}}
            if df.loc[df['ID'] == i, 'cmax+'].iloc[0]==0 and df.loc[df['ID'] == i, 'cmin+'].iloc[0]==0:
                cmax = np.max((df['cmin-'][df['ID']==i],df['cmax-'][df['ID']==i]))
                cmin = np.min((df['cmin-'][df['ID']==i],df['cmax-'][df['ID']==i]))
                r={f"{i}":{"cmax": cmax,
                        "cmin":cmin}}
            else:
                r={f"{i}":{"cmax": cmax,
                        "cmin":cmin}}
            
            
            self.ranges.append(r)
           
            
        #self.ranges.append(ranges)
        return self.ranges, a
    
        