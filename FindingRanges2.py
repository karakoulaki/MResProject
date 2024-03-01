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
        a=[17476,8738]
        #17476,8738
        for i in a:
            
            cmax =  np.max((df['cmin+'][df['ID']==i],-df['cmax-'][df['ID']==i]))
            cmin =  np.min((df['cmax+'][df['ID']==i],-df['cmin-'][df['ID']==i]))
            
            if cmax==0:
                cmax=cmin
                r={f"{i}":{"cmax": cmax,
                        "cmin":cmin}}
            if cmin==0:
                cmin=cmax
                r={f"{i}":{"cmax": cmax,
                        "cmin":cmin}}
            else:
                r={f"{i}":{"cmax": cmax,
                        "cmin":cmin}}
            
            
            self.ranges.append(r)
           
            
        #self.ranges.append(ranges)
        return self.ranges, a
    
        