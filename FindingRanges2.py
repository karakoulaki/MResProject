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
        self.mean_sd=[]
                
    def Generate(self):
        file_path = "patternsfull.csv"
        df = pd.read_csv(file_path)
        df = pd.DataFrame(df)
        #ranges=[]
        a=[17476,8738]
        for i in a:
            cmax = np.max((df['cmin+'][df['ID']==i],-df['cmax-'][df['ID']==i]))
            cmin =  np.min((df['cmax+'][df['ID']==i],-df['cmin-'][df['ID']==i]))
            r={f"{i}":{"cmax": cmax,
                    "cmin":cmin}}
            
            self.ranges.append(r)
            mean = (cmax+cmin)/2 
            sigma = (0.997*(cmax-cmin)/2)/3
            #m={f"{i}":{"mean": mean,
             #       "sigma":sigma}}
            self.mean_sd.append([mean,sigma])
            
        #self.ranges.append(ranges)
        return self.ranges, self.mean_sd
    
        
        
        
    def SavePatterns(self,filename):
        with open(str(filename)+'.pickle', 'wb') as handle:
           pickle.dump(self.ranges, handle, protocol=pickle.HIGHEST_PROTOCOL)