import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

class FindingRanges:

    
    def __init__(self):
        self.ranges=[]
      
    
                
    def Generate(self):
        file_path = "patternsfull.csv"
        df = pd.read_csv(file_path)
        df = pd.DataFrame(df)
        
        a=1170
    
        
        
        
        if df.loc[df['ID'] == a, 'cmax-'].iloc[0]==0 and df.loc[df['ID'] == a, 'cmin-'].iloc[0]==0:
            cmax = np.max((df['cmin+'][df['ID']==a],df['cmax+'][df['ID']==a]))
            cmin = np.min((df['cmin+'][df['ID']==a],df['cmax+'][df['ID']==a]))
           
        if df.loc[df['ID'] == a, 'cmax+'].iloc[0]==0 and df.loc[df['ID'] == a, 'cmin+'].iloc[0]==0:
            cmax = np.max((df['cmin-'][df['ID']==a],df['cmax-'][df['ID']==a]))
            cmin = np.min((df['cmin-'][df['ID']==a],df['cmax-'][df['ID']==a]))
        else:
            cmax =  np.max((df['cmax+'][df['ID']==a],-df['cmax-'][df['ID']==a]))
            cmin =  np.min((df['cmin+'][df['ID']==a],-df['cmin-'][df['ID']==a]))
      
        mean = (cmax+cmin)/2 
        sigma = (0.997*(cmax-cmin)/2)/3
           
        self.ranges.append(cmin)
        self.ranges.append(cmax)
        import csv
        f = open('ranges.csv','w')
        writer = csv.writer(f)
        
        writer.writerow(["cmin","cmax","mean","standard deviation"])
        writer.writerow([cmin,cmax,mean,sigma])
        
        f.close()
    
    
        
    
    
