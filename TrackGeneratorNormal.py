#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:56:22 2024

@author: kk423
"""

from math import *
import numpy as np
import pickle
import modules


modules=modules.ModuleNumber()
class TrackGenerator:
    def __init__(self):
        self.RandomSeed               = 3
        self.NumberOfTracksToGenerate = 20
        self.NumberOfEventsToGenerate = 100
        
        self.phi0_Range               = [] #0,2pi #from modules 
        self.Curvature_Range          = [] # curvature radius range from 2.5cm to 1000 m #from modules
        self.constantPt               = False

        self.Tracks = []
        self.allTracks = []
    #making the Ntuples for track infomation 
    def Generate(self):        
        #Random Generator for track making 
        for EventNumber in range(self.NumberOfEventsToGenerate):
            nTracks = self.NumberOfTracksToGenerate

            for i in range(nTracks):
                track = {}
                if self.constantPt:
                    Curvature = (self.Curvature_Range[0]+self.Curvature_Range[1])/2
                    #(self.Curvature_Range[0]+self.Curvature_Range[1])/2
                    Sign = 1
                    alltracks = {"EventNumber" : EventNumber,
                         "TrackNumber" : i,
                         "Curvature" : Curvature,
                         "Charge" : Sign}
                    self.allTracks.append(alltracks)
                else:
                    Curvature =  np.random.normal(self.Curvature_Range[0],self.Curvature_Range[1]) #0,38
                    Sign = np.random.choice([-1,1])
                    #np.random.choice([-1,1])
                    Curvature = Curvature*Sign
                    Phi      = self.phi0_Range[0]
                    alltracks = {"EventNumber" : EventNumber,
                         "TrackNumber" : i,
                         "Curvature" : Curvature,
                         "Charge" : Sign,
                         "Phi": Phi}
                    self.allTracks.append(alltracks)
                    
                Phi      = self.phi0_Range[0]

                track = {"EventNumber" : EventNumber,
                         "TrackNumber" : i,
                         "Curvature" : Curvature,
                         "Charge" : Sign,
                        "Phi" : Phi}
                self.Tracks.append(track)
    
    def SavePatterns(self,filename):
        with open(str(filename)+'.pickle', 'wb') as handle:
            pickle.dump(self.allTracks, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #print('Event = ', EventNumber, ' done')
