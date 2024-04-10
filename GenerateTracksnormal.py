#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:44:13 2024

@author: kk423
"""

import DetectorGenerator 

import HitGenerator
import PatternEncodernormal
import DetectorTrackGraphMatplotlib
import modules
import sys
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import HitEncoder

modules=modules.ModuleNumber()

file_path = 'alltracks_normal.pickle' #or alltracks_normal
df_tracks = pd.read_pickle(file_path)
df = pd.DataFrame(df_tracks)
df_pos=df['Curvature'][df['Charge']==1] #12 modules
df_neg=df['Curvature'][df['Charge']==-1]
mean = np.mean(df_pos)
mean2 = np.mean(df_neg)
sd = np.std(df_pos)
sd2 = np.std(df_neg)
plt.figure(figsize=[10,10])
plt.hist(df_neg,bins=10,density=True,color='r',label='negative particles')
plt.hist(df_pos,bins=10,density=True,color='g',label='positive particles')
plt.xlabel('Curvature')
plt.ylabel('Bins')
plt.annotate(f"mean={np.round(mean,2)},sigma={np.round(sd,2)}",(-40,0.04))
plt.legend()
plt.title('Truth Level phi0=pi/2')
plt.savefig("normaldistributionplot")
plt.close('all')
mpl.use('Agg')


# Create Axes for plotting, x and y lims need to be larger than the detector, tracks are produced from 0,0 outwards
figxy,axxy = plt.subplots(1,1,figsize=(30,30))
axxy.set_xlim(-1.5,1.5)
axxy.set_ylim(-0.5,3.5)

# Set number of events and number of tracks per event to create the pattern bank, plotting of the tracks will only happen if 
# nevents * ntracks < 100
nevents = int(sys.argv[1])
ntracks = int(sys.argv[2])
# Do we want to save the pattern bank, only necessary when running many tracks
SavePatterns = bool(sys.argv[3])


#FindingRanges2 = FindingRanges2.FindingRanges() 

# Initialise the detector
detector = DetectorGenerator.DetectorGenerator()
# How many layers in the detector
detector.NumberOfLayers  = modules.changedetector()[0]
# How many modules in each layer 
detector.NumberOfModules = modules.changedetector()[1]
#[3,3,3,3]
# How long is each module in each layer of the detector, no overlaps so module length < 2/3 where 3 is from number of modules 
# in each layer and 2 from the fact the detector spans -1 to 1
detector.ModuleLength    = modules.changedetector()[2]
#[0.63,0.63,0.63,0.63]
# How far from the origin is each layer
detector.RadialPosition  = modules.changedetector()[3]


detector.xrange = modules.changedetector()[6]
detector.Generate()


# Generate hits by checking if a track generated above crosses a module in the detector 
hits = HitGenerator.HitCoordinates()
# Minimum number of modules for each track to be kept, this equates to one hit in each layer for this example
hits.MinimumHits = modules.changedetector()[0]
tracks = hits.Generate(detector.Modules,df_tracks)
#tracks.Tracks = hits.Generate(detector.Modules,tracks.Tracks)
if nevents*ntracks <= 100:
    # Plot detector
    DGraph = DetectorTrackGraphMatplotlib.DetectorGraph(fig=figxy,ax=axxy)
    DGraph.plot(detector.Modules)
    figxy.savefig("Detector.png")
    
    # Plot just the hits
    HGraph = DetectorTrackGraphMatplotlib.HitGraph(figxy,axxy)
    HGraph.plot(hits.Hits)
    figxy.savefig("Hits.png")


    # Plot the tracks and the hits
    TGraph = DetectorTrackGraphMatplotlib.TrackGraph(figxy,axxy)
    TGraph.plot(tracks.Tracks,hits.Hits,HGraph)
    figxy.savefig("Tracks.png")

# Initialise pattern encoder
PatternEncoder = PatternEncodernormal.PatternEncoder(detector)
HitEncoder = HitEncoder.HitEncoder(detector)
# Iterate through the tracks to generate a pattern for each track
# Save the track curvature and phi
# If the track is already in the bank save the minimum and maximum curvature and phi
# Of all the tracks with that bank and tally the frequency
for i,track in enumerate(tracks):
    PID = PatternEncoder.PatternID(hits.Hits[i],track,update=True)
    PID = HitEncoder.HitID(hits.Hits[i],track,update=True)
# Save to a file
if SavePatterns:
    PatternEncoder.SavePatterns("patternsnormal")
#
#    HitEncoder.SavePatterns("hits")



# Plot the frequencies of all the patterns found in the tracks
fighist,axhist = plt.subplots(1,1,figsize=(30,30))
PGraph = DetectorTrackGraphMatplotlib.PatternGraph(fighist,axhist)
PGraph.plot(PatternEncoder)
fighist.savefig("Frequencies.png")
# This will print the pattern frequencies and pattern IDs, to see what an individual pattern
# ID looks like run "python PlotID PID" where PID is the number you want to see 
import Separation
fighist,axhist = plt.subplots(1,1,figsize=(15,15))
SGraph = Separation.Separation(fighist,axhist)
SGraph.plot()


