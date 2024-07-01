#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:48:56 2024

@author: kk423
"""
import time 
start1=time.time()
import DetectorGenerator 
import TrackGenerator
import HitGenerator
import PatternEncoder
import DetectorTrackGraphMatplotlib
import modules
import sys
import matplotlib as mpl
import HitEncoder
import matplotlib.pyplot as plt
import numpy as np
end1 = time.time()
print(end1-start1)
start2=time.time()
modules=modules.ModuleNumber()

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
end2 = time.time()
print(end2-start2)
start3=time.time()
# Generate tracks, ntracks in each event with nevents
tracks = TrackGenerator.TrackGenerator()
tracks.NumberOfTracksToGenerate = ntracks
tracks.NumberOfEventsToGenerate = nevents
# For reproducibility
tracks.RandomSeed               = 0
# Tracks generated from a uniform distribution in phi from phi0_range[0] - phi std = phi_range[1] to phi0_range[0] + phi std = phi_range[1]
tracks.phi0_Range               = [np.pi/2,np.pi/2]
# Tracks generated from a normal distribution in curvature mean = Curvature_range[0] and Curvature std = Curvature_range[1]
# Curvature is 1/pT with some factors for magnetic field, in this dummy example see curvature as 1/pT
tracks.Curvature_Range          = [0,100] # FindingRanges2.Generate()[1][0] 27
tracks.constantPt               = False
tracks.Generate()
end3=time.time()
print(end3-start3)
start4=time.time()
# Generate hits by checking if a track generated above crosses a module in the detector 
hits = HitGenerator.HitCoordinates()
# Minimum number of modules for each track to be kept, this equates to one hit in each layer for this example
hits.MinimumHits = 4
tracks.Tracks = hits.Generate(detector.Modules,tracks.Tracks)
end4=time.time()
print(end4-start4)
start5=time.time()
#tracks.Tracks = hits.Generate(detector.Modules,tracks.Tracks)
if nevents*ntracks <= 10:
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
PatternEncoder = PatternEncoder.PatternEncoder(detector)
HitEncoder = HitEncoder.HitEncoder(detector)
end5=time.time()
print(end5-start5)
start6=time.time()
# Iterate through the tracks to generate a pattern for each track
# Save the track curvature and phi
# If the track is already in the bank save the minimum and maximum curvature and phi
# Of all the tracks with that bank and tally the frequency
for i,track in enumerate(tracks.Tracks):
    PID = PatternEncoder.PatternID(hits.Hits[i],track,update=True)
    PID = HitEncoder.HitID(hits.Hits[i],track,update=True)
# Save to a file
if SavePatterns:
    PatternEncoder.SavePatterns("patterns")
    HitEncoder.SavePatterns("hits")

if SavePatterns:
    tracks.SavePatterns("alltracks")


# Plot the frequencies of all the patterns found in the tracks
#fighist,axhist = plt.subplots(1,1,figsize=(30,30))
#PGraph = DetectorTrackGraphMatplotlib.PatternGraph(fighist,axhist)
#PGraph.plot(PatternEncoder)
#fighist.savefig("Frequencies.png")
end6=time.time()
print(end6-start6)
#import FindingRanges
#FindingRanges=FindingRanges.FindingRanges()
#FindingRanges.Generate()