import DetectorGenerator 
import TrackGenerator
import HitGenerator
import PatternEncoder
import DetectorTrackGraphMatplotlib
import numpy as np
import bitstring
import sys
import matplotlib as mpl
import modules
mpl.use('Agg')

from math import *
import matplotlib.pyplot as plt
# Create Axes for plotting, x and y lims need to be larger than the detector, tracks are produced from 0,0 outwards
figxy,axxy = plt.subplots(1,1,figsize=(30,30))
axxy.set_xlim(-1.5,1.5)
axxy.set_ylim(-0.5,3.5)

plot = True
modules=modules.ModuleNumber()
def QtrackWrapper(bits):
    qbits = bits
    translatedBits = bitstring.Bits(bin='0b'+str(qbits))
    return(translatedBits)

# Initialise the detector
detector = DetectorGenerator.DetectorGenerator()
# How many layers in the detector
detector.NumberOfLayers  = modules.changedetector()[0]  
# How many modules in each layer 
detector.NumberOfModules = modules.changedetector()[1]
# How long is each module in each layer of the detector, no overlaps so module length < 2/3 where 3 is from number of modules 
# in each layer and 2 from the fact the detector spans -1 to 1
detector.ModuleLength    = modules.changedetector()[2]
# How far from the origin is each layer
detector.RadialPosition  = modules.changedetector()[3]
detector.Generate()

# Generate tracks, ntracks in each event with nevents
tracks = TrackGenerator.TrackGenerator()
tracks.NumberOfTracksToGenerate = 1
tracks.NumberOfEventsToGenerate = 10
# For reproducibility
tracks.RandomSeed               = 3
# Tracks generated from a uniform distribution in phi from phi0_range[0] - phi std = phi_range[1] to phi0_range[0] + phi std = phi_range[1]
tracks.phi0_Range               = [pi/2,pi/2]#[pi/2,pi/2]
# Tracks generated from a normal distribution in curvature mean = Curvature_range[0] and Curvature std = Curvature_range[1]
# Curvature is 1/pT with some factors for magnetic field, in this dummy example see curvature as 1/pT
tracks.Curvature_Range          = [20,18]#[20,18] 
tracks.constantPt               = False
tracks.Generate()

# Generate hits by checking if a track generated above crosses a module in the detector 
hits = HitGenerator.HitCoordinates()
# Minimum number of modules for each track to be kept, this equates to one hit in each layer for this example
hits.MinimumHits = modules.changedetector()[0]
tracks.Tracks = hits.Generate(detector.Modules,tracks.Tracks)

PatternEncoder = PatternEncoder.PatternEncoder(detector)
PatternEncoder.LoadPatterns("patterns")

TrueTrackIDs = []
QTrackIDs = []

if len(hits.Hits) > 0:
    for i,track in enumerate(tracks.Tracks):
        TrackID = PatternEncoder.PatternID(hits.Hits[i],track,update=False)
        BitID = PatternEncoder.DecodePatternID(TrackID)
        print("Track : ", i)
        print("Input hitpattern: ",BitID.bin)
        print("True Track ID: ",BitID.int)
        TrueTrackIDs.append(BitID)
        Qbits = QtrackWrapper(BitID.bin)
        QTrackIDs.append(Qbits)
        print("Q Predicted Track ID: ",Qbits.int)
        print("========================")

        if plot:
            DGraph = DetectorTrackGraphMatplotlib.DetectorGraph(fig=figxy,ax=axxy)
            HGraph = DetectorTrackGraphMatplotlib.HitGraph(figxy,axxy)
            TGraph = DetectorTrackGraphMatplotlib.TrackGraph(figxy,axxy)
            PGraph = DetectorTrackGraphMatplotlib.PatternGraph(fig=figxy,ax=axxy)

            TGraph.plot(tracks.Tracks,hits.Hits,HGraph)
            DGraph.plot(detector.Modules)
            if len(tracks.Tracks) == 1:
                PGraph.plotsinglepattern(TrackID,PatternEncoder,TGraph)
        figxy.savefig("Track.png")




