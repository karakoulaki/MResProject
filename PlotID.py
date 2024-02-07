import DetectorGenerator 
import PatternEncoder
import DetectorTrackGraphMatplotlib
import matplotlib as mpl
import modules
mpl.use('Agg')

from math import *
import matplotlib.pyplot as plt
import pickle
import sys
modules=modules.ModuleNumber()
# Generate detector again TODO save the detector from the track generator file
# Initialise the detector
detector = DetectorGenerator.DetectorGenerator()
# How many layers in the detector
detector.NumberOfLayers  = modules.changedetector()[2] 
# How many modules in each layer 
detector.NumberOfModules = modules.changedetector()[0]
# How long is each module in each layer of the detector, no overlaps so module length < 2/3 where 3 is from number of modules 
# in each layer and 2 from the fact the detector spans -1 to 1
detector.ModuleLength    = modules.changedetector()[1]
# How far from the origin is each layer
detector.RadialPosition  = [1.0,1.5,2.0,2.5]
detector.Generate()

# Load pattern bank
PatternEncoder = PatternEncoder.PatternEncoder(detector)
PatternEncoder.LoadPatterns("patterns")

# PID of interest
PID = int(sys.argv[1])
figxy,axxy = plt.subplots(1,1,figsize=(30,30))
plt.gca().set_aspect('equal')

axxy.set_xlim(-1.5,1.5)
axxy.set_ylim(-0.5,3.5)
#Plot the detector so we can see the patterns on it
DGraph = DetectorTrackGraphMatplotlib.DetectorGraph(fig=figxy,ax=axxy)
DGraph.plot(detector.Modules)
TGraph = DetectorTrackGraphMatplotlib.TrackGraph(figxy,axxy)
PGraph = DetectorTrackGraphMatplotlib.PatternGraph(fig=figxy,ax=axxy)
# Plot the pattern with the minimum and maximum possible tracks for that PID
PGraph.plotsinglepattern(PID,PatternEncoder,TGraph)

figxy.savefig(str(PID) + ".png")