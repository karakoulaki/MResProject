import DetectorGenerator 
import TrackGenerator
import HitGenerator
import PatternEncoder
import DetectorTrackGraphMatplotlib
import sys
import matplotlib as mpl
mpl.use('Agg')

from math import *
import matplotlib.pyplot as plt
# Create Axes for plotting, x and y lims need to be larger than the detector, tracks are produced from 0,0 outwards
figxy,axxy = plt.subplots(1,1,figsize=(30,30))
axxy.set_xlim(-1.5,1.5)
axxy.set_ylim(-0.5,3.5)

# Set number of events and number of tracks per event to create the pattern bank, plotting of the tracks will only happen if 
# nevents * ntracks < 100
nevents = int(sys.argv[1])
ntracks = int(sys.argv[2])
# Do we want to save the pattern bank, only necessary when running many tracks
SavePatterns = bool(sys.argv[3])

# Initialise the detector
detector = DetectorGenerator.DetectorGenerator()
# How many layers in the detector
detector.NumberOfLayers  = 4  
# How many modules in each layer 
detector.NumberOfModules = [3,3,3,3]
# How long is each module in each layer of the detector, no overlaps so module length < 2/3 where 3 is from number of modules 
# in each layer and 2 from the fact the detector spans -1 to 1
detector.ModuleLength    = [0.63,0.63,0.63,0.63]
# How far from the origin is each layer
detector.RadialPosition  = [1.0,1.5,2.0,2.5]
detector.Generate()

# Generate tracks, ntracks in each event with nevents
tracks = TrackGenerator.TrackGenerator()
tracks.NumberOfTracksToGenerate = ntracks
tracks.NumberOfEventsToGenerate = nevents
# For reproducibility
tracks.RandomSeed               = 3
# Tracks generated from a uniform distribution in phi from phi0_range[0] - phi std = phi_range[1] to phi0_range[0] + phi std = phi_range[1]
tracks.phi0_Range               = [pi/2,pi/2]
# Tracks generated from a normal distribution in curvature mean = Curvature_range[0] and Curvature std = Curvature_range[1]
# Curvature is 1/pT with some factors for magnetic field, in this dummy example see curvature as 1/pT
tracks.Curvature_Range          = [20,18] 
tracks.constantPt               = False
tracks.Generate()

# Generate hits by checking if a track generated above crosses a module in the detector 
hits = HitGenerator.HitCoordinates()
# Minimum number of modules for each track to be kept, this equates to one hit in each layer for this example
hits.MinimumHits = 4
tracks.Tracks = hits.Generate(detector.Modules,tracks.Tracks)
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
PatternEncoder = PatternEncoder.PatternEncoder(detector)
# Iterate through the tracks to generate a pattern for each track
# Save the track curvature and phi
# If the track is already in the bank save the minimum and maximum curvature and phi
# Of all the tracks with that bank and tally the frequency
for i,track in enumerate(tracks.Tracks):
    PID = PatternEncoder.PatternID(hits.Hits[i],track,update=True)
# Save to a file
if SavePatterns:
    PatternEncoder.SavePatterns("patterns")

# Plot the frequencies of all the patterns found in the tracks
fighist,axhist = plt.subplots(1,1,figsize=(30,30))
PGraph = DetectorTrackGraphMatplotlib.PatternGraph(fighist,axhist)
PGraph.plot(PatternEncoder)
fighist.savefig("Frequencies.png")
# This will print the pattern frequencies and pattern IDs, to see what an individual pattern
# ID looks like run "python PlotID PID" where PID is the number you want to see 


