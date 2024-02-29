# This class encodes the pattern ID given the (SSID,DetectorElement) values, and decodes from pattern ID to (layer,DE,SSID)

import numpy as np
import bitstring
import pickle
import DetectorGenerator

detector = DetectorGenerator.DetectorGenerator()

class HitEncoder:
    def __init__(self,detector):

        self.nelements = (sum(detector.NumberOfModules))
        self.nmodules = detector.NumberOfModules
        self.nlayers = detector.NumberOfLayers
        self.hit = {}

    def HitID(self,hits,track,update=False):
        matrix = []
        PID=[]
        bitlist=[]
        #hittotal = 0
        blankpattern = bitstring.BitArray(uint=0,length=self.nelements)
        for hit in hits:
            blankpattern[hit["ModuleID"]] = 1
            PID.append(blankpattern.int)
        #for i in range(len(PID)):
        #    bitlist.append( [int(d) for d in str(blankpattern[i].bin)])
        #    for layer in range(self.nlayers):
        #        layerlist = []
        #        for module in range(self.nmodules[layer]):
        #            layerlist.append(bitlist[i][layer*self.nmodules[layer]+module])
        #        matrix.append(layerlist)
        
        for i in range(len(PID)):
            if update:
                try:
                
                    
                    if track["Charge"] == 1:  
                        if (track["Phi"] < self.hit[PID[i]]["phimin+"]) or (self.hit[PID[i]]["phimin+"] == 0):
                            self.hit[PID[i]]["phimin+"] = track["Phi"]
                            self.hit[PID[i]]["cmin+"] = track["Curvature"]
                        if (track["Phi"] > self.hit[PID[i]]["phimax+"]) or (self.hit[PID[i]]["phimax+"] == 0):
                            self.hit[PID[i]]["phimax+"] = track["Phi"]
                            self.hit[PID[i]]["cmax+"] = track["Curvature"]

                        self.hit[PID[i]]["ctot+"] = self.hit[PID[i]]["ctot+"] + abs(track["Curvature"]) 
                        self.hit[PID[i]]["phitot+"] = self.hit[PID[i]]["phitot+"] + track["Phi"]

                    else:                   
                        if (track["Phi"] < self.hit[PID[i]]["phimin-"]) or (self.hit[PID[i]]["phimin-"] == 0):
                            self.hit[PID[i]]["phimin-"] = track["Phi"]
                            self.hit[PID[i]]["cmin-"] = track["Curvature"]
                        if (track["Phi"] > self.hit[PID[i]]["phimax-"]) or (self.hit[PID[i]]["phimax-"] == 0):
                            self.hit[PID[i]]["phimax-"] = track["Phi"]
                            self.hit[PID[i]]["cmax-"] = track["Curvature"]

                        self.hit[PID[i]]["ctot-"] = self.hit[PID[i]]["ctot-"] + abs(track["Curvature"]) 
                        self.hit[PID[i]]["phitot-"] = self.hit[PID[i]]["phitot-"] + track["Phi"]

                    self.hit[PID[i]]["frequency"] += 1

                except KeyError:
                    if track["Charge"] == 1:
                        self.hit[PID[i]] = {
                                    "cmin+": track["Curvature"],
                                    "cmax+": track["Curvature"],
                                    "cmin-": 0,
                                    "cmax-": 0,
                                    "ctot+": track["Curvature"],
                                    "ctot-": 0,
                                    "phimin+": track["Phi"],
                                    "phimax+": track["Phi"],
                                    "phitot+": track["Phi"],
                                    "phimin-": 0,
                                    "phimax-": 0,
                                    "phitot-": 0,
                                    "frequency" : 1,
                                    "matrix" : matrix
                                    }
                
                    else:
                        self.hit[PID[i]] = {
                                    "cmin+": 0,
                                    "cmax+": 0,
                                    "cmin-": track["Curvature"],
                                    "cmax-": track["Curvature"],
                                    "ctot+": 0,
                                    "ctot-": track["Curvature"],
                                    "phimin+": 0,
                                    "phimax+": 0,
                                    "phitot+": 0,
                                    "phimin-": track["Phi"],
                                    "phimax-": track["Phi"],
                                    "phitot-": track["Phi"],
                                    "frequency" : 1,
                                    "matrix" : matrix
                                    }

        return PID    

    def DecodePatternID(self,PID):
        decoded=[]
        for i in range(len(PID)):
            decoded.append( bitstring.BitArray(int=PID[i], length=self.nelements))
        return decoded
    
    def SavePatterns(self,filename):
        with open(str(filename)+'.pickle', 'wb') as handle:
            pickle.dump(self.hit, handle, protocol=pickle.HIGHEST_PROTOCOL)

        import csv
        # open the file in the write mode
        f1 = open('hitsfull.csv', 'w')
        f2 = open('hits.csv', 'w')
        # create the csv writer
        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)

        writer1.writerow(["ID","Hit Map","cmin-","cmax-","ctot-","phimin-","phimax-","phitot-","cmin+","cmax+","ctot+","phimin+","phimax+","phitot+","frequency"])
        
        for pattern in self.hit:
            patternstring = (bitstring.Bits(int=pattern,length=self.nelements).bin)
            # write a row to the csv file
                
            writer1.writerow([pattern,str(patternstring),
                              self.hit[pattern]["cmin-"],
                              self.hit[pattern]["cmax-"],
                              self.hit[pattern]["ctot-"],
                              self.hit[pattern]["phimin-"],
                              self.hit[pattern]["phimax-"],
                              self.hit[pattern]["phitot-"],
                              self.hit[pattern]["cmin+"],
                              self.hit[pattern]["cmax+"],
                              self.hit[pattern]["ctot+"],
                              self.hit[pattern]["phimin+"],
                              self.hit[pattern]["phimax+"],
                              self.hit[pattern]["phitot+"],
                              self.hit[pattern]["frequency"]])
            writer2.writerow([str(patternstring)])

        # close the file
        f1.close()
        f2.close()


    def LoadPatterns(self,filename):
        with open(str(filename)+'.pickle', 'rb') as handle:
            self.hit = pickle.load(handle)
