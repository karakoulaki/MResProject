# This class encodes the pattern ID given the (SSID,DetectorElement) values, and decodes from pattern ID to (layer,DE,SSID)

import numpy as np
import bitstring
import pickle
import DetectorGenerator

detector = DetectorGenerator.DetectorGenerator()

class PatternEncoder:
    def __init__(self,detector):

        self.nelements = (sum(detector.NumberOfModules))
        self.nmodules = detector.NumberOfModules
        self.nlayers = detector.NumberOfLayers
        self.patterns = {}

    def PatternID(self,hits,track,update=False):
        matrix = []
        hittotal = 0
        blankpattern = bitstring.BitArray(uint=0,length=self.nelements)
        for hit in hits:
            blankpattern[hit["ModuleID"]] = 1
            hittotal += 1
        PID = blankpattern.int
        bitlist = [int(d) for d in str(blankpattern.bin)]
        for layer in range(self.nlayers):
            layerlist = []
            for module in range(self.nmodules[layer]):
                layerlist.append(bitlist[layer*self.nmodules[layer]+module])
                
                
            matrix.append(layerlist)
      

        if update:
            try: 
               from csv import writer
               import os
               titles = ["ID","Matrix","Curvature", "Phi"]

               filename = "validation_data.csv"
               
               if not os.path.exists(filename) or os.stat(filename).st_size == 0:
                   with open(filename, "a") as f3:
                       writer3 = writer(f3)
        
                       writer3.writerow(titles)

               with open(filename, "a") as f3:
                   writer3 = writer(f3)
                   matrix2 = np.array(matrix)


                   flattened_matrix = matrix2.flatten()


                   binary_string = ''.join(bitstring.Bits(uint=x, length=1).bin for x in flattened_matrix)
                   ID = int(binary_string, 2)
   
                   writer3.writerow([ID, matrix,track["Curvature"], track["Phi"]])
               if track["Charge"] == 1:                   
                    if (track["Phi"] <= self.patterns[PID]["phimin+"]) or (self.patterns[PID]["phimin+"] == 0): #added =
                        self.patterns[PID]["phimin+"] = track["Phi"]
                        self.patterns[PID]["cmin+"] = track["Curvature"]
                    if (track["Phi"] > self.patterns[PID]["phimax+"]) or (self.patterns[PID]["phimax+"] == 0):
                        self.patterns[PID]["phimax+"] = track["Phi"]
                        self.patterns[PID]["cmax+"] = track["Curvature"]

                    self.patterns[PID]["ctot+"] = self.patterns[PID]["ctot+"] + abs(track["Curvature"]) 
                    self.patterns[PID]["phitot+"] = self.patterns[PID]["phitot+"] + track["Phi"]

               else:                   
                    if (track["Phi"] <= self.patterns[PID]["phimin-"]) or (self.patterns[PID]["phimin-"] == 0): #added  =
                        self.patterns[PID]["phimin-"] = track["Phi"]
                        self.patterns[PID]["cmin-"] = track["Curvature"]
                    if (track["Phi"] > self.patterns[PID]["phimax-"]) or (self.patterns[PID]["phimax-"] == 0):
                        self.patterns[PID]["phimax-"] = track["Phi"]
                        self.patterns[PID]["cmax-"] = track["Curvature"]

                    self.patterns[PID]["ctot-"] = self.patterns[PID]["ctot-"] + abs(track["Curvature"]) 
                    self.patterns[PID]["phitot-"] = self.patterns[PID]["phitot-"] + track["Phi"]

               self.patterns[PID]["frequency"] += 1

            except KeyError:
               
                
                if track["Charge"] == 1:
                    self.patterns[PID] = {
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
                    self.patterns[PID] = {
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
        decoded = bitstring.BitArray(int=PID, length=self.nelements)
        return decoded
    
    def SavePatterns(self,filename):
        with open(str(filename)+'.pickle', 'wb') as handle:
            pickle.dump(self.patterns, handle, protocol=pickle.HIGHEST_PROTOCOL)

        import csv
        # open the file in the write mode
        f1 = open('patternsfull.csv', 'w')
        f2 = open('patterns.csv', 'w')
        # create the csv writer
        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)

        writer1.writerow(["ID","Hit Map","cmin-","cmax-","ctot-","phimin-","phimax-","phitot-","cmin+","cmax+","ctot+","phimin+","phimax+","phitot+","frequency"])
        
        for pattern in self.patterns:
            patternstring = (bitstring.Bits(int=pattern,length=self.nelements).bin)
            # write a row to the csv file
                
            writer1.writerow([pattern,str(patternstring),
                              self.patterns[pattern]["cmin-"],
                              self.patterns[pattern]["cmax-"],
                              self.patterns[pattern]["ctot-"],
                              self.patterns[pattern]["phimin-"],
                              self.patterns[pattern]["phimax-"],
                              self.patterns[pattern]["phitot-"],
                              self.patterns[pattern]["cmin+"],
                              self.patterns[pattern]["cmax+"],
                              self.patterns[pattern]["ctot+"],
                              self.patterns[pattern]["phimin+"],
                              self.patterns[pattern]["phimax+"],
                              self.patterns[pattern]["phitot+"],
                              self.patterns[pattern]["frequency"]])
            writer2.writerow([str(patternstring)])

        # close the file
        f1.close()
        f2.close()


    def LoadPatterns(self,filename):
        with open(str(filename)+'.pickle', 'rb') as handle:
            self.patterns = pickle.load(handle)
