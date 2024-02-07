class ModuleNumber:

    def __init__(self):
        self.nlayers = 4
        self.div = 2
    
    def changedetector(self):
        modules=[]
        length=[]
        for i in range(self.nlayers):
            modules.append(3*self.div)
            
            length.append(0.63/self.div)
            
        return modules, length, self.nlayers
        