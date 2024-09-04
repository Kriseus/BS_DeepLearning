from abc import ABC, abstractmethod
from numpy import load
from tensorflow import convert_to_tensor as CTT

class Material(ABC):
    @abstractmethod
    def __init__(self):
        self.Name = ""
        self.Parameters = []
        self.BandStructure = []
    def GetBandStructure(self):
        Dir = "../DataFiles/RealMaterials/"+self.Name+"/"
        filename = self.Name+'_BandStucture.npy'
        BS= load(Dir+filename)
        self.BandStructure = CTT(BS)
        return self.BandStructure
    def GetParameters(self):
        Dir = "../DataFiles/RealMaterials/"+self.Name+"/"
        filename = self.Name+'_OpticalParameters.npy'
        Pars = load(Dir+filename)
        self.Parameters = CTT(Pars)
        return self.Parameters

class InAs(Material):
    def __init__(self):
        super().__init__()
        self.Name = "InAs"
    def GetBandStructure(self):
        return super().GetBandStructure()
    def GetParameters(self):
        return super().GetParameters()
    
class BN(Material):
    def __init__(self):
        super().__init__()
        self.Name = "BN"
    def GetBandStructure(self):
        return super().GetBandStructure()
    def GetParameters(self):
        return super().GetParameters()

materialObject = InAs()
print(materialObject.GetBandStructure())
        