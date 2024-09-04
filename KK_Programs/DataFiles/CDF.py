import sys
import numpy as np

sys.path.append('../../KG_Programs/mods')
#K.GawareckiMethods
from data import Data
from materials30KP_ZB import Materials30KP_ZB
from hamiltoniansKP   import Ham30KP_ZB
from DataMethods import EnchancedBS


class CalcBS:
    def __init__(self,  MaterialName):
        self.Parameters=[]
        self.OptPatameters=[]
        self.BS=[]
        self.MatName= MaterialName
    def CalcAll(self):
        mat       = Materials30KP_ZB(self.MatName, initValues = True)#"rnd_mat"

        path_LG = Data.createPathK(begin = [-1.0, -1.0, -1.0], end =  [0.0, 0.0, 0.0], npoints = 50) # in PI/A
        path_GX = Data.createPathK(begin = [ 0.0,  0.0,  0.0], end =  [2.0, 0.0, 0.0], npoints = 50) # in PI/A
        path = Data.mergePaths(path_LG, path_GX)

        KP_Obj  = EnchancedBS(mat, Data.scalePathK(path,np.pi/mat.a), Ham = Ham30KP_ZB())
        
        KP_Obj.calcBS(calcCompts = False)
        self.BS = KP_Obj.SaveBSToList()
        self.Parameters = [mat.a,
        mat.E1q,
        mat.E5d,
        mat.E3t,
        mat.E1u,
        mat.E5c,
        mat.E1c,
        mat.E5v,
        mat.E1w,
        mat.P0,
        mat.P1,
        mat.P2,
        mat.P3,
        mat.P4,
        mat.P5,
        mat.iP0,
        mat.iP1,
        mat.Q0,
        mat.Q1,
        mat.R0,
        mat.R1,
        mat.deltaD,
        mat.deltaC,
        mat.deltaV,
        mat.ideltaM]

        self.OptPatameters=[mat.P0,
        mat.P1,
        mat.P2,
        mat.P3,
        mat.P4,
        mat.P5,
        mat.iP0,
        mat.iP1,
        mat.Q0,
        mat.Q1,
        mat.R0,
        mat.R1]

    def WriteAsArray(self):
        self.BS=np.array(self.BS)
        self.OptPatameters=np.array(self.OptPatameters)
        self.Parameters=np.array(self.Parameters)
    def ClearAll(self):
        self.BS=[]
        self.OptPatameters=[]
        self.Parameters=[]

class PrepareDataToFiles:
    def __init__(self, MaterialName):
        self.AllBS=[]
        self.AllParameters=[]
        self.AllOptParameters=[]
        self.MatName = MaterialName
        self.BSObject = CalcBS(MaterialName)
    def GenerateOneData(self):
        self.BSObject.ClearAll()
        self.BSObject.CalcAll()
        self.AllBS.append(self.BSObject.BS)
        self.AllParameters.append(self.BSObject.Parameters)
        self.AllOptParameters.append(self.BSObject.OptPatameters)
    def WriteAsArray(self):
        self.AllBS=np.array(self.AllBS)
        self.AllParameters=np.array(self.AllParameters)
        self.AllOptParameters=np.array(self.AllOptParameters)
    def SaveToNumpy(self,dir=''):
        self.WriteAsArray()
        with open(dir+self.MatName+"_BandStucture.npy","wb") as BSdata:
            np.save(BSdata, self.AllBS)
        with open(dir+self.MatName+"_Parameters.npy" ,"wb") as Parametersdata:
            np.save(Parametersdata, self.AllParameters)
        with open(dir+self.MatName+"_OpticalParameters.npy","wb") as OpticalParametersdata:
            np.save(OpticalParametersdata, self.AllOptParameters)
    def SaveToDat(self,dir=''):
        pass


def MainComplete(HowMany = 500, MaterialName = "rnd_mat", Dir = 'NumpyFiles/20000'):
    CreateDataObj = PrepareDataToFiles(MaterialName)
    for i in range(0,HowMany):
        CreateDataObj.GenerateOneData()
        print(i)
    CreateDataObj.SaveToNumpy(dir=Dir)
    pass

MainComplete(HowMany = 1, MaterialName="BN", Dir = "RealMaterials/BN/")
