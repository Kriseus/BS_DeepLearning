import sys
import numpy as np

sys.path.append('../../KG_Programs/mods')

#K.G.Methods
from data import Data
from materials30KP_ZB import Materials30KP_ZB
from hamiltoniansKP   import Ham30KP_ZB
from DataMethods import EnchancedBS, EnterChange





class CalcBS:
    def __init__(self,  MaterialName):
        self.Parameters=[]
        self.OptParameters=[]
        self.BS=[]
        self.SecondParameters = []
        self.SecondOptParameters = []
        self.SecondBS = []
        self.MatName= MaterialName
    def CalcAll(self):
        mat = Materials30KP_ZB(self.MatName, initValues = True)#"rnd_mat"

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

        self.OptParameters=[mat.P0,
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

        CopyOfParameters = self.Parameters.copy()

        self.SecondParameters, self.SecondOptParameters = EnterChange(CopyOfParameters)
        path_LG = Data.createPathK(begin = [-1.0, -1.0, -1.0], end =  [0.0, 0.0, 0.0], npoints = 50) # in PI/A
        path_GX = Data.createPathK(begin = [ 0.0,  0.0,  0.0], end =  [2.0, 0.0, 0.0], npoints = 50) # in PI/A
        path = Data.mergePaths(path_LG, path_GX)
        
        mat = Materials30KP_ZB("Calc_mat", initValues = True,MyData = self.SecondParameters)#"rnd_mat"
        KP_Obj  = EnchancedBS(mat, Data.scalePathK(path,np.pi/mat.a), Ham = Ham30KP_ZB())
        KP_Obj.calcBS(calcCompts = False)
        self.SecondBS = KP_Obj.SaveBSToList()
        
        
        
    def WriteAsArray(self):
        self.BS=np.array(self.BS)
        self.OptParameters=np.array(self.OptParameters)
        self.Parameters=np.array(self.Parameters)
        self.SecondBS=np.array(self.SecondBS)
        self.SecondOptParameters=np.array(self.SecondOptParameters)
        self.SecondParameters=np.array(self.SecondParameters)
    def ClearAll(self):
        self.BS=[]
        self.OptParameters=[]
        self.Parameters=[]
        self.SecondBS=[]
        self.SecondOptParameters=[]
        self.SecondParameters=[]

class PrepareDataToFiles:
    def __init__(self, MaterialName):
        self.AllBS=[]
        self.AllParameters=[]
        self.AllOptParameters=[]
        self.TwinAllBS=[]
        self.TwinAllParameters = []
        self.TwinAllOptParameters = []
        self.MatName = MaterialName
        self.BSObject = CalcBS(MaterialName)
    def GenerateOneData(self):
        # print(self.TwinAllOptParameters)
        self.BSObject.ClearAll()
        self.BSObject.CalcAll()
        
        self.TwinAllBS.append(self.BSObject.SecondBS)
        self.TwinAllParameters.append(self.BSObject.SecondParameters)
        self.TwinAllOptParameters.append(self.BSObject.SecondOptParameters)

        self.AllBS.append(self.BSObject.BS)
        self.AllParameters.append(self.BSObject.Parameters)
        self.AllOptParameters.append(self.BSObject.OptParameters)
    def WriteAsArray(self):
        self.AllBS=np.array(self.AllBS)
        self.AllParameters=np.array(self.AllParameters)
        self.AllOptParameters=np.array(self.AllOptParameters)
        self.TwinAllBS=np.array(self.TwinAllBS)
        self.TWinAllParameters=np.array(self.TwinAllParameters)
        self.TwinAllOptParameters=np.array(self.TwinAllOptParameters)
    def SaveToNumpy(self,dir=''):
        self.WriteAsArray()
        with open(dir+self.MatName+"_BandStucture.npy","wb") as BSdata:
            np.save(BSdata, self.AllBS)
        with open(dir+self.MatName+"_Parameters.npy" ,"wb") as Parametersdata:
            np.save(Parametersdata, self.AllParameters)
        with open(dir+self.MatName+"_OpticalParameters.npy","wb") as OpticalParametersdata:
            np.save(OpticalParametersdata, self.AllOptParameters)
        with open(dir+"Twin"+self.MatName+"_BandStucture.npy","wb") as TwinBSdata:
            np.save(TwinBSdata, self.TwinAllBS)
        with open(dir+"Twin"+self.MatName+"_Parameters.npy" ,"wb") as TwinParametersdata:
            np.save(TwinParametersdata, self.TwinAllParameters)
        with open(dir+"Twin"+self.MatName+"_OpticalParameters.npy","wb") as TwinOpticalParametersdata:
            np.save(TwinOpticalParametersdata, self.TwinAllOptParameters)
                
    def SaveToDat(self,dir=''):
        pass


def MainComplete(HowMany = 10000, MaterialName = "rnd_mat", Dir = 'SiameseNumpyFiles/10000/'):
    CreateDataObj = PrepareDataToFiles(MaterialName)
    for i in range(0,HowMany):
        CreateDataObj.GenerateOneData()
        print(i)
    CreateDataObj.SaveToNumpy(dir=Dir)
    pass

MainComplete()
