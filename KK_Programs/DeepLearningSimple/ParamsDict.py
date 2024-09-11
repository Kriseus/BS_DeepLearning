import numpy as np
import pickle as pick
from abc import abstractmethod, ABC



class ParamsDict:
    def __init__(self):
        self.Dictionary = {
            "Epsilon":1e-5,
            "LearningRate":1e-6,
            "Epoch":2048,
            "BatchSize":64
        }
    def SetBatchSize(self,BatchSize):
        self.Dictionary["BatchSize"] = BatchSize
    def SetEpochs(self,Epochs):
        self.Dictionary["Epoch"] = Epochs
    def SetLearningRate(self,LearningRate):
        self.Dictionary["LearningRate"] = LearningRate
    def SetEpsilon(self,Epsilon):
        self.Dictionary["Epsilon"] = Epsilon
    def SaveDictionary(self, dir='', filename = "ParamsDict.pkl"):
        with open(dir+filename, 'wb') as f:
            pick.dump(self.Dictionary, f)



Dict = ParamsDict()
Dict.SetBatchSize(101)
Dict.SetEpochs(250) 
Dict.SetEpsilon(1e-5)
Dict.SetLearningRate(1e-7)
Dict.SaveDictionary()