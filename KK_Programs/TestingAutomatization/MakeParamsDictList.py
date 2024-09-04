import numpy as np
import pickle as pick
from abc import abstractmethod, ABC
import tensorflow as tf
from tensorflow import keras 


class ParamsDict:
    def __init__(self):
        self.Dictionary = {
            "Epsilon": [],
            "LearningRate": [],
            "Epochs": [],
            "BatchSize":[],
            "Activation" : [],
            "Optimizer" : {},
            "Loss" : {}
        }
    def SetBatchSize(self, Base, HowMany):
        BatchSize = []
        for i in range(0,HowMany):
            BatchSize.append(Base * (i+1))

        self.Dictionary["BatchSize"] = BatchSize.copy()
        BatchSize = []
    def SetEpochs(self, Base, HowMany):
        Epochs = []
        for i in range(0,HowMany):
            Epochs.append(Base * (i+1))

        self.Dictionary["Epochs"] = Epochs.copy()
        Epochs = []
    def SetLearningRate(self, From, To, HowMany):
        LearningRate = []
        for i in range(0,HowMany):
            LearningRate.append(((To-From)/HowMany)*i+From)
        self.Dictionary["LearningRate"]=LearningRate.copy()
        LearningRate = []
    def SetEpsilon(self, From, To, HowMany):
        Epsilon = []
        for i in range(0,HowMany):
            Epsilon.append(((To-From)/HowMany)*i+From)
        self.Dictionary["Epsilon"]=Epsilon.copy()
        Epsilon = []
    def SetOptimizer(self, Opt):
        optimizers_dict = {
        "Adam": keras.optimizers.Adam,
        "SGD": keras.optimizers.SGD,
        "RMSprop": keras.optimizers.RMSprop,
        "Adadelta": keras.optimizers.Adadelta,
        "Adagrad": keras.optimizers.Adagrad,
        # "FTRL": keras.optimizers.FTRL,
        "Nadam": keras.optimizers.Nadam,
        # "LBFGS": keras.optimizers.LBFGS
        }
        
        Optimizer = {}
        Optimizer[Opt] = optimizers_dict[Opt]
        self.Dictionary["Optimizer"] = Optimizer
        Optimizer = {}
    def SetLossFunction(self, loss):
        loss_functions_dict = {
        "BinaryCrossentropy": keras.losses.BinaryCrossentropy,
        "CategoricalCrossentropy": keras.losses.CategoricalCrossentropy,
        "SparseCategoricalCrossentropy": keras.losses.SparseCategoricalCrossentropy,
        "MSE": keras.losses.MeanSquaredError,
        "MAE": keras.losses.MeanAbsoluteError,
        "Hinge": keras.losses.Hinge,
        "KLDivergence": keras.losses.KLDivergence,
        "CosineSimilarity": keras.losses.CosineSimilarity
        }   
        LossFunc = {}
        LossFunc[loss] = loss_functions_dict[loss]
        self.Dictionary["Loss"] = LossFunc
        LossFunc = {}

    def SetActivation(self, Activ):
        # Activ = [
        # "linear",
        # "relu",
        # "leaky_relu",
        # "elu",
        # "swish",
        # "gelu",
        # "mish"
        # ]   
        self.Dictionary["Activation"] = Activ

    def SaveDictionary(self, dir='', filename = "ParamsDict.pkl"):
        with open(dir+filename, 'wb') as f:
            pick.dump(self.Dictionary, f)



activation_functions = [
    # "linear",
    # "relu",
    "leaky_relu",
    # "elu",
    # "swish",
    "gelu",
    "mish"
]



Dict = ParamsDict()
Dict.SetBatchSize(101,1)
Dict.SetEpochs(5,1) 
Dict.SetEpsilon(1e-6,1e-6,1)
Dict.SetLearningRate(1e-7,1e-5,3)
Dict.SetOptimizer("Adam")
Dict.SetActivation(activation_functions)
Dict.SetLossFunction("MSE")
print(Dict.Dictionary)
Dict.SaveDictionary()