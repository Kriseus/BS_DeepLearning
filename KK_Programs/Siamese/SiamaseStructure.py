from CB import MyCallback 
import tensorflow as tf
from tensorflow import keras
from LossFunctions import MineLossFunction0
import pickle as pick
from RangeDictionaries import MakeLowRange, MakeHighRange
from tensorflow.keras import models, Input
import numpy as np
import sys
sys.path.append('../TheDeep/')
from MLModels2 import QuickBaseModel

dir = '../DataFiles/SiameseNumpyFiles/10000/'
TrainingDataA = np.load(dir+'rnd_mat_BandStucture.npy')
TrainingDataB = np.load(dir+'Twinrnd_mat_BandStucture.npy')
ResultsA = np.load(dir+'rnd_mat_OpticalParameters.npy')
ResultsB = np.load(dir+'Twinrnd_mat_OpticalParameters.npy')
Shape = TrainingDataA.shape[1:]
def MyParams():
    with open("ParamsDict.pkl", 'rb') as f:
        Params = pick.load(f)
    return Params

def MyModel():
    AInput = Input(TrainingDataA.shape[1:])
    BInput = Input(TrainingDataB.shape[1:])

    Inputs = [AInput, BInput]

    MyModel = QuickBaseModel()
    ReturnedModel = MyModel.GetModel(TrainingDataB.shape[1:], "elu")

    outA = ReturnedModel(AInput)
    outB = ReturnedModel(BInput)
    FinalModel = models.Model(Inputs, [outA,outB])
    return FinalModel

def Main():
    FinalModel = MyModel()

    Params = MyParams()

    FinalModel.compile(loss = MineLossFunction0(), optimizer = keras.optimizers.Adam(learning_rate=Params["LearningRate"], epsilon=Params["Epsilon"]))

    TrainingData = [TrainingDataA,TrainingDataB]
    Results = [ResultsA, ResultsB]

    OneCallback = MyCallback()
    FinalModel.summary()
    FinalModel.fit(TrainingData, Results, epochs = Params["Epoch"], batch_size = Params["BatchSize"], validation_split = 0.05, callbacks = [OneCallback])

    print(min(OneCallback.Val_losses))

    FinalModel.save("nic.keras")


Main()