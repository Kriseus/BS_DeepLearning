import tensorflow 
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, MaxPooling1D, Conv1D, Flatten
from tensorflow.keras.models import Sequential, Model
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self):
        self.MySequence = Sequential()
    def GetModel(self):
        pass
    def CompileAndFit(self, ParsDict, TrainingData, Results, Loss, Optimizer = keras.optimizers.Adam ):
        Epsilon = ParsDict["Epsilon"]
        LearningRate = ParsDict["LearningRate"]
        self.MySequence.compile(loss = Loss, optimizer = Optimizer(learning_rate=LearningRate, epsilon=Epsilon))
        
        self.MySequence.summary()

        BatchSize = ParsDict["BatchSize"]
        Epochs = ParsDict["Epochs"]
        
        self.MyModel.fit(x=TrainingData, y=Results, validation_split = 0.01, epochs = Epochs, batch_size = BatchSize)
    def Save(self, Dir = "", filename = "nic.keras"):
        self.MySequence.save(Dir+filename )


class QuickDenseModel(Model):
    def __init__(self):
        super().__init__()
    def GetModel(self, shape,Activation):
        self.MySequence.add(Input(shape))
        self.MySequence.add(Dense(1024,Activation))
        self.MySequence.add(Dense(512,Activation))
        self.MySequence.add(Dense(256,Activation))
        self.MySequence.add(Dense(128,Activation))
        self.MySequence.add(Dense(64,Activation))
        self.MySequence.add(Dense(12))
    def CompileAndFit(self, *args):
        super().CompileAndFit(*args)
    def Save(self, *args):
        super().Save(*args)

class QuickConvModel(Model):
    def __init__(self):
        super().__init__()
    def GetModel(self, shape, Activation):
        self.MySequence.add(Conv1D(filters = 384, kernel_size = 2, avtivation = Activation, input_shape = shape))
        self.MySequence.add(MaxPooling1D(2, padding = "valid"))
        self.MySequence.add(Conv1D(filters = 256, kernel_size = 2, avtivation = Activation))
        self.MySequence.add(MaxPooling1D(2, padding = "valid"))
        self.MySequence.add(Conv1D(filters = 192, kernel_size = 2, avtivation = Activation))
        self.MySequence.add(MaxPooling1D(2, padding = "valid"))
        self.MySequence.add(Conv1D(filters = 128, kernel_size = 2, avtivation = Activation))
        self.MySequence.add(Flatten())
        self.MySequence.add(Dense(64,activation=Activation))
        self.MySequence.add(Dense(12))
    def CompileAndFit(self, *args):
        super().CompileAndFit(*args)
    def Save(self, *args):
        super().Save(*args)


class QuickBaseModel(Model):
    def __init__(self):
        super().__init__()
    def GetModel(self, shape, Activation):
        self.MySequence.add(Conv1D(filters = 512, kernel_size = 2, activation = Activation, input_shape = shape))
        self.MySequence.add(MaxPooling1D(2, padding = "valid"))
        self.MySequence.add(Conv1D(filters = 384, kernel_size = 2, activation = Activation))
        self.MySequence.add(MaxPooling1D(2, padding = "valid"))
        self.MySequence.add(Conv1D(filters = 256, kernel_size = 2, activation = Activation))
        self.MySequence.add(MaxPooling1D(2, padding = "valid"))
        self.MySequence.add(Conv1D(filters = 192, kernel_size = 2, activation = Activation))
        self.MySequence.add(Flatten())
        self.MySequence.add(Dense(92,activation=Activation))
        self.MySequence.add(Dense(64,activation=Activation))
        self.MySequence.add(Dense(12))        
        return self.MySequence

