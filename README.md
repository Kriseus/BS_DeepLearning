# BS_DeepLearning
Programs create datasets, predicts optical parameters of bulk zincblende III-V materials and allow to compare deep learning models.

The project contains 5 directories: DataFiles, TheDeep, Siamese, TestingAutomatization, ModelsComparison. DataFiles is directory 
which methods work only with connection to programs that I can't put here. Still I put few datasets for simple deep
learning model and siamese deep learning model. In this directory there are also parameters and bandstructures of few real 
materials. In the TheDeep directory there are files responsible for deeplearning models, my callbacks and parameters 
used in machine learning. Siamese directory has implemented my own loss function that makes deep learning process suprevised.
It forces parameters to not go over or under some values (which are not real) and also forces weights of the model to 
predict equal parameters in paired datasets with the smallest possible difference. TestingAutomatization directory allows 
to easily find best parameters of the model(like: learning rate, activation function, batch_size, epsilon ) and register
that parameters to .dat files. ModelsComparison directory compares models (for example siamese and simple convolutional model)
becouse of diffrent loss function in siamese model. It contains also realmaterials class, which gets the data from Datafiles directory.
