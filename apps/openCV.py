import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import models , layers
from keras.utils import load_img
from keras.utils import img_to_array
import os
import numpy as np


class gettingTrainingData :
    def __init__(self , dirname , trainTestSplit , iterator):
        self.dirName = dirname
        self.files = os.listdir(dirname)
        self.all_images = []
        self.all_labels = []
        self.trainTestSplit = trainTestSplit
        self.iterator = iterator

        print("<------- Getting The Data ------->")
        # calling a function to get the data we need to work on
        self.gettingData()



    # used to get the data from the given folder 
    def gettingData(self) :
        clock = 0
        for i in self.files :

            clock += 1

            if clock%self.iterator == 0 :
                print("**** Reached ****")

            if 'dog' in i and clock%self.iterator == 0:
                img = load_img('compImages/'+i)
                self.all_images.append(img_to_array(img))
                self.all_labels.append(0)
            elif clock%self.iterator == 0:
                img = load_img('compImages/'+i)
                self.all_images.append(img_to_array(img))
                self.all_labels.append(1)

         
    #function to split the dataset into training and testing sets
    def splittingTheData(self) :
        
        splitLimit = int(len(self.all_images)*self.trainTestSplit)

        self.testImages = self.all_images[:splitLimit]
        self.testLabels = self.all_labels[:splitLimit]

        self.trainImages = self.all_images[splitLimit:]
        self.trainLabels = self.all_labels[splitLimit:]
        
        print("testSize : " + str(len(self.testLabels)))
        print("trainSize : " + str(len(self.trainLabels)))


        return np.array(self.testImages) , np.array(self.testLabels) , np.array(self.trainImages) , np.array(self.trainLabels)


# this class is used to create the neuralModel
class neuralModel :
    def __init__(self, testImages , testLabels , trainImages , trainLabels) -> None:

        print("Model is getting Created")
        print("Model : Sequential")
        self.testImages = testImages
        self.testLabels = testLabels
        self.trainImages = trainImages
        self.trainLabels = trainLabels
        
    def createModel(self) :
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32 , (3,3) , activation="relu" , input_shape=(32,32,3)))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(64 , (3,3) , activation="relu"))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(64 , (3,3) , activation="relu"))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64 , activation="relu"))
        self.model.add(layers.Dense(2 , activation="softmax"))
        self.model.compile(optimizer="adam" , loss="sparse_categorical_crossentropy" , metrics=["accuracy"])

    def trainModel(self , epoch) :
        print("Started training the model")

        print(self.trainImages[0].shape)

        self.model.fit(self.trainImages , self.trainLabels , epochs=epoch , validation_data=(self.testImages , self.testLabels))
        print("Training Ended")
        print(self.model.summary())

    def saveModel(self,directory = "./savedModels/"):
        self.model.save(directory)

    def updateModel(self,directory="./savedModels/"):
        self.model = models.load_model(directory)
        print("Updated Model")
        print(self.model.summary())

datas = gettingTrainingData("./compImages", 0.2 , 25)

testImages , testLabels, trainImages , trainLabels = datas.splittingTheData()


print("Number of test images : " + str(len(testImages)))
print("Number of train images : " + str(len(trainImages)))


createdModel = neuralModel(testImages , testLabels , trainImages , trainLabels)


createdModel.updateModel()
# createdModel.createModel()  
# createdModel.trainModel(10)
# createdModel.saveModel()