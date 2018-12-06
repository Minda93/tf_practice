# -*- coding: utf-8 -*-+

import numpy as np
# import pandas as pd

''' keras lib '''
from keras.utils import np_utils
from keras.datasets import mnist

''' Drawing '''
import matplotlib.pyplot as plt

np.random.seed(10)

class Preprocessing(object):
    def __init__(self):
        r""" 
            img.shape => 28*28 = 784
            label class => 0~9 = 10
        """
        self.__imgSize = 784
        self.__classSize = 10

        self.__xTrain = None
        self.__yTrain = None
        self.__xTest = None
        self.__yTest = None

        self.__xTrainNorml = None
        self.__yTrainOneHot = None
        self.__xTestNorml = None
        self.__yTestOneHot = None

        self.__xTrain,self.__yTrain,self.__xTest,self.__yTest = self.Load_DataSet()

        self.Processing()
    
    def Load_DataSet(self):
        r'''
            train image(x_train): 60000
            train label(y_label): 60000
            train image(x_train): 10000
            train label(y_label): 10000 
        '''
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
        return x_train,y_train,x_test,y_test
    
    def Processing(self):
        # """ reshape """
        # self.__xTrainNorml = self.__xTrain.reshape(self.__xTrain.shape[0],-1).astype('float32')
        # self.__xTestNorml = self.__xTest.reshape(self.__xTest.shape[0],-1).astype('float32')
        
        # """ normalize """
        # self.__xTrainNorml = self.__xTrainNorml/255
        # self.__xTestNorml = self.__xTestNorml/255

        self.__xTrainNorml = self.Normalize(self.__xTrain)
        self.__xTestNorml = self.Normalize(self.__xTest)

        """ label """
        self.__yTrainOneHot = np_utils.to_categorical(self.__yTrain,num_classes=self.__classSize)
        self.__yTestOneHot = np_utils.to_categorical(self.__yTest,num_classes=self.__classSize)

        print(self.__xTrainNorml.shape)
        print(self.__yTrainOneHot.shape)
        print(self.__xTestNorml.shape)
        print(self.__yTestOneHot.shape)

    def Normalize(self,imgs):
        if(len(imgs.shape) == 3):
            if(imgs.shape[1] == self.__xTrain.shape[1]):
                imgs_ = imgs.reshape(imgs.shape[0],28,28,1).astype('float32')
                imgs_ = imgs_/255
                return imgs_
        #     elif(imgs.shape[1] == self.imgSize):
        #         return imgs
        # elif(len(imgs.shape) == 2):
        #     if(imgs.shape[0] == self.__xTrain.shape[1]):
        #         imgs_ = imgs.reshape(1,-1).astype('float32')
        #         print(imgs_.shape)
        #         imgs_ = imgs_/255
        #         return imgs_
        #     elif(imgs.shape[0] == self.imgSize):
        #         imgs_ = imgs.reshape(1,imgs.shape).astype('float32')
        #         return imgs_


    def Single_Display(self,img):
        fig = plt.gcf()
        fig.set_size_inches(2,2)
        plt.imshow(img,cmap='binary')
        plt.show()

    def Multi_Display_prediction(self,imgs,labels,prediction = [],idx = 0,num=10):
        fig = plt.gcf()
        fig.set_size_inches(12,14)

        if(num>25):
            num = 25
        for i in range(0,num):
            ax = plt.subplot(5,5,1+i)
            ax.imshow(imgs[idx],cmap='binary')
            title = "label= "+ str(labels[idx])
            if(len(prediction) > 0):
                title += " ,predict= "+ str(prediction[idx])

            ax.set_title(title,fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([]) 
            idx += 1
        plt.show()
        
    """ param """
    @property
    def imgSize(self):
        return self.__imgSize
    @property
    def classSize(self):
        return self.__classSize

    ''' database '''
    @property
    def xTrain(self):
        return self.__xTrain
    @property
    def yTrain(self):
        return self.__yTrain
    @property
    def xTest(self):
        return self.__xTest
    @property
    def yTest(self):
        return self.__yTest

    ''' normalize '''
    @property
    def xTrainNorml(self):
        return self.__xTrainNorml
    @property
    def yTrainOneHot(self):
        return self.__yTrainOneHot
    @property
    def xTestNorml(self):
        return self.__xTestNorml
    @property
    def yTestOneHot(self):
        return self.__yTestOneHot

    # @dis.setter
    # def dis(self, value):
    #     self.__dis = value