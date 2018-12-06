# -*- coding: utf-8 -*-+

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Flatten,Conv2D,MaxPooling2D

import matplotlib.pyplot as plt

""" preprocessing """
from lib.preprocessing import Preprocessing

class CNN(object):
    def __init__(self):
        self.pre = Preprocessing()

        self.model = Sequential()

        """ limit GPU """
        self.Limit_GPU_Memory(0.1)

        """ param """
        self.Set_Model_Param()
        self.Set_Train_Param()

        """ build model """
        self.Build_Model()

    def Limit_GPU_Memory(self,percent):
        """ limit GPU mermory resource """
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = percent
        set_session(tf.Session(config=config))
    
    def Set_Model_Param(self):
        """ convolution layer """

        """ conv1 """
        self.filters_c1 = 16
        self.kernelSize_c1 = (5,5)
        self.padding_c1 = 'same'
        self.inputShape_c1 = (28,28,1)
        self.activation_c1 = 'relu'
        
        """ pool1 """
        self.poolSize_p1 = (2,2)

        """ conv2 """
        self.filters_c2 = 36
        self.kernelSize_c2 = (5,5)
        self.padding_c2 = 'same'
        self.activation_c2 = 'relu'

        """ pool2 """
        self.poolSize_p2 = (2,2)

        """ dropout """
        self.dropout_conv = 0.25
        
        """ neural Network """
        r"""
            flatten
                input : filters_c2*pool_ouput_img_rows*pool_ouput_img_cols*channels 
        """
        """ hidden """
        self.unit_h1 = 128
        self.activation_h1 = 'relu'

        """ dropout """
        self.dropout_hidden = 0.5

        """ output """
        self.unit_out = self.pre.classSize
        self.activation_out = 'softmax'
    
    def Set_Train_Param(self):
        """ compile """
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'adam'
        self.metrics = ['accuracy']

        """ train """
        self.validAtion = 0.2
        self.epoch = 10
        self.batchSize = 300
        self.verbose = 2 
    
    def Build_Model(self):
        """ convolution layer """
        """ conv1 """
        self.model.add(Conv2D(name = 'conv1',\
                              filters = self.filters_c1,\
                              kernel_size = self.kernelSize_c1,\
                              padding = self.padding_c1,\
                              input_shape = self.inputShape_c1,\
                              activation = self.activation_c1))
        """ pool1 """
        self.model.add(MaxPooling2D(name = 'pool1',\
                                    pool_size = self.poolSize_p1))

        """ conv2 """
        self.model.add(Conv2D(name = 'conv2',\
                              filters = self.filters_c2,\
                              kernel_size = self.kernelSize_c2,\
                              padding = self.padding_c2,\
                              activation = self.activation_c2))

        """ pool2 """
        self.model.add(MaxPooling2D(name = 'pool2',\
                                    pool_size = self.poolSize_p2))
        
        """ dropout """
        self.model.add(Dropout(self.dropout_conv,name="droput_conv"))

        """ neural Network """
        """ flatten """
        self.model.add(Flatten(name="flatten"))

        """ hidden1 """
        self.model.add(Dense(name = "hidden1",\
                             units = self.unit_h1,\
                             activation = self.activation_h1))
        """ dropout """
        self.model.add(Dropout(self.dropout_hidden,name="droput_hidden"))

        """ output """
        self.model.add(Dense(name="output",\
                             units = self.unit_out,\
                             activation = self.activation_out))
        
        """ compile """
        self.model.compile(loss = self.loss,\
                           optimizer = self.optimizer,\
                           metrics = self.metrics)

        print(self.model.summary())

    def Training(self):
        trainHistory = self.model.fit(x = self.pre.xTrainNorml,\
                                      y = self.pre.yTrainOneHot,\
                                      validation_split = self.validAtion,\
                                      epochs = self.epoch,\
                                      batch_size = self.batchSize,\
                                      verbose = self.verbose)
        
        self.Show_Train_Result(trainHistory,'acc','val_acc')
        self.Show_Train_Result(trainHistory,'loss','val_loss')

        self.Evaluate_Model(self.pre.xTestNorml,self.pre.yTestOneHot)
        self.Predict(self.pre.xTest,self.pre.yTest,show=True)
    
    def Evaluate_Model(self,xTest,yTest):
        r""" 
            scores: size => 2
                0: loss function value
                1: accuracy

        """
        scores = self.model.evaluate(xTest,yTest)
        print('accuracy = ',scores[1])
    
    def Predict(self,xTest,yTest,idx=0,num=10,show=False):
        prediction = self.model.predict_classes(self.pre.Normalize(xTest))
        
        if(show):
            self.pre.Multi_Display_prediction(xTest,yTest,prediction,idx,num)
            self.Show_Confusion(yTest,prediction)
            
    def Show_Confusion(self,yTest,prediction):
        confusion = pd.crosstab(yTest,prediction,rownames=['label'],colnames=['predict'])
        print(confusion)
    
    def Show_Train_Result(self,trainHistory,train,validation):
        plt.plot(trainHistory.history[train])
        plt.plot(trainHistory.history[validation])
        plt.title('train history')
        plt.xlabel('epochs')
        plt.ylabel('train')
        plt.legend(['train','validation'],loc='upper left')
        plt.show()
