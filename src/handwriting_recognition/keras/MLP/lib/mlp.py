# -*- coding: utf-8 -*-+

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import matplotlib.pyplot as plt

""" preprocessing """
from lib.preprocessing import Preprocessing



class MLP(object):
    r"""
        pre: Preprecessing class
        model:
        param:
            input and hidden layer
                unitH: number of neurals in hidden layer
                input_dim: number of nerurals for input 
                initializerH: weight and bias init for hidden layer
                activationH: activation fucntion for hidden layer
            output layer
                unitO: number of neurals in output layer
                initializerO: weight and bias init for output layer
                activationO: activation fucntion for output layer
            compile
                loss: loss function
                optimizer: 
                metrics:
            train
                validAction:
                epoch:
                batchSize:
                verbose: display train message state
                    0: no display
                    1: display progress bar
                    2: each epoch message
    """
    def __init__(self):
        self.pre = Preprocessing()

        self.model = Sequential()

        """ limit GPU """
        self.Limit_GPU_Memory(0.1)

        """ param """
        self.Set_Model_Param()

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
        """ layer """
        # self.unitH = 256
        self.unitH = 1000
        self.input_dim = self.pre.imgSize
        self.initializerH = 'normal'
        self.activationH = 'relu'

        self.dropoutPercent = 0.5

        self.unitO = self.pre.classSize
        self.initializerO = 'normal'
        self.activationO = 'softmax'

        """ compile """
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'adam'
        self.metrics = ['accuracy']

        """ train """
        self.validAtion = 0.2
        self.epoch = 10
        self.batchSize = 200
        self.verbose = 2 
    
    def Build_Model(self):
        """ hidden 1 """
        self.model.add(Dense(name = "hidden1",\
                             units = self.unitH,\
                             input_dim = self.input_dim,\
                             kernel_initializer = self.initializerH,\
                             activation = self.activationH))

        """ reduce overfitting """
        self.model.add(Dropout(self.dropoutPercent,name="dropout"))
        """ output """
        self.model.add(Dense(name="output",\
                             units = self.unitO,\
                             kernel_initializer = self.initializerO,\
                             activation = self.activationO))

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
        
