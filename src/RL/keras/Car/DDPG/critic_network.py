#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" tool """
import numpy as np

""" keras init """
from keras.initializers import normal,identity,VarianceScaling

""" keras model """
from keras.models import load_model
from keras.models import Model

""" keras layers """
from keras.layers import Dense,Input,add
from keras.layers import Flatten
from keras.layers import Concatenate,concatenate
from keras import regularizers

""" keras optimizers """
from keras.optimizers import Adam

import keras.backend as K

""" tensorflow """
import tensorflow as tf

import json

FILENAME = "Critic_Model"
FILENAME_TARGET = "Critic_Target_Model"

class CriticNetwork(object):
    def __init__(self,sess,state_size,action_size,batch_size,TAU,learning_rate):
        self.sess = sess
        self.batch_size = batch_size
        self.TAU = TAU
        self.lr = learning_rate
        self.action_size = action_size

        K.set_session(sess)

        self.Init_Params()
        self.model, self.action, self.state = self.Create_Network(state_size,action_size)
        self.target_model, self.target_action, self.target_state = self.Create_Network(state_size,action_size)
        # print(self.model.summary())
        
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        # self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))
        self.sess.run(tf.global_variables_initializer())

        # self.Load_Model()

    def Init_Params(self):

        """ state hidden layer """
        self.unit_sh1 = 300
        self.activation_sh1 = 'relu'

        self.unit_sh2 = 600
        self.activation_sh2 = 'linear'

        """ action hidden layer """
        self.unit_ah1 = 600
        self.activation_ah1 = 'linear'

        """ merge hidden layer """
        self.unit_mh1 = 600
        self.activation_mh1 = 'relu'

    
    def Gradient(self,states,actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
            })[0]
    
    def Target_Train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)
    
    def Create_Network(self,state_size,action_dim):
        state = Input(shape = (state_size,))
        action = Input(shape = (action_dim,),name="criticAction")

        sh_1 = Dense(name = "state_h1",\
                     units = self.unit_sh1,\
                     activation = self.activation_sh1)(state)
        sh_2 = Dense(name = "state_h2",\
                     units = self.unit_sh2,\
                     activation = self.activation_sh2)(sh_1)
        ah_1 = Dense(name = "action_h1",\
                    units = self.unit_ah1,\
                    activation = self.activation_ah1)(action)

        # mh_layer = add([sh_2,ah_1])
        mh_layer = concatenate([sh_2,ah_1])
        mh_1 = Dense(name = "merge_h1",\
                    units = self.unit_mh1,\
                    activation = self.activation_mh1)(mh_layer)
                    
        # V = Dense(name='output',units = 3,activation = "linear")(mh_1)
        V = Dense(name='output',units = action_dim,activation = "linear")(mh_1)

        model = Model(inputs = [state,action], outputs = V)
        adam = Adam(lr=self.lr)
        model.compile(loss='mse',optimizer=adam)
        return model,action,state
    
    def Evaluate_Actor(self,state):
        if(type(state).__module__ != np.__name__):
            state = np.asarray(state)
        return self.model.predict(state)

    def Evaluate_Target_Actor(self,state):
        # if(type(state).__module__ != np.__name__):
        #     state = np.asarray(state)
        return self.target_model.predict(state)
    
    def Save_Model(self):
        self.model.save_weights(FILENAME, overwrite=True)
        self.target_model.save_weights(FILENAME_TARGET, overwrite=True)
    
    def Save_Weight_JSON(self):
        with open("Critic_Model.json", "w") as f:
            json.dump(self.model.to_json(), f)
    
    def Load_Model(self):
        self.model.load_weights(FILENAME)
        self.target_model.load_weights(FILENAME_TARGET)