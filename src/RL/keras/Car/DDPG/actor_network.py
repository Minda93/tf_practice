#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" keras init """
from keras.initializers import normal,identity,VarianceScaling

""" keras model """
from keras.models import load_model
from keras.models import Model

""" keras layers """
from keras.layers import Dense,Input
from keras.layers import Flatten,Activation,Lambda
from keras.layers import Concatenate

import keras.backend as K

""" tensorflow """
import tensorflow as tf

""" tool """
import numpy as np

FILENAME = "Actor_Model"
FILENAME_TARGET = "Actor_Target_Model"

import math
import json

A_BOUND = math.pi

class ActorNetwork(object):
    def __init__(self,sess,state_size,action_size,batch_size,TAU,learning_rate):
        self.sess = sess
        self.batch_size = batch_size
        self.TAU = TAU
        self.lr = learning_rate

        K.set_session(sess)

        self.Init_Params()
        self.model, self.weights, self.state = self.Create_Network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.Create_Network(state_size, action_size)

        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

        # self.Load_Model()

    def Init_Params(self):
        self.unit_h1 = 300
        self.activation_h1 = 'relu'

        self.unit_h2 = 300
        self.activation_h2 = 'relu'

    def Train(self, state, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: state,
            self.action_gradient: action_grads})

    def Target_Train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def Create_Network(self,state_size,action_dim):
        state = Input(shape=[state_size])

        h_1 = Dense(name = "h1",\
                         units = self.unit_h1,\
                         activation = self.activation_h1)(state)
        h_2 = Dense(name = "h2",\
                         units = self.unit_h2,\
                         activation = self.activation_h2)(h_1)
        # vec = Dense(1,activation='tanh',kernel_initializer=lambda shape:VarianceScaling(scale=1e-4)(shape))(h_2)
        # yaw = Dense(1,activation='sigmoid',kernel_initializer=lambda shape:VarianceScaling(scale=1e-4)(shape))(h_2)
        # V = Concatenate()([vec,yaw])

        # V  = Dense(action_dim,activation='tanh',kernel_initializer=lambda shape:VarianceScaling(scale=1e-4)(shape))(h_2)

        out = Dense(name = "out",\
                         units = action_dim)(h_2)
        actions = Activation("tanh")(out)
        V = Lambda(lambda x: x * A_BOUND)(actions) 

        model = Model(inputs = state, outputs = V)
        return model, model.trainable_weights, state
    
    def Evaluate_Actor(self,state):
        if(type(state).__module__ != np.__name__):
            state = np.asarray(state)
        return self.model.predict(state)

    def Evaluate_Target_Actor(self,state):
        if(type(state).__module__ != np.__name__):
            state = np.asarray(state)
        return self.target_model.predict(state)
    
    def Save_Model(self):
        self.model.save_weights(FILENAME, overwrite=True)
        self.target_model.save_weights(FILENAME_TARGET, overwrite=True)
    
    def Save_Weight_JSON(self):
        with open("Actor_Model.json", "w") as f:
            json.dump(self.model.to_json(), f)
    
    def Load_Model(self):
        self.model.load_weights(FILENAME)
        self.target_model.load_weights(FILENAME_TARGET)
