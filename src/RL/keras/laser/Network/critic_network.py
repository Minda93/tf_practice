#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" tool """
import numpy as np

""" keras utils """
from keras.utils import plot_model

""" keras init """
from keras.initializers import normal,identity,VarianceScaling

""" keras model """
from keras.models import load_model
from keras.models import Model

""" keras layers """
from keras.layers import Dense,Input,add
from keras.layers import Flatten,Activation
from keras.layers import Concatenate,concatenate

from keras.layers import MaxPooling1D,AveragePooling1D,Conv1D
from keras.layers import BatchNormalization
from keras.layers import Add

from keras import regularizers

""" keras optimizers """
from keras.optimizers import Adam

import keras.backend as K

""" tensorflow """
import tensorflow as tf

import json

SAVE_PATH = './config/'
FILENAME = SAVE_PATH+"Critic_Model"
FILENAME_TARGET = SAVE_PATH+"Critic_Target_Model"

class CriticNetwork(object):
    r"""
        state_size:
            type: list
            0: pos 
            1: scan
        action_size 
            size: 1
            0: yaw
    """
    def __init__(self,sess,state_size,action_size,batch_size,TAU,learning_rate):
        self.sess = sess
        self.batch_size = batch_size
        self.TAU = TAU
        self.lr = learning_rate
        self.action_size = action_size

        self.Init_Param()
        self.model, self.action, self.state = self.Create_Network(state_size,action_size)
        self.target_model, self.target_action, self.target_state = self.Create_Network(state_size,action_size)
        
        # print('critic model')
        # print(self.model.summary())
        # print('critic state',self.state)
        # plot_model(self.model,to_file='critic.png',show_shapes=True)
        
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())
    
    def Init_Param(self):
        """ scan hidden layer"""
        
        """ sh1 """
        self.filter_sh1 = 64
        self.kernel_size_sh1 = 7
        self.strides_sh1 = 3
        self.padding_sh1 = 'same'
        self.activation_sh1 = 'relu'
        self.pool_kernel_size_sh1 = 3

        """ sh2 """
        self.filter_sh2 = 64
        self.kernel_size_sh2 = 3
        self.strides_sh2 = 1
        self.padding_sh2 = 'same'
        self.activation_sh2 = 'relu'

        """ sh3 """
        self.filter_sh3 = 64
        self.kernel_size_sh3 = 3
        self.strides_sh3 = 1
        self.padding_sh3 = 'same'

        """ merge relu 1 """
        self.activation_m1 = 'relu'

        """ sh4 """
        self.filter_sh4 = 64
        self.kernel_size_sh4 = 3
        self.strides_sh4 = 1
        self.padding_sh4 = 'same'
        self.activation_sh4 = 'relu'

        """ sh5 """
        self.filter_sh5 = 64
        self.kernel_size_sh5 = 3
        self.strides_sh5 = 1
        self.padding_sh5 = 'same'

        """ out """
        self.pool_kernel_size_out = 3
        self.padding_out = 'same'

        """ pos hidden layer"""
        self.unit_ph1 = 300
        self.activation_ph1 = 'relu'

        """ action hidden layer """
        self.unit_ah1 = 600
        self.activation_ah1 = 'linear'

        """ merge """
        self.unit_mh1 = 1024
        self.activation_mh1 = 'relu'

        self.unit_mh2 = 512
        self.activation_mh2 = 'relu'

    
    def Create_Network(self,state_size,action_dim):
        """ state """
        pos = Input(shape=[state_size[0]],name="ActorPos")
        scan = Input(shape=(state_size[1],1),name="criticScan")

        """ action """
        action = Input(shape = (action_dim,),name="criticAction")

        
        """ scan """
        cnn_scan = self.CNN_Laser_One(scan)

        """ pos """
        ph_1 = Dense(name = "pos_h1",\
                    units = self.unit_ph1,\
                    activation = self.activation_ph1)(pos)
        """ action """
        ah_1 = Dense(name = "action_h1",\
                    units = self.unit_ah1,\
                    activation = self.activation_ah1)(action)

        """ merge """
        mh_layer = concatenate([cnn_scan,ph_1,ah_1])
        mh_1 = Dense(name = "merge_h1",\
                    units = self.unit_mh1,\
                    activation = self.activation_mh1)(mh_layer)
        mh_2 = Dense(name = "merge_h2",\
                    units = self.unit_mh2,\
                    activation = self.activation_mh2)(mh_1)
        
        V = Dense(name='output',units = action_dim,activation = "linear")(mh_2)

        model = Model(inputs = [pos,scan,action], outputs = V)
        adam = Adam(lr=self.lr)
        model.compile(loss='mse',optimizer=adam)
        return model,action,[pos,scan]
    
    def CNN_Laser_One(self,input_data):

        """ sh1 """
        sh_1 = Conv1D(filters = self.filter_sh1,\
                      kernel_size = self.kernel_size_sh1,\
                      strides = self.strides_sh1,\
                      padding = self.padding_sh1)(input_data)
        sh_1 = BatchNormalization()(sh_1)
        sh_1 = Activation(self.activation_sh1)(sh_1)

        sh_1 = MaxPooling1D(self.pool_kernel_size_sh1,padding='same')(sh_1)
        
        """ sh2 """
        sh_2 = Conv1D(filters = self.filter_sh2,\
                      kernel_size = self.kernel_size_sh2,\
                      strides = self.strides_sh2,\
                      padding = self.padding_sh2)(sh_1)
        sh_2 = BatchNormalization()(sh_2)
        sh_2 = Activation(self.activation_sh2)(sh_2)

        """ sh3 """
        sh_3 = Conv1D(filters = self.filter_sh3,\
                      kernel_size = self.kernel_size_sh3,\
                      strides = self.strides_sh3,\
                      padding = self.padding_sh3)(sh_2)
        sh_3 = BatchNormalization()(sh_3)

        """ merge relu 1 """
        add_1 = Add()([sh_1,sh_3])
        mRelu_1 = Activation(self.activation_m1)(add_1)

        """ sh4 """
        sh_4 = Conv1D(filters = self.filter_sh5,\
                      kernel_size = self.kernel_size_sh5,\
                      strides = self.strides_sh5,\
                      padding = self.padding_sh5)(mRelu_1)
        sh_4 = BatchNormalization()(sh_4)
        sh_4 = Activation(self.activation_sh4)(sh_4)

        """ sh5 """
        sh_5 = Conv1D(filters = self.filter_sh5,\
                      kernel_size = self.kernel_size_sh5,\
                      strides = self.strides_sh5,\
                      padding = self.padding_sh5)(sh_4)
        sh_5 = BatchNormalization()(sh_5)

        """ merge relu 2 """
        add_2 = Add()([sh_3,sh_5])
        mRelu_2 = Activation(self.activation_m1)(add_2)

        """ out """
        out = AveragePooling1D(self.pool_kernel_size_out,padding='same',name='scan_cnn')(mRelu_2)
        out = Flatten(name='flatten')(out)
        return out

    def Gradient(self,states,actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state[0]: states[0],
            self.state[1]: states[1],
            self.action: actions
            })[0]
    
    def Target_Train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def Evaluate_Actor(self,state):
        # if(type(state).__module__ != np.__name__):
        #     state = np.asarray(state)
        return self.model.predict(state)

    def Evaluate_Target_Actor(self,state):
        # if(type(state).__module__ != np.__name__):
        #     state = np.asarray(state)
        return self.target_model.predict(state)
    
    def Save_Model(self):
        self.model.save_weights(FILENAME, overwrite=True)
        self.target_model.save_weights(FILENAME_TARGET, overwrite=True)
    
    def Save_Weight_JSON(self):
        with open(SAVE_PATH+"Critic_Model.json", "w") as f:
            json.dump(self.model.to_json(), f)
    
    def Load_Model(self):
        self.model.load_weights(FILENAME)
        self.target_model.load_weights(FILENAME_TARGET)



