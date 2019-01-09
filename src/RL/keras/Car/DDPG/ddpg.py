#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

"""  """
import keras.backend as K

""" tensorflow """
import tensorflow as tf

"""  """
import numpy as np

""" lib """
from .actor_network import ActorNetwork
from .critic_network import CriticNetwork
from .replay_buffer import ReplayBuffer

import math


MAX_EPISODES = 1000
MAX_EP_STEPS = 200
GAMMA = 0.99

class DDPG(object):
    def __init__(self,env,state_size,action_size,action_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.env = env

        """ Tensorflow GPU optimization """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

        """ init """
        self.Init_Params()

        """ network """
        self.actor = ActorNetwork(self.sess,state_size,action_size,self.batch_size,self.TAU,self.lr_a)
        self.critic = CriticNetwork(self.sess,state_size,action_size,self.batch_size,self.TAU,self.lr_c)
        
        self.actor.Load_Model()
        self.critic.Load_Model()
        """ replayBuffer """
        self.replay_buffer = ReplayBuffer(self.buffer_size,1337)

    def Init_Params(self):
        """ Target Network HyperParameters """
        self.TAU = 0.001

        """ learning rate """
        self.lr_a = 0.0001
        self.lr_c = 0.001

        """ seed """
        np.random.seed(1337)

        """ buffer size """
        self.buffer_size = 100000

        """ batch size """
        self.batch_size = 32
    
    def Sim_Train_HER(self,train = 0):
        for i in range(MAX_EPISODES):
            state = self.env.reset(True)
            state = np.reshape(state,(1,self.state_size))

            ep_reward = 0
            ep_ave_max_q = 0
            test = 2.0
            for j in range(MAX_EP_STEPS):
            # for j in range(self.env.spec.timestep_limit): 
                loss = 0
                self.env.render()
            
                action = self.actor.Evaluate_Actor(state)
                action_ = np.clip(np.random.normal(action,1), *self.env.action_bound)
                # print(action)
                # if(action[0][0] >= action[0][1]):
                #     action_ = 0
                # else:
                #     action_ = 1

                # tmp = abs(action[0])
                # action_ = np.argmax(tmp)
                # action_ = 0

                # new_state,reward,done,info = self.env.step(action)
                new_state,reward,done,info = self.env.step(action_)
                
                self.replay_buffer.Add_Buffer(np.reshape(state,(self.state_size,)),\
                                              np.reshape(action,(self.action_size,)),\
                                              reward,\
                                              np.reshape(new_state,(self.state_size,)),\
                                              done)

                """ update batch """
                s_batch,a_batch,r_batch,s2_batch,d_batch,y_t = self.replay_buffer.Get_Batch(self.batch_size)

                target_q_value = self.critic.Evaluate_Target_Actor([s2_batch,self.actor.Evaluate_Target_Actor(s2_batch)])

                for k in range(len(r_batch)):
                    if(d_batch[k]):
                        y_t[k] = r_batch[k]
                    else:
                        y_t[k] = r_batch[k] + GAMMA*target_q_value[k]
                if(train):
                    # print(y_t)
                    loss += self.critic.model.train_on_batch([s_batch,a_batch], y_t)
                    a_for_grad = self.actor.Evaluate_Actor(s_batch)
                    grads = self.critic.Gradient(s_batch,a_for_grad)
                    self.actor.Train(s_batch,grads)
                    self.actor.Target_Train()
                    self.critic.Target_Train()

                ep_reward += reward
                state = np.reshape(new_state,(1,self.state_size))
                # print("Episode", i, "Step", j, "Actions", action, "Reward", reward,'loss',loss)
                print("Episode", i, "Step", j,"Actions", action, 'action',action_,"Reward", reward,'loss',loss)
                
                if(done):
                    break
            print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(ep_reward))
            print("Total Step: " + str(j))
            print("")
        if(train):
            print("save model")
            self.actor.Save_Model()
            self.actor.Save_Weight_JSON()
            self.critic.Save_Model()
            self.critic.Save_Weight_JSON()
        print("Finish")


            



