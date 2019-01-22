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

        print(state_size,action_size,action_bound)

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
            # state = np.reshape(state,(1,self.state_size[0]+self.state_size[1]))
            pos = np.reshape(state[:self.state_size[0]],(1,self.state_size[0]))
            scan = np.array(state[self.state_size[0]:]).reshape(1,self.state_size[1],1)
            state_ = [pos,scan]

            ep_reward = 0
            for j in range(MAX_EP_STEPS):
                loss = 0
                self.env.render()
                # print(state_)
                action = self.actor.Evaluate_Actor(state_)
                
                action_ = 0
                
                new_state,reward,done,info = self.env.step(action)

                pos = np.reshape(new_state[:self.state_size[0]],(self.state_size[0],))
                scan = np.array(new_state[self.state_size[0]:]).reshape(1,self.state_size[1],1)
                new_state_ = [pos,scan]

                self.replay_buffer.Add_Buffer(state_,\
                                              np.reshape(action,(self.action_size,)),\
                                              reward,\
                                              new_state_,\
                                              done)
                
                s_batch,a_batch,r_batch,s2_batch,d_batch,y_t = self.replay_buffer.Get_Batch(self.batch_size)
                
                s_batch_pos = []
                s_batch_scan = []
                for pos,scan in s_batch:
                    pos = np.reshape(pos,(self.state_size[0]))
                    scan = np.reshape(scan,(self.state_size[1],1))

                    s_batch_pos.append(pos)
                    s_batch_scan.append(scan)
                s_batch_pos = np.asarray(s_batch_pos)
                s_batch_scan = np.asarray(s_batch_scan)

                s2_batch_pos = []
                s2_batch_scan = []
                for pos,scan in s2_batch:
                    pos = np.reshape(pos,(self.state_size[0]))
                    scan = np.reshape(scan,(self.state_size[1],1))

                    s2_batch_pos.append(pos)
                    s2_batch_scan.append(scan)
                s2_batch_pos = np.asarray(s2_batch_pos)
                s2_batch_scan = np.asarray(s2_batch_scan)
                
                target_q_value = self.critic.Evaluate_Target_Actor([s2_batch_pos,s2_batch_scan ,self.actor.Evaluate_Target_Actor([s2_batch_pos,s2_batch_scan])])
                # print(target_q_value)
                for k in range(len(r_batch)):
                    if(d_batch[k]):
                        y_t[k] = r_batch[k]
                    else:
                        y_t[k] = r_batch[k] + GAMMA*target_q_value[k]
                
                if(train):
                    # print(y_t)
                    loss += self.critic.model.train_on_batch([s_batch_pos,s_batch_scan,a_batch], y_t)
                    a_for_grad = self.actor.Evaluate_Actor([s_batch_pos,s_batch_scan])
                    grads = self.critic.Gradient([s_batch_pos,s_batch_scan],a_for_grad)
                    self.actor.Train([s_batch_pos,s_batch_scan],grads)
                    self.actor.Target_Train()
                    self.critic.Target_Train()

                ep_reward += reward
                pos = np.reshape(new_state[:self.state_size[0]],(1,self.state_size[0]))
                scan = np.array(new_state[self.state_size[0]:]).reshape(1,self.state_size[1],1)
                state_ = [pos,scan]

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
