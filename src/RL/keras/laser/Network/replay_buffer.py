#!/usr/bin/env python3
# -*- coding: utf-8 -*-+
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self,buffer_size,seed=123):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
        random.seed(seed)

    def Get_Batch(self,batch_size):
        batch = []
        if(self.num_experiences < batch_size):
            batch = random.sample(self.buffer, self.num_experiences)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        # s_batch = np.array([e[0] for e in batch])
        # a_batch = np.array([e[1] for e in batch])
        # r_batch = np.array([e[2] for e in batch])
        # s2_batch = np.array([e[3] for e in batch])
        # d_batch = np.array([e[4] for e in batch])
        # y_t = np.array([e[1] for e in batch])

        s_batch = [e[0] for e in batch]
        a_batch = np.array([e[1] for e in batch])
        r_batch = np.array([e[2] for e in batch])
        s2_batch = [e[3] for e in batch]
        d_batch = np.array([e[4] for e in batch])
        y_t = np.array([e[1] for e in batch])

        return s_batch,a_batch,r_batch,s2_batch,d_batch,y_t
    
    def Get_Buffer_Size(self):
        return self.buffer_size
    
    def Add_Buffer(self,state,action,reward,new_state,done):
        experience = (state,action,reward,new_state,done)
        if(self.num_experiences < self.buffer_size):
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
    
    def Get_Count(self):
        return self.num_experiences
    
    def Clear(self):
        self.buffer.clear()
        self.num_experiences = 0