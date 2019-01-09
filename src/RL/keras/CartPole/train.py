#!/usr/bin/env python3
# -*- coding: utf-8 -*-+
import gym

ENV_NAME = 'Pendulum-v0'
# ENV_NAME = 'MountainCar-v0'
# ENV_NAME = 'CartPole-v0'

from DDPG.ddpg import DDPG

def main():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    if(ENV_NAME == 'Pendulum-v0'):
        s_dim = env.observation_space.shape[0] # 3
        a_dim = env.action_space.shape[0]      # 1
        a_bound = env.action_space.high        # 2.
    elif(ENV_NAME == 'MountainCar-v0'):
        s_dim = env.observation_space.shape[0] # 2
        a_dim = env.action_space.n             # 3
        a_bound = env.observation_space.high
    elif(ENV_NAME == 'CartPole-v0'):
        s_dim = env.observation_space.shape[0] # 4
        a_dim = env.action_space.n             # 2
        a_bound = env.observation_space.high

    print(s_dim)
    print(a_dim)
    print(a_bound)
    # print(env.action_space)
    # print(env.observation_space)
    # print(env.observation_space.high)
    # print(env.observation_space.low)
    # ddpg = DDPG(env,s_dim,a_dim,a_bound)
    
    # ddpg.Sim_Train_HER(1)
    

if __name__ == '__main__':
    main()