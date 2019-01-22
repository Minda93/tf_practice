#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

from env.car_env_new import CarEnv

""" RL """
from Network.ddpg import DDPG

import math

def main():
    env = CarEnv()
    
    state_size = [2,180]
    action_size = 1
    action_bound = [-math.pi,math.pi]

    ddpg = DDPG(env,state_size,action_size,action_bound)
    ddpg.Sim_Train_HER(0)


if __name__ == '__main__':
    main()