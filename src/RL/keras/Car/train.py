#!/usr/bin/env python3
# -*- coding: utf-8 -*-+


from DDPG.ddpg import DDPG
from DDPG.car_env_new import CarEnv
import math

def main():
    env = CarEnv()

    
    ddpg = DDPG(env,22,1,[-math.pi, math.pi])
    
    ddpg.Sim_Train_HER(0)
    

if __name__ == '__main__':
    main()