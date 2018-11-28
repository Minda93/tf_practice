# -*- coding: utf-8 -*-+

from lib.keyboardINT import interruptDecalre
from lib.mlp import MLP

def main():
    interruptDecalre()
    mlp = MLP()
    mlp.Training()
    # pre = Preprocessing()
    # Single_Display(x_train[0])
    # pre.Multi_Display_prediction(pre.xTrain,pre.yTrain,[],0,10)



if __name__ == "__main__":
    main()