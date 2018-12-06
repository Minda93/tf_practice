import os
import signal

''' ctrl+c '''
def keyBoardINT(signum, frame):
    r"""
        Keyboard interrupt
    """
    print('You pressed Ctrl+C!')
    os.kill(os.getpid(), signal.SIGKILL)
    plt.close()


def interruptDecalre():
    r"""
        Declaration of interrupt
    """
    signal.signal(signal.SIGINT, keyBoardINT)
    signal.signal(signal.SIGTERM, keyBoardINT)



if __name__ == "__main__":
    pass