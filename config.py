
class NimConfig():
    N = 10
    K = 3
    DIMENSIONS = (1, 2, 3)
    LR = 3e-4

class LedgeConfig():
    LENGTH = 10
    COPPER = 2
    DIMENSIONS = (1, 23, 46)
    LR = 3e-4

class ANNConfig():
    FIT_BATCH_SIZE = 16
    BATCH_SIZE = 1
    DECAY = 25
    P_INIT = 0.3
    P_END = 0.01
    optimizers = ["SGD","Adam","Adagrad","RMSprop"]
    OPT =optimizers[3]
    LR = 1e-3
    BOARD_SIZE = 5
    INPUT_DIM = (BOARD_SIZE**2+1,)
    CNN_INPUT_DIM_1D = (1,BOARD_SIZE**2+1)
    CNN_INPUT_DIM = (5,BOARD_SIZE,BOARD_SIZE)
    OUTPUT_DIM = BOARD_SIZE**2

    DIMENSIONS = (BOARD_SIZE**2+1,32,BOARD_SIZE**2)

    SAVE_PATH = 'savedmodels/'
    LOAD_PATH = 'savedmodels/'

class HexConfig(ANNConfig):
    THREADS = 10
    PLAYERS = {'none': 0, 'black': 1, 'white': 2}
    MULTI = True
    EDGE1 = 1
    EDGE2 = 2
    NEIGHBOR_PATTERNS = ((-1, 0), (0, -1), (-1, 1), (0, 1), (1, 0), (1, -1))
    GAME_OVER = -1
    M = 10
    EPISODES = 500
    STEP = EPISODES//M
    ROLLOUTS = 500
    GAMES = 50

