import math
import random

import numpy as np
import tensorflow as tf

from LiteModel import LiteModel
from anet import init_ANN, init_ANN_CNN
from timeit import default_timer as timer
from simworlds.hex.HexEnvironmentState import HexEnvironmentState
from config import HexConfig as GameMeta


class Actor():

    def __init__(self, anet=None, greedy=False) -> None:
        """
        :param anet: ANET to start Actor from, otherwise creates a new ANET
        :param greedy: True if Actor should return best move always
        """
        if anet is None:
            self.V = init_ANN_CNN()
        else:
            self.V = anet
        self.episode = 0
        self.h = HexEnvironmentState
        self.epsilon = 0
        self.l = LiteModel.from_keras_model(self.V)
        self.greedy = greedy

    def choose_action(self, state):
        """
        :param state: State of hexboard as a 1d array
        :return: cell - tuple of best move as predicted by ANET or random legal action with probability epsilon that decays over times
        """
        r = max((GameMeta.DECAY - self.episode) / GameMeta.DECAY, 0)
        epsilon = (GameMeta.P_INIT - GameMeta.P_END) * r + GameMeta.P_END
        epsilon = 0 if self.greedy else epsilon
        features = create_features(state)
        start = timer()
        prob_distribution = self.l.predict_single(features)
        end = timer()
        to_play = state[0]
        state = state[1:]
        state = np.reshape(state, (GameMeta.BOARD_SIZE, GameMeta.BOARD_SIZE))

        legal_actions = self.h.legal_actions(state)
        # legal_actions = self.h.legal_actions(state)#HexEnvironmentState.legal_actions(state)
        moves = [(x, y) for x in range(len(state)) for y in range(len(state))]
        # CNN
        for a in range(len(prob_distribution)):
            # Makes sure only legal actions will be chosen
            if moves[a] not in legal_actions:
                prob_distribution[a] = 0

        if random.random() >= epsilon:
            action = np.argmax(prob_distribution)
            return moves[action]
        else:
            action = random.choice(legal_actions)
            return action

    def save(self, episode=0):
        self.V.save(GameMeta.SAVE_PATH + str(episode))

    def fit(self, RBUF, threads=1):
        """
        :param RBUF: Training data to train model on
        :param threads: How many batches to train on
        :return: history of training
        """
        # self.episode += episodes
        RBUF = np.array(RBUF)
        batch_size = min(GameMeta.FIT_BATCH_SIZE * threads, len(RBUF) // 4)
        # batch_size = len(RBUF)//2

        # Makes random choies from RBUF, without repetition of elements
        indices = np.random.choice(RBUF.shape[0], batch_size, replace=False)

        batch = RBUF[indices]

        y = []
        X = []
        for item in batch:
            features = create_features(item[0])
            X.append(np.array(features))

            # First normalizes the values to sum of 1
            raw = item[1]
            norm = np.array([float(raw[i]) / sum(raw.values()) for i in raw])  # range(len(raw))])
            # One hot encodes the targets
            """norm[norm < np.max(norm)] = 0
            norm[np.argmax(norm)] = 1"""
            y.append(norm)

        y = np.asarray(y)
        X = np.asarray(X)

        X = np.asarray(X).astype('float32')
        rp = self.V(X)

        # if self.episode % GameMeta.STEP == 0:
        #    self.V.save(GameMeta.SAVE_PATH + str(self.episode))

        # Fits model, then saves to file and adds amount of threads to episode since we are fitting multiple at the same time
        history = self.V.fit(X, y)
        self.l = LiteModel.from_keras_model(self.V)
        self.episode += threads
        self.V.save(GameMeta.SAVE_PATH + str(self.episode))
        return history


def create_features(X):
    """

    :param X: 1D array of hex board, containing to play as first element.
    :return: Input transformed to a FEATURES*BOARDSIZE*BOARDSIZE array
    """
    item = X
    x_none = []
    x_p1 = []
    x_p2 = []
    x_toplay1 = []
    x_toplay2 = []

    rosko = np.reshape(np.array(item[1:]), (GameMeta.BOARD_SIZE, GameMeta.BOARD_SIZE))
    to_play = item[0]
    for x in rosko:
        for x_x in x:
            if x_x == 0:
                x_none.append(1)
                x_p1.append(0)
                x_p2.append(0)
            elif x_x == 1:
                x_p1.append(1)
                x_p2.append(0)
                x_none.append(0)
            elif x_x == 2:
                x_p2.append(1)
                x_p1.append(0)
                x_none.append(0)
            if to_play == 1:
                x_toplay1.append(1)
                x_toplay2.append(0)
            else:
                x_toplay1.append(0)
                x_toplay2.append(1)
    res = []
    size = GameMeta.BOARD_SIZE
    res.append(np.reshape(x_none, (size, size)))
    res.append(np.reshape(x_p1, (size, size)))
    res.append(np.reshape(x_p2, (size, size)))
    res.append(np.reshape(x_toplay1, (size, size)))
    res.append(np.reshape(x_toplay2, (size, size)))

    res = np.reshape(res, (5, GameMeta.BOARD_SIZE, GameMeta.BOARD_SIZE))
    return res


def pad_board(board):
    size = GameMeta.BOARD_SIZE
    fs = []
    for i in range(2):
        for j in range(size + 4):
            fs.append(1)

    for i in range(size):
        fs.append(2)
        fs.append(2)
        for j in range(size):
            fs.append(board[j])
            if j == (size - 1):
                fs.append(2)
                fs.append(2)
    for j in range(2):
        for i in range(size + 4):
            fs.append(1)
    return np.reshape(fs, (size + 4, size + 4))


if __name__ == "__main__":
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    p = pad_board(board)
    print(p)
    pr = np.reshape(p, (7, 7))
    print(pr)
