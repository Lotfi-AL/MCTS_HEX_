import itertools
import multiprocessing
import time

import numpy as np

from Actor import Actor
from config import HexConfig as config
from tensorflow import keras
from multiprocessing import Process
from simworlds.hex.HexEnvironmentState import HexEnvironmentState


class globalVars():
    def __init__(self):
        self.ANET = None


G = globalVars()  # empty object to pass around global state
G.value = 0
G.kill = False
G.show = False
G.skip = 1


def ask_input():
    G.skip += 1
    choice = input("1: skip games\n2: show\n3: kill thread \ninput: ")
    if choice == "1":
        choice2 = input("Type in how many games to skip: ")
        G.show = False
        try:
            G.skip = int(choice2)
        except Exception:
            return
    elif choice == "2":
        G.show = True
        choice2 = input("Type in how many games to show: ")
        try:
            G.skip = int(choice2)
        except Exception:
            return
    elif choice == "3":
        G.kill = True
    else:
        G.kill = True


def topp():
    actors = []
    for i in range(config.M + 1):
        r = i * config.STEP
        print("Loading model from timestep: ", r)
        model = keras.models.load_model(config.LOAD_PATH + str(r))
        a = Actor(model, greedy=True)
        actors.append(a)

    games = list(itertools.combinations(list(range(config.M + 1)), 2))
    print(games)
    res = []
    res_h = {}
    total_games = 0
    for itk in set(itertools.chain.from_iterable(games)):
        res_h[itk] = {}

    for item in games:
        if item.__contains__(0):
            total_games += 1
        a1 = actors[item[0]]
        a2 = actors[item[1]]
        play_games(a1, a2, item, res, res_h)
    total_games = total_games * config.GAMES
    for i, item in enumerate(actors):
        print("Actor number: ", i, "wins: ", res.count(item) / total_games)
    res_strings = []
    for key, value in res_h.items():
        res_string = ""
        res_string += "Player ID" + str(key) + " winrates: \n"
        for k, v in value.items():
            res_string += "vs " + str(k) + ": " + str(v) + "\t"
        res_strings.append(res_string)
    for item in res_strings:
        print(item)


def play_games(p1, p2, item, res, res_h):
    time.sleep(1)
    print("Game between: ", item, "has started")
    print("---------------------")

    winners = []
    players = {p1: item[0], p2: item[1]}
    to_play = {1: p1, 2: p2}
    to_play_winner = []
    for i in range(config.GAMES):

        if G.kill:
            G.kill = False
            return
        if G.skip == 0:
            ask_input()
        G.skip -= 1

        print("Game number: ", i, "between", item, "has started")

        game_board = HexEnvironmentState()

        if i % 2 == 0:
            to_play[1] = p1
            to_play[2] = p2
        else:
            to_play[1] = p2
            to_play[2] = p1

        print("Starting player is:", players[to_play[1]])
        while game_board.winner == 0:
            if G.show:
                print(game_board)
            if i == 5:
                print(players[to_play[1]], players[to_play[2]])
            """sr = state.state.flatten()
            sr = np.insert(sr, 0, state.to_play)
            a1 = p1.choose_action(sr)
            game_board.play(a1)"""
            play_move(to_play[1], game_board)

            if game_board.winner != 0:
                if G.show:
                    print(game_board)
                break
            play_move(to_play[2], game_board)
            if game_board.winner != 0:
                if G.show:
                    print(game_board)
                print("Game number: ", i, "between", item, "was won by", players[to_play[game_board.winner]])
                break
        print("Game number: ", i, "between", item, "was won by", players[to_play[game_board.winner]])

        winners.append(to_play[game_board.winner])
        res.append(to_play[game_board.winner])
        to_play_winner.append(game_board.winner)

    print("Game between ", item, " is done ")
    print("Player number 1 wins: ", to_play_winner.count(1) / config.GAMES, "of the time")
    print("Player number 2 wins: ", to_play_winner.count(2) / config.GAMES, "of the time")
    print("Player of id: ", players[p1], "wins: ", winners.count(p1) / config.GAMES, "of the time")
    print("Player of id: ", players[p2], "wins: ", winners.count(p2) / config.GAMES, " of the time")
    print("---------------------")
    res_h[item[0]][item[1]] = winners.count(p1) / config.GAMES
    res_h[item[1]][item[0]] = winners.count(p2) / config.GAMES
    print("\n")


def play_move(p, state):
    sr = state.state.flatten()
    sr = np.insert(sr, 0, state.to_play)
    a = p.choose_action(sr)
    state.play(a)


if __name__ == "__main__":
    topp()
