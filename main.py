# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import keras.models
import numpy as np
import multiprocessing
from multiprocessing import Process, Pipe

from Actor import Actor
from MonteCarloTree import MonteCarloTree
from anet import init_ANN_CNN
from simworlds.hex.HexEnvironmentState import HexEnvironmentState
from config import NimConfig, HexConfig as config
from humanfriendly import format_timespan
from timeit import default_timer as timer

from threading import Thread, Lock


class globalVars():
    def __init__(self):
        pass

    pass


G = globalVars()  # empty object to pass around global state
G.value = 0
G.kill = False
G.show = False
G.skip = 1


def play_game(episode):
    """
    Plays one full game and retur
    :param episode: The episode to load ANET from
    :return: rets: RBUF collected from the full game played
    """
    anet2 = keras.models.load_model(config.SAVE_PATH + str(episode))
    ANET = Actor(anet2)
    game_board = HexEnvironmentState()
    MCT = MonteCarloTree(game_board, ANET)
    RBUF = []

    # Loop goes untill game is finished
    while game_board.winner == 0:
        # Plays ROLLOUTS amount of simulation games
        MCT.uct_search(config.ROLLOUTS)
        move = MCT.best_move()
        game_board.play(move)
        stat = game_board.moves
        stat = {x: 0 for x in game_board.moves}

        # Firstly goes from size*size state to 1D state, with to_play as first element
        # Then appends the case to RBUF
        state_rbuf = MCT.root_state.state
        state = state_rbuf.flatten()
        state_toplay = np.insert(state, 0, MCT.root_state.to_play)
        print(game_board)
        RBUF.append((state_toplay, MCT.statistics(stat)))
        MCT.move(move)

    return RBUF





def multi_threaded(threads=10):
    """
    multi_threaded will play threads amount of games in parallell
    then fitting ANET once on the collected RBUF
    :param threads: The amount of parallel games to run
    :return:
    """
    threads = config.THREADS
    ANET = Actor()
    ANET.save()
    episode = 0
    accuracy = []
    losses = []
    RBUF = []

    n_games = config.EPISODES
    for i in range(0, config.EPISODES + threads, threads):
        start = timer()
        r = multiprocessing.Pool(threads)
        print("Starting progress on game: ", i, "out of: ", n_games)
        # Starts threads amount of full games in parallell
        with r:
            res = r.map(play_game, (episode for x in range(threads)))
            for item in res:
                for ite in item:
                    RBUF.append(ite)

        episode += threads

        history = ANET.fit(RBUF, threads)

        losses.append(history.history['loss'])
        accuracy.append(history.history['accuracy'])

        end = timer()
        seconds_passed = end - start
        print("Game number: ", i, "took ", seconds_passed)
        print("ETA: ", format_timespan(seconds_passed * ((n_games - i) // threads)))
    print(losses)
    print(accuracy)


def single_thread():
    """
    Plays the games sequentially and fitting the ANET after each game played
    :return:
    """
    RBUF = []
    ANET = Actor()
    winners = []
    n_games = config.EPISODES
    history_games = []
    losses = []
    accuracy = []

    for i in range(n_games + 1):
        if G.kill:
            G.kill = False
            return
        start = timer()
        print("Starting progress on game: ", i, "out of: ", n_games)
        if G.skip == 0:
            ask_input()
        G.skip -= 1
        game_board = HexEnvironmentState()
        if G.show:
            print(game_board)

        history_games.append(game_board)
        MCT = MonteCarloTree(game_board, ANET)
        while game_board.winner == 0:
            MCT.uct_search(config.ROLLOUTS)
            move = MCT.best_move()
            game_board.play(move)
            stat = game_board.moves
            stat = {x: 0 for x in game_board.moves}
            state_rbuf = MCT.root_state.state

            sr = state_rbuf.flatten()
            sr = np.insert(sr, 0, MCT.root_state.to_play)

            RBUF.append((sr, MCT.statistics(stat)))
            MCT.move(move)
            if G.show:
                print(game_board)

        winners.append(game_board.winner)
        history = ANET.fit(RBUF)

        losses.append(history.history['loss'])
        accuracy.append(history.history['accuracy'])

        end = timer()
        seconds_passed = end - start
        print("Game number: ", i, "took ", seconds_passed)
        print("ETA: ", format_timespan(seconds_passed * (n_games - i)))
    print("1 wins ", winners.count(1) / n_games, " amount of the time")
    print("1 won: ", winners.count(1), "games out of ", n_games)


def ask_input():
    G.skip += 1
    choice = input("1: skip games\n2: show\n3: kill thread\ninput: ")
    if choice == "1":
        G.show = False
        choice2 = input("Type in how many games to skip: ")
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

def main():
    multi_threaded() if config.MULTI else single_thread()
    #multi_threaded()
    #single_thread()

if __name__ == '__main__':
    main()

