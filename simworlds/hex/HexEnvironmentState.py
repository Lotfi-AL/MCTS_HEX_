import itertools

from numpy import zeros, int_

from config import HexConfig as config
from simworlds.hex.UnionFind import UnionFind
import sys
from termcolor import colored, cprint
from colored import fg


def init_action_move():
    return [x for x in range(1, config.BOARD_SIZE + 1)]


# neighbor_patterns = ((-1, 0), (0, -1), (-1, 1), (0, 1), (1, 0), (1, -1))

class HexEnvironmentState:

    def __init__(self):
        self.size = config.BOARD_SIZE
        self.board = zeros((self.size, self.size))
        self.board = int_(self.board)
        # self.board = [[0 for x in range(self.size)] for y in range(self.size)]
        self.moves = [(x, y) for x in range(self.size) for y in range(self.size)]
        self.actions = [x for x in range(len(self.moves))]
        self.action_to_move = [i for i in range(len(self.moves))]
        self.to_play = 1
        self.white_played = 0
        self.black_played = 0
        self.white_groups = UnionFind()
        self.black_groups = UnionFind()
        self.white_groups.set_ignored_elements([config.EDGE1, config.EDGE2])
        self.black_groups.set_ignored_elements([config.EDGE1, config.EDGE2])
        self.history = [self.board]

    @property
    def state(self):
        return self.board

    @staticmethod
    def legal_actions(state):
        moves = []
        for y in range(len(state)):
            for x in range(len(state)):
                if state[x, y] == 0:
                    moves.append((x, y))
        return moves

    def __repr__(self):
        return self.board

    def final_state(self):
        pass

    def play(self, cell):
        """
        Play a stone of the player that owns the current turn in input cell.
        Args:
           cell (tuple): row and column of the cell
        """
        if self.to_play == config.PLAYERS['white']:
            self.place_white(cell)
            self.to_play = config.PLAYERS['black']
        elif self.to_play == config.PLAYERS['black']:
            self.place_black(cell)
            self.to_play = config.PLAYERS['white']
        self.history.append(self.board)

    def place_white(self, cell: tuple) -> None:
        """
        Place a white stone regardless of whose turn it is.
        Args:
            cell (tuple): row and column of the cell
        """
        if self.board[cell] == config.PLAYERS['none']:
            self.board[cell] = config.PLAYERS['white']
            self.white_played += 1
        else:
            raise ValueError("Cell occupied")
        # if the placed cell touches a white edge connect it appropriately
        if cell[0] == 0:
            self.white_groups.join(config.EDGE1, cell)
        if cell[0] == self.size - 1:
            self.white_groups.join(config.EDGE2, cell)
        # join any groups connected by the new white stone
        for n in self.neighbors(cell):
            if self.board[n] == config.PLAYERS['white']:
                self.white_groups.join(n, cell)

    def place_black(self, cell: tuple) -> None:
        """
        Place a black stone regardless of whose turn it is.
        Args:
            cell (tuple): row and column of the cell
        """
        if self.board[cell] == config.PLAYERS['none']:
            self.board[cell] = config.PLAYERS['black']
            self.black_played += 1
        else:
            raise ValueError("Cell occupied")
        # if the placed cell touches a black edge connect it appropriately
        if cell[1] == 0:
            self.black_groups.join(config.EDGE1, cell)
        if cell[1] == self.size - 1:
            self.black_groups.join(config.EDGE2, cell)
        # join any groups connected by the new black stone
        for n in self.neighbors(cell):
            if self.board[n] == config.PLAYERS['black']:
                self.black_groups.join(n, cell)

    def neighbors(self, cell: tuple) -> list:
        """
        Return list of neighbors of the passed cell.
        Args:
            cell tuple):
        """
        x = cell[0]
        y = cell[1]
        return [(n[0] + x, n[1] + y) for n in config.NEIGHBOR_PATTERNS
                if (0 <= n[0] + x < self.size and 0 <= n[1] + y < self.size)]

    @property
    def winner(self) -> int:
        """
        Return a number corresponding to the winning player,
        or none if the game is not over.
        """
        if self.white_groups.connected(config.EDGE1, config.EDGE2):
            return config.PLAYERS['white']
        elif self.black_groups.connected(config.EDGE1, config.EDGE2):
            return config.PLAYERS['black']
        else:
            return config.PLAYERS['none']

    def __str__(self):

        """
        :return: Diamond shaped board with values and edges marked
        """
        br = [[] for x in range(self.size + self.size - 1)]
        br[0].append((0, 0))
        for i in range(self.size + self.size - 1):
            for item in br[i]:
                for k in self.neighbors(item):
                    if k == (item[0] + 1, item[1] + 0):
                        br[i + 1].append(k)
                    if k == (item[0] + 0, item[1] + 1):
                        br[i + 1].append(k)
        for i in range(len(br)):
            br[i] = list(set(br[i]))

        for item in br:
            item.sort(reverse=True)
        n = self.size
        re = self.size ** 2
        b = br
        result = ""
        text = colored('Hello, World!', 'red', attrs=['reverse', 'blink'])
        for i in range(n):
            if i == 0:
                result += "\t "
                result += " " * (n - i - 1)
                result += "2/1"
                result += "\n"

            if i == n - 1:
                result += "  "
                result += " " * (n - i - 1)
                result += "2/1 "
            else:
                result += "\t"
                result += " " * (n - i - 1)
                result += "1 "
            for k in range(i + 1):
                result += str(self.board[b[i][k]]) + " "
            if i == n - 1:
                result += "2/1"
            else:
                result += "2"
            result += "\n"
        r = -1
        for j in range(n - 1, 0, -1):
            r += 2
            result += "\t"
            result += " " * (n - j)
            result += "2 "
            for k in range(j):
                result += str(self.board[b[j + r][k]]) + " "
            result += "1"
            result += "\n"
            if j == 1:
                result += "\t  "
                result += " " * (n - j - 1)
                result += "2/1"
                result += "\n"

        result = result.replace("1", colored("1", "grey", attrs=["reverse", "blink"]))
        result = result.replace("2", colored("2", "white", attrs=["reverse", "blink"]))
        return result


if __name__ == "__main__":
    h = HexEnvironmentState()
    print(h.board)
    # r = HexEnvironmentState.legal_actions(h.board)
    print(h.moves)
    print(h.action_to_move)
    print(h)
