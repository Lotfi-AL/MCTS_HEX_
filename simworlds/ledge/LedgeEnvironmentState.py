import random
from copy import deepcopy

from config import GameMeta
from config import LedgeConfig

def init_action_move():
    action_to_move = []
    for i in range(LedgeConfig.LENGTH):
        for j in range(i,LedgeConfig.LENGTH):
            if i == j and i != 0:
                continue
            action_to_move.append((j,i))
    return action_to_move


class LedgeEnvironmentState:
    def __init__(self):
        self.length = LedgeConfig.LENGTH
        self.copper = LedgeConfig.COPPER
        self.action_to_move = init_action_move()
        self.actions = [x for x in range(len(list(self.action_to_move)))]
        self.gold_pos = 0
        self.board = self.init_board()
        self.history = []
        self.winner = GameMeta.PLAYERS['none']
        self.removed = []
        self.to_play = 1
    
    @property
    def state(self):
        return self.board
    def init_board(self):
        board = [0 for _ in range(self.length)]
        l = [x for x in range(self.length)]
        kk = random.choice(l)
        l.remove(kk)
        board[kk] = 2
        while len(l) > 0 and self.copper > 0:
            kk = random.choice(l)
            l.remove(kk)
            board[kk] = 1
            self.copper -= 1
        return board


    def play(self, move):
        #move = self.get_moves()[move]
        move = self.action_to_move[move]
        pos_from, pos_to = (move)
        
        if self.move(pos_from, pos_to):
            if self.success():
                self.winner = self.to_play
            if self.to_play == 1:
                self.to_play = 2
            else:
                self.to_play = 1
            return True
            #print("Valid move, nim state moves to: ",self.current_episode)
        else:
            return False
            #print("invalid move, nim state remains on: ",self.current_episode)

    def move(self, pos_from, pos_to):
        if pos_to > pos_from:
            return False
        elif self.board[pos_from] == 0 or pos_from > self.length-1:
            return False
        elif pos_from == 0 and pos_to == 0:
            self.save_state(self.board,pos_from,pos_to)
            self.board = deepcopy(self.board)
            self.removed.append(self.board[pos_from])
            self.board[pos_from] = 0
            return True
        elif self.board[pos_to] != 0:
            return False
        
        else:
            self.save_state(self.board, pos_from, pos_to)
            self.board = deepcopy(self.board)
            self.board[pos_from], self.board[pos_to] = self.board[pos_to], self.board[pos_from]
            return True
        
    def success(self):
        if self.removed.__contains__(2):
            return True
        else:
            return False

    def save_state(self, board, pos_from,pos_to):
        self.history.append((board,(pos_from,pos_to),self.to_play))

    def get_legal_actions(self):
        moves = []
        for i in range(self.length):
            if self.board[i] == 0:
                continue
            else:
                nxt = self.find_next_coin(i)
                if nxt == -1:
                    moves.append((i, 0))
                    continue
                for j in range(nxt + 1, i):
                    moves.append((i, j))
        actions = [self.action_to_move.index(x) for x in moves]
        return actions
    
    def get_legal_moves(self):
        moves = []
        for i in range(self.length):
            if self.board[i] == 0:
                continue
            else:
                nxt = self.find_next_coin(i)
                if nxt == -1:
                    moves.append((i,0))
                    continue
                for j in range(nxt+1,i):
                    moves.append((i,j))
     
        return moves
    
    def find_next_coin(self,start):
        for i in range(start-1,-1,-1):
            if self.board[i] != 0:
                return i
        return -1
    
    def final_state(self) -> bool:
        return self.success()
    
if __name__ == "__main__":
    r = LedgeEnvironmentState()
    #while r.get_moves():
     #   pos_from,pos_to = (r.get_moves()[0])
      #  r.play((pos_from,pos_to))
    print(r.action_to_move)
    print(len(r.action_to_move))
    print(r.board)