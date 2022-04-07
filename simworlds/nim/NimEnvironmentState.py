
from config import GameMeta
from config import NimConfig

def init_action_move():
    return [x for x in range(1,NimConfig.K+1)]

class NimEnvironmentState:
    def __init__(self, N, K):
        self.N = NimConfig.N
        self.K = NimConfig.K
        self.history = []
        self.winner = GameMeta.PLAYERS['none']
        self.to_play = 1
        self.action_to_move = init_action_move()
        self.actions = [x for x in range(len(list(self.action_to_move)))]

    @property
    def state(self):
        return self.N

    def play(self, n):
        n = self.action_to_move[n]
        #if self.current_episode.action(n):
        if self.move(n):
            if self.success():
                self.winner = self.to_play
            if self.to_play == 1:
                self.to_play = 2
            else:
                self.to_play = 1
            return True
        else:
            return False
            #print("invalid move, nim state remains on: ",self.current_episode)

    def move(self, n):
        if n < 1 or n > self.K:
            self.save_state(self.N, n,self.to_play)
            return False
        else:
            self.save_state(self.N, n,self.to_play)
            self.N -= n
            return True
    
    def save_state(self,N,n,to_play):
        self.history.append((N,n,to_play))
    
    def get_legal_actions(self):
        m = self.get_legal_moves()
        return [self.action_to_move.index(x) for x in m]
    
    def get_legal_moves(self):
        moves = []
        for i in range(1,self.K+1):
            if self.N-i >= 0:
                moves.append(i)
        return moves
                
        
    
    def success(self):
        return self.N == 0

    def failure(self):
        return self.N == 0

    def final_state(self) -> bool:
        return self.success() or self.failure()



