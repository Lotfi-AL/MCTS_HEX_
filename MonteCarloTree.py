import math
from copy import deepcopy
from queue import Queue
from random import random
import random

import numpy as np
from timeit import default_timer as timer
from simworlds.hex.HexEnvironmentState import HexEnvironmentState
from config import HexConfig as config


class Node():
    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.number_simulations = 0  # N
        self.number_rollouts_won = 0  # Q - wins in this position?
        self.children = {}
    
    @property
    def Q(self):
        return self.number_rollouts_won
    
    @property
    def N(self):
        return self.number_simulations
    
    @property
    def value(self):
        explore = 1
        if self.number_simulations == 0:
            return 0 if explore == 0 else math.inf
        else:
            return self.Q / self.N + explore * math.sqrt(
                2 * math.log(self.parent.N) / self.N)
    
    def add_children(self, children):
        for child in children:
            self.children[child.move] = child


class MonteCarloTree():
    def __init__(self, initial_state,ANET):
        self.root = Node(initial_state)
        self.actor = ANET
        self.root_state = deepcopy(initial_state)
    
    def uct_search(self, rollouts):
        for i in range(rollouts):
            node, state = self.select_node()
            to_play = state.to_play
            outcome = self.rollout(state)
            self.backup(node, to_play, outcome)
    
    def rollout(self, state):

        while state.winner == config.PLAYERS['none']:

            s = state.state.flatten()
            s = np.insert(s,0,state.to_play)
            action = self.actor.choose_action(s)
            state.play(action)
        
        return state.winner
    
    def select_node(self) -> tuple:
        """
        Select a node in the tree to preform a single simulation from.

        """
        node = self.root
        state = deepcopy(self.root_state)
        
        # stop if we find reach a leaf node
        while len(node.children) != 0:
            # descend to the maximum value node, break ties at random
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value).value
            max_nodes = [n for n in node.children.values()
                         if n.value == max_value]
            node = random.choice(max_nodes)
            state.play(node.move)
           
            # if some child node has not been explored select it before expanding
            # other children
            if node.N == 0:
                return node, state
        
        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        if self.expand(node, state):
            node = random.choice(list(node.children.values()))
            state.play(node.move)
        return node, state
    
    @staticmethod
    def expand(parent: Node, state) -> bool:
        """
        Generate the children of the passed "parent" node based on the available
        moves in the passed gamestate and add them to the tree.

        Returns:
            bool: returns false If node is leaf (the game has ended).

        """
        children = []
        if state.winner != config.PLAYERS['none']:
            # game is over at this node so nothing to expand
            return False
        
        for move in state.legal_actions(state.board):
            children.append(Node(move, parent))
        
        parent.add_children(children)
        return True
    
    def backup(self, node: Node, to_play, outcome):
        if outcome == to_play:
            reward = 0
        else:
            reward = 1
        
        while node is not None:
            node.number_simulations += 1
            node.number_rollouts_won += reward
            node = node.parent
            reward = reward != 1
    
    def statistics(self,distribution):
        Q = Queue()
        #distribution = {}
        Q.put(self.root)
        while not Q.empty():
            node = Q.get()
            for child in node.children.values():
                Q.put(child)
        
        for move,node in zip(self.root.children,self.root.children.values()):
            distribution[move] = node.N
        return distribution

    
    def move(self, move: tuple) -> None:
        """
        Make the passed move and update the tree appropriately. It is
        designed to let the player choose an action manually (which might
        not be the best action).
        Args:
            move:
        """
        if move in self.root.children:
            child = self.root.children[move]
            child.parent = None
            self.root = child
            self.root_state.play(child.move)
            return
        
        # if for whatever reason the move is not in the children of
        # the root just throw out the tree and start over
        self.root_state.play(move)
        self.root = Node()
    

    
    def best_move(self) -> tuple:
        """
        Return the best move according to the current tree.
        Returns:
            best move in terms of the most simulations number unless the game is over
        """
        if self.root_state.winner != config.PLAYERS['none']:
            return None

        # choose the move of the most simulated node breaking ties randomly
        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        bestchild = random.choice(max_nodes)
        return bestchild.move

