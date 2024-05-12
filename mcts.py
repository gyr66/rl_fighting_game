import numpy as np
import math
import logging

from fightingice_env import FightingiceEnv
from model_wrapper import ModelWrapper
from config import Config

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env: FightingiceEnv, model: ModelWrapper):
        self.env = env
        self.engine = env.engine
        self.model = model
        self.model.eval()
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.T = {} # stores the transition

        self.Es = {}  # stores game.getGameEnded ended for board s

    def getActionProb(self, frame_data, player, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for _ in range(Config.numMCTSSims):
            self.search(frame_data, player)

        s = self.engine.get_representation(frame_data, player)
        
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.env.action_space.n)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs
    
    def step(self, frame_data, player, action):
        s = self.engine.get_representation(frame_data, player)
        if (s, action) in self.T:
            return self.T[(s, action)]
        next_frame_data = self.engine.simulate(frame_data, player, action, None, 60)
        self.T[(s, action)] = next_frame_data
        return next_frame_data

    def search(self, frame_data, player):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.engine.get_representation(frame_data, player)

        if s not in self.Es:
            self.Es[s] = self.engine.check_game_result(frame_data, player)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            obs = self.engine.get_obs(frame_data, player)
            self.Ps[s], v = self.model.predict(obs)
            self.Ns[s] = 0
            return -v

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.env.action_space.n):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + Config.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)])
            else:
                u = Config.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act

        next_frame_data = self.step(frame_data, player, a)
        next_player = not player

        v = self.search(next_frame_data, next_player)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
