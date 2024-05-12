import logging
from collections import deque
from random import shuffle

import numpy as np
from tqdm import tqdm

from mcts import MCTS
from fightingice_env import FightingiceEnv
from model_wrapper import ModelWrapper
from arena import Arena
from config import Config


log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, env: FightingiceEnv, model: ModelWrapper, p_model: ModelWrapper):
        self.env = env
        self.engine = env.engine
        self.model = model
        self.p_model = p_model
        self.mcts = MCTS(self.env, self.model)
        self.trainExamplesHistory = []  # history of examples from Config.numItersForTrainExamplesHistory latest iterations

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        frame_data = self.env.init_frame_data
        self.curPlayer = True
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < Config.tempThreshold)

            pi = self.mcts.getActionProb(frame_data, self.curPlayer, temp=temp)

            obs = self.engine.get_obs(frame_data, self.curPlayer)
            trainExamples.append([obs, pi, None])

            action = np.random.choice(len(pi), p=pi)

            # Update frame_data
            frame_data = self.engine.simulate(frame_data, self.curPlayer,action, None, 60)

            # Check the game result
            r = self.engine.check_game_result(frame_data, self.curPlayer)

            if r != 0:
                return [(x[0], x[1], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]
            
            # Update the current player
            self.curPlayer = not self.curPlayer

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, Config.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            iterationTrainExamples = deque([], maxlen=Config.maxlenOfQueue)

            for _ in tqdm(range(Config.numEps), desc="Self Play"):
                self.mcts = MCTS(self.env, self.model)  # reset search tree
                iterationTrainExamples += self.executeEpisode()

            # save the iteration examples to the history 
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > Config.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.model.save_checkpoint(folder=Config.checkpoint, filename='temp')
            self.p_model.load_checkpoint(folder=Config.checkpoint, filename='temp')
            pmcts = MCTS(self.env, self.p_model)

            # training new network
            self.model.train(trainExamples)
            nmcts = MCTS(self.env, self.model)

            # arena compare new and old model
            # player 1 is the new model, player 2 is the old model
            arena = Arena(lambda x: np.argmax(nmcts.getActionProb(x, True, temp=0)),
                          lambda x: np.argmax(pmcts.getActionProb(x, False, temp=0)), self.env)
            nwins, pwins, draws = arena.playGames(Config.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < Config.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.model.load_checkpoint(folder=Config.checkpoint, filename='temp')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.model.save_checkpoint(folder=Config.checkpoint, filename='checkpoint_' + str(i))
                self.model.save_checkpoint(folder=Config.checkpoint, filename='best_model')