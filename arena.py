import logging

from tqdm import tqdm
from fightingice_env import FightingiceEnv
from config import Config

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, env: FightingiceEnv):
        self.player1 = player1
        self.player2 = player2
        self.env = env
        self.engine = env.engine

    def playGame(self):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1] # player1 is the new model, player2 is the old model
        curPlayer = 1 # player1 starts
        frame_data = self.env.init_frame_data
        it = 0
        while True:
            it += 1
            action = players[curPlayer + 1](frame_data)
            opp_action = players[-curPlayer + 1](frame_data)
            frame_data = self.engine.simulate(frame_data, curPlayer == 1, action, opp_action, Config.simulate_frame)
            result = self.engine.check_game_result(frame_data, True, Config.game_time_limit) # Always at the perspective of player1 
            if result != 0:
                return result
            curPlayer = -curPlayer

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames"):
            gameResult = self.playGame()
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
        return oneWon, twoWon, draws

