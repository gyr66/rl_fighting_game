from typing import Dict

import os
import numpy as np
import random

class Result:
  """
  Store the result of a game
  """
  def __init__(self, my_hp: int, opponent_hp: int) -> None:
    self.my_hp = my_hp
    self.opponent_hp = opponent_hp

class OpponentStatistic:
  """
  Store the static information of a opponent
  """
  def __init__(self) -> None:
    self.history = [] # record the playing history of the opponent
    self.avg_hp_diff = 0 # the average hp difference between the my_hp and opponent_hp
    self.win = 0 # the number of winning games, from my perspective
    self.lose = 0 # the number of losing games, from my perspective
    self.draw = 0 # the number of draw games
  
  def update(self, result: Result):
    self.avg_hp_diff = (self.avg_hp_diff * len(self.history) + result.my_hp - result.opponent_hp) / (len(self.history) + 1)
    self.history.append(result)
    if result.my_hp > result.opponent_hp:
      self.win += 1
    elif result.my_hp < result.opponent_hp:
      self.lose += 1
    else:
      self.draw += 1
    
  @property
  def score(self):
      return -self.avg_hp_diff
    
class OpponentPool:
  """
  Opponent pool
  """
  def __init__(self) -> None:
    builtin_ai = os.listdir('data/ai')
    builtin_ai = list(map(lambda x: x.split('.')[0], builtin_ai))
    builtin_ai = list(filter(lambda x: x != 'RHEA_PI', builtin_ai)) # make the REHA_PI the final boss to test, we don't learn from it
    self.opponents: Dict[str, OpponentStatistic] = {} # the stats of all the opponents
    for ai in builtin_ai:
      self.opponents[ai] = OpponentStatistic() # intialize the stats of the opponent
  
  def add_ai(self, ai: str):
    self.opponents[ai] = OpponentStatistic()
    
  def update_opponent(self, ai: str, result: Result):
    self.opponents[ai].update(result)
  
  def choose_opponent(self) -> str:
    if random.random() < 0.7:
      scores = np.array([opponent.score for opponent in self.opponents.values()])
      exp_scores = np.exp(scores - np.max(scores))
      probs = exp_scores / np.sum(exp_scores)
      chosen_opponent = np.random.choice(list(self.opponents.keys()), p=probs)
    else:
      chosen_opponent = random.choice(list(self.opponents.keys()))  
    return chosen_opponent
  
  def display_opponent_statistics(self):
    for ai, stats in self.opponents.items():
      print(f"Opponent: {ai}, Win: {stats.win}, Lose: {stats.lose}, Draw: {stats.draw}, Avg HP Diff: {stats.avg_hp_diff}", flush=True)
