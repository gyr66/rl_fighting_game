import torch

class Config:
  numIters = 50
  numEps = 5
  tempThreshold = 10
  updateThreshold = 0.6
  maxlenOfQueue = 200000
  numMCTSSims = 10 # At each step in a spisode, we perform numMCTSSims simulations. At the next step, we will reuse the search tree. So when the steps per episode are very large, we can reduce the numMCTSSims to reduce the time cost.
  arenaCompare = 1
  cpuct = 10
  checkpoint = "./checkpoint/mcts"
  load_model = False
  load_folder_file = ("./checkpoint/ppo", "ppo_100000")
  numItersForTrainExamplesHistory = 20
  lr = 0.001
  epochs = 10
  batch_size = 64
  cuda = torch.cuda.is_available()
  simulate_frame = 14 # We consider the opponent's action, so the simulate frame can be a little bit large
  game_time_limit = 60
  gamma = 0.99
