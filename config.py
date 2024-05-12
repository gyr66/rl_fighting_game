import torch

class Config:
  numIters = 10
  numEps = 10
  tempThreshold = 15
  updateThreshold = 0.6
  maxlenOfQueue = 200000
  numMCTSSims = 25
  arenaCompare = 10
  cpuct = 1
  checkpoint = "./checkpoint/mcts"
  load_model = True
  load_folder_file = ("./checkpoint/mcts", "best_model")
  numItersForTrainExamplesHistory = 20
  lr = 0.001
  epochs = 10
  batch_size = 64
  cuda = torch.cuda.is_available()
