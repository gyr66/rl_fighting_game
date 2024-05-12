from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
import os
from pathlib import Path
from config import Config

class GameDataset(Dataset):
    def __init__(self, examples):
        self.obs, self.pis, self.vs = zip(*examples)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.obs[idx]),
            torch.FloatTensor(self.pis[idx]),
            torch.FloatTensor([self.vs[idx]]),
        )

class ModelWrapper:
    def __init__(self, model: PPO):
        self.model = model
        self.policy = model.policy
        if Config.cuda:
            self.model.policy.action_net().cuda()
            self.model.policy.value_net().cuda()

        self.optimizer = torch.optim.Adam(list(self.policy.action_net.parameters()) + list(self.policy.value_net.parameters()), lr=Config.lr)

    def eval(self):
        self.policy.set_training_mode(False)

    def training(self):
        self.policy.set_training_mode(True)

    def predict(self, data):
        return self.model.predict(data)
    
    def forward(self, obs: torch.Tensor) -> Tuple[np.ndarray, int]:
      features = self.policy.extract_features(obs)
      if self.policy.share_features_extractor:
          latent_pi, latent_vf = self.policy.mlp_extractor(features)
      else:
          pi_features, vf_features = features
          latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
          latent_vf = self.policy.mlp_extractor.forward_critic(vf_features)
      # Evaluate the values for the given observations
      values = self.policy.value_net(latent_vf) # expected return
      values = torch.tanh(values) # map values to [-1, 1]
      actions = self.policy.action_net(latent_pi) # logits before softmax
      return actions, values
    
    def train(self, examples):
        dataset = GameDataset(examples)
        data_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
        self.training()  # sets model to training mode
        for epoch in range(Config.epochs):
            print("Epoch", epoch + 1)

            running_loss = 0.0
            for obs, target_pis, target_vs in data_loader:
                if Config.cuda:
                    obs = obs.cuda()
                    target_pis = target_pis.cuda()
                    target_vs = target_vs.cuda()

                # Forward pass
                out_pi, out_v = self.forward(obs)

                l_pi = nn.CrossEntropyLoss()(out_pi, target_pis)
                l_v = nn.MSELoss()(out_v, target_vs)

                # total_loss = self.nnet.loss(out_pi, out_v, target_pis, target_vs)
                total_loss = l_pi + l_v

                # Backward and optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.item()

            avg_loss = running_loss / len(data_loader)
            print("Average Loss:", avg_loss)
    
    def predict(self, obs):
        self.eval()  # sets model to evaluation mode
        with torch.no_grad():
            obs = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
            if Config.cuda:
                obs = obs.cuda()
            pi, v = self.forward(obs)
            pi = torch.softmax(pi, dim=1) # Convert to probabilities
            return pi.cpu().numpy()[0], v.cpu().numpy()[0, 0]
        
    def save_checkpoint(self, filename, folder="./checkpoint/mcts"):
        filepath = Path(folder) / filename
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.makedirs(folder, exist_ok=True)
        self.model.save(filepath)

    def load_checkpoint(self, folder="./checkpoint/mcts", filename="best_model"):
        filepath = Path(folder) / filename
        self.model.set_parameters(filepath)
