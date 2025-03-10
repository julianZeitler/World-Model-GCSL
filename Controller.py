import random
from typing import Optional
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Controller:
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        a_dim: int,
        env_action_space,
        state_dict_path: Optional[str] = None,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay: float = 10000,
        loss_fn = nn.CrossEntropyLoss(),
        optimizer = torch.optim.AdamW,
        learning_rate: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = device
        self.policy_net = PolicyNet(2*z_dim + h_dim, a_dim)
        if state_dict_path:
            self.policy_net.load_state_dict(torch.load(state_dict_path, weights_only=True))

        self.env_action_space = env_action_space

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.steps_done = 0

        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

    def select_action(self, state):
        # state is the concatenation of latent_state, predictive_hidden_state and goal
        # Epsilon-Greedy strategy
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.*self.steps_done/self.eps_decay)
        val = random.random()
    
        if val > eps_threshold:
            with torch.no_grad():
                # Select action with policy_net from given state
                out = self.policy_net(state)
                print("Controller output: ", out)
                action_index = F.softmax(out, dim=1).max(1).indices
        else:
            action_index = torch.tensor([self.env_action_space.sample()], device=self.device, dtype=torch.long)
        return F.one_hot(action_index, num_classes=self.env_action_space.n).float()

    def optimization_step(self, transition):
        self.steps_done += 1
        concat_state = torch.cat((transition.latent_state, transition.predictive_hidden_state, transition.goal), dim=1)
        logits = self.policy_net(concat_state)
    
        loss = self.loss_fn(logits, transition.action)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
