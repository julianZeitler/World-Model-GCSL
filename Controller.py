from typing import Optional

import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
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
        learning_rate: float = 1e-1,
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

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=500)
        #self.loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def select_action_epsilon_greedy(self, state):
        # state is the concatenation of latent_state, predictive_hidden_state and goal
        # Epsilon-Greedy strategy
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.*self.steps_done/self.eps_decay)
        val = random.random()
    
        if val > eps_threshold:
            with torch.no_grad():
                # Select action with policy_net from given state
                out = self.policy_net(state)
                action_index = F.softmax(out, dim=1).max(1).indices
        else:
            action_index = torch.tensor([self.env_action_space.sample()], device=self.device, dtype=torch.long)
        return F.one_hot(action_index, num_classes=self.env_action_space.n).float()

    def select_action_softmax(self, state):
        """
        Selects an action by sampling from the softmax probability distribution of the policy network's output.
        
        Args:
            state (Tensor): Concatenation of latent_state, predictive_hidden_state, and goal.
        
        Returns:
            Tensor: One-hot encoded action tensor.
        """
        with torch.no_grad():
            logits = self.policy_net(state)  # Shape: [batch_size, num_actions]
            
            action_probs = F.softmax(logits, dim=1)
    
            dist = torch.distributions.Categorical(action_probs)
            action_index = dist.sample()  # Shape: [batch_size]
    
        return F.one_hot(action_index, num_classes=self.env_action_space.n).float()

    def optimization_step(self, transition):
        self.steps_done += 1
        concat_state = torch.cat((transition.latent_state, transition.predictive_hidden_state, transition.goal), dim=1)
        logits = self.policy_net(concat_state)

        log_probs = torch.log_softmax(logits, dim=1)
        target = transition.action.argmax(dim=1)

        loss = self.loss_fn(logits, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(loss.item())

        return loss.item()
    
