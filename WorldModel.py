import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MDNRNN(nn.Module):
    def __init__(self, latent_dim: int = 128, action_dim: int = 4, hidden_dim: int = 128, num_layers: int = 1, num_gaussians = 5, lr: float = 0.01):
        super(MDNRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.input_dim = latent_dim + action_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers)
        
        # Mixture Density Network (MDN) output
        self.pi_layer = nn.Linear(hidden_dim, num_gaussians)  # Mixing coefficients
        self.sigma_layer = nn.Linear(hidden_dim, num_gaussians * latent_dim)  # Std deviations
        self.mu_layer = nn.Linear(hidden_dim, num_gaussians * latent_dim)  # Means

        # Hidden and Cell states (need to be managed manually because of online learning)
        self.history = []
        self.h_t = None
        self.c_t = None

    def reset_hidden_state(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reset hidden and cell states.

        Args:
            batch_size (int): Batch size.
        """
        self.h_t = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        self.c_t = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        x = F.layer_norm(self.h_t, self.h_t.shape[-1:])
        latent = x[-1]

        # Compute MDN outputs
        pi = F.softmax(self.pi_layer(latent), dim=1)  # Mixing coefficients
        sigma = torch.exp(self.sigma_layer(latent))  # Ensure positive std dev
        mu = self.mu_layer(latent)

        return latent, pi, sigma, mu 

    def forward(self, action: torch.Tensor, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass for a single time step.

        Args:
            action (torch.Tensor): Input tensor of shape (sequence_length, batch_size, action_dim).
            observation (torch.Tensor): Input tensor of shape (sequence_length, batch_size, latent_dim).

        Returns:
            out (torch.Tensor): Output (Prediction of next observation) of shape (sequence_length, batch_size, latent_dim).

        Raises:
            ValueError: If the input tensor dimensions are incorrect.
        """
        if action.shape[2] != self.action_dim:
            raise ValueError(f"Expected action to have shape (batch_size, {self.action_dim}), but got {action.shape}")
        if observation.shape[2] != self.latent_dim:
            raise ValueError(f"Expected observation to have shape (batch_size, {self.latent_dim}), but got {observation.shape}")
        if self.h_t is None or self.c_t is None:
            raise ValueError(f"Hidden and cell states not initialized. Please call reset_hidden_state!")
        if len(self.history) == 0:
            self.reset_hidden_state(observation.size(dim=0))

        x = torch.cat((action, observation), dim=2)
        _, (self.h_t, self.c_t) = self.lstm(x, (self.h_t.detach(), self.c_t.detach()))
        x = F.layer_norm(self.h_t, self.h_t.shape[-1:])
        latent = x[-1]

        # Compute MDN outputs
        pi = F.softmax(self.pi_layer(latent), dim=1)  # Mixing coefficients
        sigma = torch.exp(self.sigma_layer(latent))  # Ensure positive std dev
        mu = self.mu_layer(latent)

        return latent, pi, sigma, mu 

    def train_sequence(self, actions: torch.Tensor, observations: torch.Tensor, next_observations: torch.Tensor, loss_fn):
        """
        Train the network on a sequence.

        Args:
            actions (torch.Tensor): Input tensor of shape (sequence_length, batch_size, action_dim).
            observations (torch.Tensor): Input tensor of shape (sequence_length, batch_size, latent_dim).
            next_observations (torch.Tensor): Observation the network should predict. (sequence_length, batch_size, latent_dim)
            loss_fn: Loss function
        Returns:
            float: cumulative loss
        """
        self.reset_hidden_state()
        self.optimizer.zero_grad()
        x = torch.cat((actions, observations), dim=2)

        breakpoint()
        y, _ = self.lstm(x)
        y = F.layer_norm(y, y.shape[-1:])
        latent = y[-1]

        pi = F.softmax(self.pi_layer(latent), dim=1)  # Mixing coefficients
        sigma = torch.exp(self.sigma_layer(latent))  # Ensure positive std dev
        mu = self.mu_layer(latent)

        loss = mdn_loss(pi, sigma, mu, next_observations)
        loss.backward()
        self.optimizer.step()

        return loss.item()

def mdn_loss(pi, sigma, mu, target):
    """Compute the negative log-likelihood loss for MDN."""
    breakpoint()
    target = target.unsqueeze(1).expand_as(mu)  # Expand target to match mixture components
    
    # Compute Gaussian probability for each mixture
    gaussian = (1.0 / (torch.sqrt(torch.tensor(2 * torch.pi)) * sigma)) * \
               torch.exp(-0.5 * ((target - mu) / sigma) ** 2)
    
    # Weighted sum of Gaussians
    weighted_gaussians = pi * gaussian
    prob = torch.sum(weighted_gaussians, dim=1)
    
    # Negative log-likelihood loss
    nll_loss = -torch.log(prob + 1e-8).mean()  # Adding small epsilon for stability
    
    return nll_loss

class WorldModelRNN(nn.Module):
    def __init__(self, latent_size: int = 128, action_size: int = 4, hidden_size: int = 128, num_layers=1, lr: float = 0.01):
        """
        Initialize WorldModelRNN

        Args:
            latent_size (int): Size of latent encoding of observations.
            action_size (int): Size of action space.
            hidden_size (int): Size of hidden dimension.
            lr (float): Learning Rate
        """

        super(WorldModelRNN, self).__init__()
        self.input_size = latent_size + action_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size 
        self.num_layers = num_layers

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=num_layers, dropout=0.2)
        self.fc = nn.Linear(self.hidden_size, 64) # only used for training
        self.output_layer = nn.Linear(64, self.latent_size) # only used for training

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=100)

        # Hidden and Cell states (need to be managed manually because of online learning)
        self.history = []
        self.h_t = None
        self.c_t = None

    def reset_hidden_state(self, batch_size: int = 1):
        """
        Reset hidden and cell states.

        Args:
            batch_size (int): Batch size.
        """
        self.h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        self.c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        x = F.layer_norm(self.h_t, self.h_t.shape[-1:])
        return x[-1] # return last layer's hidden state

    def forward(self, action: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass for a single time step.

        Args:
            action (torch.Tensor): Input tensor of shape (sequence_length, batch_size, action_size).
            observation (torch.Tensor): Input tensor of shape (sequence_length, batch_size, latent_size).

        Returns:
            out (torch.Tensor): Output (Prediction of next observation) of shape (sequence_length, batch_size, latent_size).

        Raises:
            ValueError: If the input tensor dimensions are incorrect.
        """
        if action.shape[2] != self.action_size:
            raise ValueError(f"Expected action to have shape (batch_size, {self.action_size}), but got {action.shape}")
        if observation.shape[2] != self.latent_size:
            raise ValueError(f"Expected observation to have shape (batch_size, {self.latent_size}), but got {observation.shape}")
        if self.h_t is None or self.c_t is None:
            raise ValueError(f"Hidden and cell states not initialized. Please call reset_hidden_state!")
        if len(self.history) == 0:
            self.reset_hidden_state(observation.size(dim=0))

        x = torch.cat((action, observation), dim=2)
        _, (self.h_t, self.c_t) = self.rnn(x, (self.h_t.detach(), self.c_t.detach()))
        x = F.layer_norm(self.h_t, self.h_t.shape[-1:])
        return x[-1] # return last layer's hidden state
    
    def online_step(self, action: torch.Tensor, observation: torch.Tensor, next_observation: torch.Tensor, loss_fn):
        """
        Performs a single online learning step.

        Args:
            action (torch.Tensor): Input tensor of shape (sequence_length, batch_size, action_size).
            observation (torch.Tensor): Input tensor of shape (sequence_length, batch_size, latent_size).
            next_observation (torch.Tensor): Observation the network should predict. Shape of (sequence_length, batch_size, latent_size).
            loss_fn: Loss Function.

        Returns:
            float: Loss value for the step.
        """

        with torch.autograd.set_detect_anomaly(True):
            self.optimizer.zero_grad()
            prediction = self(action, observation)
    
            loss = loss_fn(prediction, next_observation)
    
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        return loss.item()

    def train_sequence(self, actions: torch.Tensor, observations: torch.Tensor, next_observations: torch.Tensor, loss_fn):
        """
        Train the network on a sequence.

        Args:
            actions (torch.Tensor): Input tensor of shape (sequence_length, batch_size, action_size).
            observations (torch.Tensor): Input tensor of shape (sequence_length, batch_size, latent_size).
            next_observations (torch.Tensor): Observation the network should predict. (sequence_length, batch_size, latent_size)
            loss_fn: Loss function
        Returns:
            float: cumulative loss
        """
        self.reset_hidden_state()
        self.optimizer.zero_grad()
        x = torch.cat((actions, observations), dim=2)

        y, _ = self.rnn(x)
        y = F.layer_norm(y, y.shape[-1:])
        y = F.leaky_relu(self.fc(y[-1]))
        y = self.output_layer(y)

        loss = loss_fn(y, next_observations)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()


def test_world_model_training(sequence_length: int = 500):
    latent_size = 128
    action_size = 4
    batch_size = 1

    model = WorldModelRNN(latent_size, action_size)
    model.reset_hidden_state()
    criterion = nn.MSELoss()

    dummy_observations = [torch.randn(batch_size, latent_size) for _ in range(sequence_length+1)]
    dummy_actions = [torch.randn(batch_size, action_size) for _ in range(sequence_length)]

    for t in range(sequence_length):
        loss = model.online_step(dummy_actions[t], dummy_observations[t], dummy_observations[t+1], criterion)
 
        print(f"Step {t + 1}, Loss: {loss}")



