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
        self.temperature = 0.1
        self.sigma_layer = nn.Linear(hidden_dim, num_gaussians * latent_dim)  # Std deviations
        self.mu_layer = nn.Linear(hidden_dim, num_gaussians * latent_dim)  # Means

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=100)

        self.h_t = None
        self.c_t = None

    def reset_hidden_state(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reset hidden and cell states.

        Args:
            batch_size (int): Batch size.
        """
        self.h_t = torch.randn(self.num_layers, batch_size, self.hidden_dim) * 0.01
        self.c_t = torch.randn(self.num_layers, batch_size, self.hidden_dim) * 0.01


        # perform initial fake action
        action = torch.zeros(1, batch_size, self.action_dim)
        observation = torch.zeros(1, batch_size, self.latent_dim)

        return self.forward(action, observation)

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

        x = torch.cat((action, observation), dim=2)
        y, (self.h_t, self.c_t) = self.lstm(x, (self.h_t.detach(), self.c_t.detach()))
        y = y[-1]

        # Compute MDN outputs
        pi = F.softmax(self.pi_layer(y)/self.temperature, dim=1)  # Mixing coefficients
        sigma = torch.exp(self.sigma_layer(y))  # Ensure positive std dev
        mu = self.mu_layer(y)

        return y, pi, sigma, mu 

    def train_sequence(self, actions: torch.Tensor, observations: torch.Tensor, next_observations: torch.Tensor):
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

        y, _ = self.lstm(x)

        pi = F.softmax(self.pi_layer(y)/self.temperature, dim=2)  # Mixing coefficients
        sigma = torch.exp(self.sigma_layer(y))  # Ensure positive std dev
        mu = self.mu_layer(y)

        sampled_y, _ = sample_mdn(pi, sigma, mu)
        mse_loss = F.mse_loss(sampled_y, next_observations)
        loss = mdn_loss(pi, sigma, mu, next_observations) + 0.2*mse_loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

def mdn_loss(pi, sigma, mu, target):
    """Compute the negative log-likelihood loss for MDN."""
    seq_len, batch_size, _ = mu.shape
    _, _, num_gaussians = pi.shape
    _, _, output_dim = target.shape
    mu = mu.view(seq_len, batch_size, num_gaussians, output_dim)
    sigma = sigma.view(seq_len, batch_size, num_gaussians, output_dim)
    target = target.unsqueeze(2).expand_as(mu)
    pi = pi.unsqueeze(-1)
    
    # Compute Gaussian probability for each mixture
    gaussian = (1.0 / (torch.sqrt(torch.tensor(2 * torch.pi)) * sigma)) * \
               torch.exp(-0.5 * ((target - mu) / sigma) ** 2)
    
    # Weighted sum of Gaussians
    weighted_gaussians = pi * gaussian
    prob = torch.sum(weighted_gaussians, dim=2) # sum over mixture components
    
    # Negative log-likelihood loss
    nll_loss = -torch.log(prob + 1e-8).mean()  # Adding small epsilon for stability
    
    return nll_loss

def sample_mdn(pi, sigma, mu):
    """
    Samples from a Gaussian Mixture Model given the MDN parameters.
    
    Args:
    - pi: Tensor of shape ([seq_len], batch_size, num_gaussians), softmax normalized (mixing coefficients)
    - sigma: Tensor of shape ([seq_len], batch_size, num_gaussians * output_dim), positive values (std deviation)
    - mu: Tensor of shape ([seq_len], batch_size, num_gaussians * output_dim) (means)

    Returns:
    - sampled_y: Tensor of shape (batch_size, output_dim), sampled outputs
    - sampled_mu: Tensor of shape (batch_size, output_dim), sampled mean (more stable)
    """
    # Add sequence length dimension if not present
    if len(pi.shape) == 2:
        pi = pi.unsqueeze(0)
    if len(sigma.shape) == 2:
        sigma = sigma.unsqueeze(0)
    if len(mu.shape) == 2:
        mu = mu.unsqueeze(0)

    seq_len, batch_size, num_gaussians = pi.shape
    output_dim = mu.shape[2] // num_gaussians  # Infer output dimension
    
    # Reshape mu and sigma to (seq_len, batch_size, num_gaussians, output_dim)
    mu = mu.view(seq_len, batch_size, num_gaussians, output_dim)
    sigma = sigma.view(seq_len, batch_size, num_gaussians, output_dim)

    # Sample a Gaussian index from the categorical distribution of π
    categorical = torch.distributions.Categorical(pi)
    sampled_indices = categorical.sample().view(seq_len, batch_size, 1, 1)  # Shape: (seq_len, batch_size, 1, 1)

    # Gather the corresponding μ and σ
    sampled_mu = torch.gather(mu, 2, sampled_indices.expand(-1, -1, -1, output_dim)) # Shape: (seq_len, batch_size, 1, output_dim)
    sampled_sigma = torch.gather(sigma, 2, sampled_indices.expand(-1, -1, -1, output_dim)) # Shape: (seq_len, batch_size, 1, output_dim)
    # Remove empty dimension 2 (num_gaussians)
    sampled_mu = sampled_mu.squeeze(2)
    sampled_sigma = sampled_sigma.squeeze(2)

    epsilon = torch.randn_like(sampled_sigma)
    sampled_y = sampled_mu + sampled_sigma * epsilon  # Reparameterization trick

    return sampled_y.squeeze(0), sampled_mu.squeeze(0)

def sample_mdn_mix(pi, sigma, mu):
    """
    Samples from a Gaussian Mixture Model given the MDN parameters.
    
    Args:
    - pi: Tensor of shape (batch_size, num_gaussians), softmax normalized (mixing coefficients)
    - sigma: Tensor of shape (batch_size, num_gaussians * output_dim), positive values (std deviation)
    - mu: Tensor of shape (batch_size, num_gaussians * output_dim) (means)

    Returns:
    - sampled_y: Tensor of shape (batch_size, output_dim), sampled outputs
    - sampled_mu: Tensor of shape (batch_size, output_dim), sampled mean (more stable)
    """

    batch_size, num_gaussians = pi.shape
    output_dim = mu.shape[1] // num_gaussians  # Infer output dimension
    
    # Reshape mu and sigma to (batch_size, num_gaussians, output_dim)
    mu = mu.view(batch_size, num_gaussians, output_dim)
    sigma = sigma.view(batch_size, num_gaussians, output_dim)

    epsilon = torch.randn_like(sigma)  # Sample standard normal noise
    sampled_y = mu + sigma * epsilon  # Reparameterization trick

    # Weighted sum of Gaussians
    weighted_gaussians = pi.unsqueeze(2) * sampled_y
    y = torch.sum(weighted_gaussians, dim=1) # sum over mixture components

    weighted_mu = pi.unsqueeze(2) * mu
    mu = torch.sum(weighted_mu, dim=1)

    return y, mu

