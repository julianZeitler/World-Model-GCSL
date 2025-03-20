import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Normal

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
        self.temperature = 1
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
            action (torch.Tensor): Input tensor of shape ([sequence_length], batch_size, action_dim).
            observation (torch.Tensor): Input tensor of shape ([sequence_length], batch_size, latent_dim).

        Returns:
            out (torch.Tensor): Output (Prediction of next observation) of shape ([sequence_length], batch_size, latent_dim).

        Raises:
            ValueError: If the input tensor dimensions are incorrect.
        """
        if self.h_t is None or self.c_t is None:
            raise ValueError(f"Hidden and cell states not initialized. Please call reset_hidden_state!")
        # Add seq_len dim if necessary
        num_action_dims = len(action.shape)
        num_observation_dims = len(observation.shape)
        if num_action_dims == 2:
            action = action.unsqueeze(0)
        if num_observation_dims == 2:
            observation = observation.unsqueeze(0)

        x = torch.cat((action, observation), dim=-1)
        y, (self.h_t, self.c_t) = self.lstm(x, (self.h_t.detach(), self.c_t.detach()))

        # Compute MDN outputs
        pi = F.softmax(self.pi_layer(y)/self.temperature, dim=-1)  # Mixing coefficients
        sigma = torch.exp(self.sigma_layer(y))  # Ensure positive std dev
        mu = self.mu_layer(y)

        if num_action_dims == 2 or num_observation_dims == 2:
            return y.squeeze(0), pi.squeeze(0), sigma.squeeze(0), mu.squeeze(0) 
        else:
            return y, pi, sigma, mu 

    def train_sequence(self, actions: torch.Tensor, observations: torch.Tensor, next_observations: torch.Tensor, mask: torch.Tensor):
        """
        Train the network on a sequence.

        Args:
            actions (torch.Tensor): Input tensor of shape (sequence_length, batch_size, action_dim).
            observations (torch.Tensor): Input tensor of shape (sequence_length, batch_size, latent_dim).
            next_observations (torch.Tensor): Observation the network should predict. (sequence_length, batch_size, latent_dim)
            mask (torch.Tensor): Used to mask ends of sequences, as shorter ones had to be padded with zeros.
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

        loss = mdn_loss(pi, sigma, mu, next_observations)# + 0.2*mse_loss
        loss = loss * mask
        loss = loss.sum() / mask.sum()
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
    #target = target.unsqueeze(2).expand_as(mu)
    target = target.unsqueeze(-2)

    normal_dist = Normal(mu, sigma)
    g_log_probs = normal_dist.log_prob(target)
    g_log_probs = pi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)
    log_prob = max_log_probs.squeeze(-1) + torch.log(probs)

    return -log_prob
    
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
