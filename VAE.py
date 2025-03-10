import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=5, latent_dim=12, hidden_dim=128):
        """
        VAE for world models: Compress observations in latent space.

        Args:
            input_dim (Int): size of observations
            latent_dim (Int): dimensionality of latent space
            hidden_dim (Int): number of neurons in hidden layers
        """

        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * input_dim * input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 32 * input_dim * input_dim)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Add missing singleton dimensions
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(0)

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc1(h))
        mean = self.fc_mu(h)
        log_var = self.fc_logvar(h) # torch.clamp(self.log_var(x), min=-10, max=10)
        return mean, log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        assert mean.shape == log_var.shape
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # sample with size of std from normal distribution with mean 0 and var 1
        return mean + eps * std # calculate z and return

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc2(z))
        h = F.relu(self.fc3(h))
        h = h.view(h.size(0), 32, 5, 5)
        h = F.relu(self.deconv1(h))
        return torch.sigmoid(self.deconv2(h))  # Sigmoid for binary values

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mean, log_var

def vae_loss_fn(x: torch.Tensor, x_reconstructed: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor, beta: float = 1, target_std: float = 0.1) -> torch.Tensor:
    """
    Loss function for VAE. Combination of cross-entropy and KL divergence for latent space regularization.

    Args:
        x (torch.Tensor): Input into VAE. Tensor of shape (batch_size, input_dim).
        x_reconstructed (torch.Tensor): Output of VAE. Tensor of shape (batch_size, input_dim).
        mean (torch.Tensor): Latent space mean. Tensor of shape(batch_size, latent_dim).
        log_var (torch.Tensor): Latent space log-variance. Tensor of shape(batch_size, latent_dim).

    Returns:
        loss (torch.Tensor). Combined loss. Tensor of shape ( )

    """
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(x_reconstructed, x)

    # Modified KL-divergence that allows the mean to be arbitrary
    target_var = target_std ** 2
    kl_loss = 0.5 * torch.sum(log_var.exp()/target_var + mean.pow(2)/target_var - 1 - log_var + torch.log(torch.tensor(target_var)))
    return recon_loss + beta*kl_loss


