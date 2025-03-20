import matplotlib.pyplot as plt
import torch

class Plotter():
    def __init__(self):
        self.episode_durations = []
        self.vae_loss = []
        self.wm_loss = []
        self.controller_loss = []

        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.subplots_adjust(hspace=0.4)  # Adjust space between subplots
        
        # Titles for each subplot
        self.plot_titles = ['Episode Durations', 'VAE Loss', 'World Model Loss', 'Controller Loss']
        
        # Initialize the plots
        for ax, title in zip(self.axs.flat, self.plot_titles):
            ax.set_title(title)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Value')

    def plot_training(self):
        # Clear the axes
        for ax in self.axs.flat:
            ax.cla()
    
        # Plot Episode Durations
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        vae_t = torch.tensor(self.vae_loss, dtype=torch.float)
        wm_t = torch.tensor(self.wm_loss, dtype=torch.float)
        controller_t = torch.tensor(self.controller_loss, dtype=torch.float)
    
        self.axs[0, 0].set_title('Episode Durations')
        self.axs[0, 0].plot(durations_t.numpy(), label='Episode Durations')
        # Plot VAE Loss
        self.axs[0, 1].set_title('VAE Loss')
        self.axs[0, 1].plot(self.vae_loss, label='VAE Loss', color='tab:blue')
        # Plot World Model Loss
        self.axs[1, 0].set_title('World Model Loss')
        self.axs[1, 0].plot(self.wm_loss, label='World Model Loss', color='tab:orange')
        # Plot Controller Loss
        self.axs[1, 1].set_title('Controller Loss')
        self.axs[1, 1].plot(self.controller_loss, label='Controller Loss', color='tab:green')
    
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            self.axs[0, 0].plot(means.numpy(), label='100-Episode Mean')
        if len(vae_t) >= 100:
            means = vae_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            self.axs[0, 1].plot(means.numpy(), label='100-Episode Mean', color="orange")
#        if len(wm_t) >= 10:
#            means = wm_t.unfold(0, 10, 1).mean(1).view(-1)
#            means = torch.cat((torch.zeros(99), means))
#            self.axs[1, 0].plot(means.numpy(), label='10-Episode Mean', color="blue")
        if len(controller_t) >= 100:
            means = controller_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            self.axs[1, 1].plot(means.numpy(), label='100-Episode Mean', color="blue")
    
        # Adjust y-limits for the last 200 episodes
        def set_dynamic_ylim(ax, data, label):
            if len(data) >= 200:
                last_200 = data[-200:]
                max_val = max(last_200)
                ax.set_ylim(0, max_val + 0.1 * abs(max_val))
            ax.legend([label])
    
        set_dynamic_ylim(self.axs[1, 1], self.controller_loss, 'Controller Loss')
    
        # Refresh the figure
        plt.tight_layout()
        plt.pause(0.001)  # Pause a bit so that plots are updated


def plot_loss(losses):
    """
    Plots the loss curve after each epoch.

    Args:
        losses (list): A list of loss values recorded during training.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("World Model Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
