from torch.optim.lr_scheduler import ReduceLROnPlateau
from Controller import Controller
from Environment import GridWorld

from itertools import count
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, TensorDataset

from Memory import TransitionGCSL, ReplayMemory, ExperienceMemory
from VAE import VAE, vae_loss_fn
from WorldModel import MDNRNN, sample_mdn, sample_mdn_mix
from worlds import world_small, world_big, small_worlds

VAE_DIR = "models/vae_small_latent_12.pth"
VAE_TRAIN = False
WM_DIR = "models/world_model_01.pth"
#WM_TRAIN = True
WM_TRAIN = False
CONTROLLER_DIR = "models/controller_01.pth"
#CONTROLLER_TRAIN = True
CONTROLLER_TRAIN = False

OBS_SIZE = 5
LATENT_DIM = 12
HIDDEN_DIM = 64
ACTION_SIZE = 4

WORLD_INDEX = 0
VAE_BATCH_SIZE = 32

grid_worlds = [GridWorld(width=world.shape[0], height=world.shape[1], horizon=100, obstacles=np.argwhere(world==1)) for world in small_worlds]

controller = Controller(z_dim=LATENT_DIM, h_dim=HIDDEN_DIM, a_dim=ACTION_SIZE, env_action_space=grid_worlds[WORLD_INDEX].action_space)
vae = VAE(input_dim=OBS_SIZE, latent_dim=LATENT_DIM, hidden_dim=256)
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-5)
vae_scheduler = ReduceLROnPlateau(vae_optimizer, mode='min', factor=0.1, patience=10000)
world_model = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_SIZE, hidden_dim=HIDDEN_DIM, num_layers=2, num_gaussians=2, lr=1e-3)

# init world model weights
for name, param in world_model.named_parameters():
    if "weight_hh" in name:
        nn.init.xavier_uniform_(param)
    elif "bias" in name:
        nn.init.zeros_(param)

replay_memory = ReplayMemory(10000)
experience_memory = ExperienceMemory(10000)

episode_durations = []
vae_loss = []
wm_loss = []
controller_loss = []
# Initialize figure and axes globally
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(hspace=0.4)  # Adjust space between subplots

# Titles for each subplot
plot_titles = ['Episode Durations', 'VAE Loss', 'World Model Loss', 'Controller Loss']

# Initialize the plots
for ax, title in zip(axs.flat, plot_titles):
    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value')

def plot_training(episode_durations, vae_loss, wm_loss, controller_loss, show_result=False):
    # Clear the axes
    for ax in axs.flat:
        ax.cla()

    # Plot Episode Durations
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    vae_t = torch.tensor(vae_loss, dtype=torch.float)
    wm_t = torch.tensor(wm_loss, dtype=torch.float)
    controller_t = torch.tensor(controller_loss, dtype=torch.float)

    axs[0, 0].set_title('Episode Durations' if not show_result else 'Result')
    axs[0, 0].plot(durations_t.numpy(), label='Episode Durations')
    # Plot VAE Loss
    axs[0, 1].set_title('VAE Loss')
    axs[0, 1].plot(vae_loss, label='VAE Loss', color='tab:blue')
    # Plot World Model Loss
    axs[1, 0].set_title('World Model Loss')
    axs[1, 0].plot(wm_loss, label='World Model Loss', color='tab:orange')
    # Plot Controller Loss
    axs[1, 1].set_title('Controller Loss')
    axs[1, 1].plot(controller_loss, label='Controller Loss', color='tab:green')

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        axs[0, 0].plot(means.numpy(), label='100-Episode Mean')
    if len(vae_t) >= 100:
        means = vae_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        axs[0, 1].plot(means.numpy(), label='100-Episode Mean', color="orange")
    if len(wm_t) >= 100:
        means = wm_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        axs[1, 0].plot(means.numpy(), label='100-Episode Mean', color="blue")
    if len(controller_t) >= 100:
        means = controller_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        axs[1, 1].plot(means.numpy(), label='100-Episode Mean', color="blue")
    #axs[0, 0].legend()
    #axs[0, 1].legend()
    #axs[1, 0].legend()
    #axs[1, 1].legend()

    # Adjust y-limits for the last 200 episodes
    def set_dynamic_ylim(ax, data, label):
        if len(data) >= 200:
            last_200 = data[-200:]
            max_val = max(last_200)
            ax.set_ylim(0, max_val + 0.1 * abs(max_val))
        ax.legend([label])

    #set_dynamic_ylim(axs[0, 1], vae_loss, 'VAE Loss')
    #set_dynamic_ylim(axs[1, 0], wm_loss, 'World Model Loss')
    set_dynamic_ylim(axs[1, 1], controller_loss, 'Controller Loss')

    # Refresh the figure
    plt.tight_layout()
    plt.pause(0.001)  # Pause a bit so that plots are updated

# Train VAE
if not VAE_TRAIN:
    vae.load_state_dict(torch.load(VAE_DIR, weights_only=True))
    print("Loaded VAE")
else:
    # Collect all agent observations from grid worlds
    observations = [world.get_all_obs() for world in grid_worlds]
    observations = np.array([obs["agent"] for world_obs in observations for obs in world_obs])

    # Convert observations to a PyTorch tensor
    observations_tensor = torch.tensor(observations).unsqueeze(1)  # Add channel dim (N, 1, H, W)
    
    # Create a DataLoader for batching
    dataset = TensorDataset(observations_tensor)
    dataloader = DataLoader(dataset, batch_size=VAE_BATCH_SIZE, shuffle=True)

    print("Training VAE...")
    for epoch in range(100000 // len(dataloader)):  # Iterate over epochs
        for batch in dataloader:
            batch_obs = batch[0]  # Get the batch
            
            vae_optimizer.zero_grad()
            obs_reconstructed, mean, log_var = vae(batch_obs)
            loss = vae_loss_fn(batch_obs, obs_reconstructed, mean, log_var, beta=0.0001, target_std=0.1)
            
            loss.backward()
            vae_optimizer.step()
            vae_scheduler.step(loss)
            vae_loss.append(loss.item())

        # Log every 100 iterations
        if epoch % 100 == 0:
            plot_training(episode_durations, vae_loss, wm_loss, controller_loss)

    torch.save(vae.state_dict(), VAE_DIR)


def show_hidden(state):
    plt.figure()
    plt.imshow(vae.decode(state).squeeze().detach().numpy())
    plt.show(block=False)

if not WM_TRAIN:
    world_model.load_state_dict(torch.load(WM_DIR, weights_only=True))
    print("Loaded world model")

if not CONTROLLER_TRAIN:
    controller.policy_net.load_state_dict(torch.load(CONTROLLER_DIR, weights_only=True))
    print("Loaded controller")

goal = np.full((5, 5), 0, dtype=np.float32)
for i in range(1200):
    print("Epoch: " + str(i))
    state, info = grid_worlds[WORLD_INDEX].reset(goal)
    predictive_hidden_state, _, _, _ = world_model.reset_hidden_state()

    with torch.no_grad():
        latent_state, _ = vae.encode(torch.tensor(state["agent"]).unsqueeze(0))
        goal, _ = vae.encode(torch.tensor(goal).unsqueeze(0))
        action = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
    trajectory = []
    for t in count():
        with torch.no_grad():
            # Controller
            concat_state = torch.cat((latent_state, predictive_hidden_state, goal), dim=1)
            action = controller.select_action_epsilon_greedy(concat_state)
            #print("Selected action: ", action)
    
            next_state, reward, terminated, truncated, _ = grid_worlds[WORLD_INDEX].step(torch.argmax(action, dim=1).item())

            # World Model
            predictive_hidden_state, pi, sigma, mu  = world_model(action.unsqueeze(0), latent_state.unsqueeze(0)) # Add sequence length dim with unsqueeze
    
            # VAE
            next_latent_state, _ = vae.encode(torch.tensor(next_state["agent"]).unsqueeze(0))

        trajectory.append(TransitionGCSL(state["agent"], latent_state, predictive_hidden_state, next_latent_state, action, goal))

        # Apply mapping to latent space
        #y, _ = sample_mdn_mix(pi, sigma, mu)
        _, y = sample_mdn(pi, sigma, mu)
        breakpoint()

        prediction_error = experience_memory.criterion(y, next_latent_state)
        experience_memory.append(state["agent"], latent_state, predictive_hidden_state, next_latent_state, prediction_error)

        state = next_state
        latent_state = next_latent_state

        if terminated or truncated:
            episode_durations.append(t + 1)
            
            if terminated:
                print("Reached Goal!")
            else:
                print("Failed")
            plot_training(episode_durations, vae_loss, wm_loss, controller_loss)
            break

    # sample new goal
    goal = experience_memory.sample().state

    ### TRAINING
    if WM_TRAIN:
        # Train WorldModelRNN
        print("Training World Model...")
        world_model.reset_hidden_state()
        actions = []
        latent_states = []
        next_latent_states = []
        for transition in trajectory:
            actions.append(transition.action)
            latent_states.append(transition.latent_state)
            next_latent_states.append(transition.next_latent_state)
        actions = torch.stack(actions, dim=0)
        latent_states = torch.stack(latent_states, dim=0)
        next_latent_states = torch.stack(next_latent_states, dim=0)
    
        loss = world_model.train_sequence(actions, latent_states, next_latent_states)
        world_model.scheduler.step(loss)
        print(f"World model learning rate = {world_model.scheduler.get_last_lr()}")
        wm_loss.append(loss)

    if CONTROLLER_TRAIN:
        if i >= 700:
            # Train controller only if enough data is available
            replay_memory.insert_trajectory(trajectory, i-700)
            if len(replay_memory) < 20:
                continue
    
            print("Training Controller...")
            data_available = True
            while data_available:
                transition = replay_memory.sample()[0]
                if len(replay_memory) < 1:
                    data_available = False
    
                loss = controller.optimization_step(transition)
                controller_loss.append(loss)
            print(f"Controller learning rate = {controller.scheduler.get_last_lr()}")

torch.save(world_model.state_dict(), WM_DIR)
torch.save(controller.policy_net.state_dict(), CONTROLLER_DIR)
breakpoint()
