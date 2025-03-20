from Controller import Controller
from Environment import GridWorld

from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from Memory import TransitionGCSL, ReplayMemory, ExperienceMemory
from VAE import VAE, vae_loss_fn
from WorldModel import MDNRNN, sample_mdn
from worlds import small_worlds
from utils import Plotter
from params import *

plotter = Plotter()

grid_worlds = [GridWorld(width=world.shape[0], height=world.shape[1], horizon=100, obstacles=np.argwhere(world==1)) for world in small_worlds]

controller = Controller(z_dim=LATENT_DIM, h_dim=HIDDEN_DIM, a_dim=ACTION_SIZE, env_action_space=grid_worlds[WORLD_INDEX].action_space)
vae = VAE(input_dim=OBS_SIZE, latent_dim=LATENT_DIM, hidden_dim=256)
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-5)
vae_scheduler = ReduceLROnPlateau(vae_optimizer, mode='min', factor=0.1, patience=10000)
world_model = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_SIZE, hidden_dim=HIDDEN_DIM, num_layers=1, num_gaussians=1, lr=1e-3)

# init world model weights
for name, param in world_model.named_parameters():
    if "weight_hh" in name:
        nn.init.xavier_uniform_(param)
    elif "bias" in name:
        nn.init.zeros_(param)

replay_memory = ReplayMemory(10000)
experience_memory = ExperienceMemory(10000)

BATCH_SIZE = 32
# Train VAE
if not VAE_TRAIN:
    vae.load_state_dict(torch.load(VAE_DIR, weights_only=True))
    print("Loaded VAE")
else:
    # Collect all agent observations from grid worlds
    observations = [world.get_all_obs() for world in grid_worlds]
    observations = np.array([obs["agent"] for world_obs in observations for obs in world_obs])

    # Convert observations to a PyTorch tensor
    observations_tensor = torch.tensor(observations)
    
    # Create a DataLoader for batching
    dataset = TensorDataset(observations_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Training VAE...")
    for epoch in range(10000):  # Iterate over epochs
        epoch_loss = 0
        for batch in dataloader:
            batch_obs = batch[0]  # Get the batch
            
            vae_optimizer.zero_grad()
            obs_reconstructed, _, mean, log_var = vae(batch_obs)
            loss = vae_loss_fn(batch_obs, obs_reconstructed, mean, log_var, beta=0.0001, target_std=0.1)
            
            loss.backward()
            vae_optimizer.step()
            vae_scheduler.step(loss)
            epoch_loss += loss.item()
            plotter.vae_loss.append(loss.item())

        # Log every 100 iterations
        if epoch % 10 == 0:
            plotter.plot_training()
    torch.save(vae.state_dict(), VAE_DIR)

def show_hidden(state):
    plt.figure()
    plt.imshow(F.sigmoid(vae.decode(state)).squeeze().detach().numpy())
    plt.show(block=False)

if not WM_TRAIN:
    world_model.load_state_dict(torch.load(WM_DIR, weights_only=True))
    print("Loaded world model")

if not CONTROLLER_TRAIN:
    controller.policy_net.load_state_dict(torch.load(CONTROLLER_DIR, weights_only=True))
    print("Loaded controller")

goal = np.full((5, 5), 0, dtype=np.float32)
batch_trajectories = []
wm_best_loss = float("inf")
for i in range(100000):
    print("Epoch: " + str(i))
    state, info = grid_worlds[WORLD_INDEX].reset(goal)
    predictive_hidden_state, _, _, _ = world_model.reset_hidden_state()
    predictive_hidden_state = predictive_hidden_state.squeeze(0)

    with torch.no_grad():
        latent_state, _, _ = vae.encode(torch.tensor(state["agent"]).unsqueeze(0))
        goal, _, _ = vae.encode(torch.tensor(goal).unsqueeze(0))
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
            predictive_hidden_state, pi, sigma, mu = world_model(action, latent_state)
    
            # VAE
            next_latent_state, _, _ = vae.encode(torch.tensor(next_state["agent"]).unsqueeze(0))

        trajectory.append(TransitionGCSL(state["agent"], latent_state, predictive_hidden_state, next_latent_state, action, goal))

        # Apply mapping to latent space
        y, _ = sample_mdn(pi, sigma, mu)
        breakpoint()

        prediction_error = experience_memory.criterion(y, next_latent_state)
        experience_memory.append(state["agent"], latent_state, predictive_hidden_state, next_latent_state, prediction_error)

        state = next_state
        latent_state = next_latent_state

        if terminated or truncated:
            plotter.episode_durations.append(t + 1)
            
            if terminated:
                print("Reached Goal!")
            else:
                print("Failed")
            plotter.plot_training()
            break

    batch_trajectories.append(trajectory)
    goal = experience_memory.sample().state

    ### TRAINING
    if len(batch_trajectories) >= BATCH_SIZE and WM_TRAIN:
        print("Training World Model on batch...")
        
        # Prepare batch tensors
        max_seq_len = max(len(traj) for traj in batch_trajectories)
        
        # Initialize padded tensors
        actions = torch.zeros(max_seq_len, BATCH_SIZE, ACTION_SIZE)
        latent_states = torch.zeros(max_seq_len, BATCH_SIZE, LATENT_DIM)
        next_latent_states = torch.zeros(max_seq_len, BATCH_SIZE, LATENT_DIM)
        mask = torch.zeros(max_seq_len, BATCH_SIZE)  # Mask for variable-length sequences

        # Fill batch tensors
        for batch_idx, traj in enumerate(batch_trajectories):
            seq_len = len(traj)
            for t in range(seq_len):
                actions[t, batch_idx] = traj[t].action
                latent_states[t, batch_idx] = traj[t].latent_state
                next_latent_states[t, batch_idx] = traj[t].next_latent_state
                mask[t, batch_idx] = 1  # Mark valid timesteps

        # Train the RNN
        loss = world_model.train_sequence(actions, latent_states, next_latent_states, mask)
        world_model.scheduler.step(loss)
        print(f"World model learning rate = {world_model.scheduler.get_last_lr()}")
        plotter.wm_loss.append(loss)

        batch_trajectories = []
        # Checkpoints
        if loss < wm_best_loss:
            wm_best_loss = loss
            print("Saving checkpoint...")
            torch.save(world_model.state_dict(), WM_DIR)

    if CONTROLLER_TRAIN:
        if i >= 100000:
            # Train controller only if enough data is available
            replay_memory.insert_trajectory(trajectory, i-100000)
            if len(replay_memory) < 20:
                continue
    
            print("Training Controller...")
            data_available = True
            while data_available:
                transition = replay_memory.sample()[0]
                if len(replay_memory) < 1:
                    data_available = False
    
                loss = controller.optimization_step(transition)
                plotter.controller_loss.append(loss)
            print(f"Controller learning rate = {controller.scheduler.get_last_lr()}")

torch.save(world_model.state_dict(), WM_DIR)
torch.save(controller.policy_net.state_dict(), CONTROLLER_DIR)
breakpoint()
