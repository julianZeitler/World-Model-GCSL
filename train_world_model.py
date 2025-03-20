import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from itertools import count

from WorldModel import MDNRNN
from VAE import VAE
from worlds import small_worlds
from Environment import GridWorld
from utils import plot_loss

from params import OBS_SIZE, LATENT_DIM, ACTION_SIZE, HIDDEN_DIM, WORLD_INDEX, BATCH_SIZE

VAE_DIR = "models/vae_small_latent_12.pth"
WM_DIR = "models/world_model_01.pth"
DATASET_PATH = "wm_data.pt"
PRELOAD_WM = False
NUM_TRAJECTORIES = 10000
EARLY_STOPPING_PATIENCE = 10

class TrajectoryDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Args:
            dataset_path (str): Path to the saved dataset file.
        """
        self.data = torch.load(dataset_path)  # Load dataset from disk

    def __len__(self):
        return len(self.data)  # Number of trajectories

    def __getitem__(self, idx):
        """
        Returns:
            actions (Tensor): Tensor of shape (max_seq_len, ACTION_SIZE)
            latent_states (Tensor): Tensor of shape (max_seq_len, LATENT_DIM)
            next_latent_states (Tensor): Tensor of shape (max_seq_len, LATENT_DIM)
            mask (Tensor): Tensor of shape (max_seq_len,)
        """
        return self.data[idx]  # Each entry is already a tuple of tensors

def custom_collate(batch):
    actions, latent_states, next_latent_states, masks = zip(*batch)
    
    actions = torch.stack(actions)  # (batch, seq_len, action_dim)
    latent_states = torch.stack(latent_states)
    next_latent_states = torch.stack(next_latent_states)
    masks = torch.stack(masks)

    # Transpose to (seq_len, batch, feature_dim)
    return actions.permute(1, 0, 2), latent_states.permute(1, 0, 2), next_latent_states.permute(1, 0, 2), masks.permute(1, 0)

def generate_dataset(vae, grid_worlds, world_index=WORLD_INDEX, num_trajectories=NUM_TRAJECTORIES):
    """
    Generates a dataset of agent trajectories in the grid world and saves it to a file.

    Args:
        grid_worlds: The grid world environments.
        WORLD_INDEX: The index of the current grid world.
        vae: Variational autoencoder for state encoding.
        controller: Policy controller for action selection.
        world_model: The learned world model.
        experience_memory: Storage for experience replay.
        BATCH_SIZE: Number of trajectories per batch.

    Returns:
        None (Saves dataset to disk)
    """
    print(f"Generating {num_trajectories} trajectories")
    goal = np.full((5, 5), 0, dtype=np.float32)
    dataset = []  # Store all trajectories

    for idx in range(num_trajectories):
        if idx % 100 == 0:
            print(f"Generated {idx} trajectories")
        # Reset environment and hidden state
        state, _ = grid_worlds[world_index].reset(goal)

        with torch.no_grad():
            latent_state, _ = vae.encode(torch.tensor(state["agent"]).unsqueeze(0))

        # Pad tensors
        max_seq_len = 99
        actions = torch.zeros(max_seq_len, ACTION_SIZE)
        latent_states = torch.zeros(max_seq_len, LATENT_DIM)
        next_latent_states = torch.zeros(max_seq_len, LATENT_DIM)
        mask = torch.zeros(max_seq_len)  # Mask for variable-length sequences

        terminated, truncated = False, False
        for t in count():
            with torch.no_grad():
                action_index = torch.randint(0, 4, (1,))  # Random integer between 0 and 3
                action = torch.zeros(4, dtype=torch.float32)  
                action[action_index] = 1
                # Step in environment
                next_state, _, terminated, truncated, _ = grid_worlds[world_index].step(action_index.item())
                # Encode next state
                next_latent_state, _ = vae.encode(torch.tensor(next_state["agent"]).unsqueeze(0))

            actions[t] = action
            latent_states[t] = latent_state.squeeze(0)
            next_latent_states[t] = next_latent_state.squeeze(0)
            mask[t] = 1
            breakpoint()

            # Update state
            state = next_state
            latent_state = next_latent_state

            if terminated or truncated:
                break

        dataset.append((actions, latent_states, next_latent_states, mask))

        # Save dataset periodically
        if idx % 1000 == 0:
            torch.save(dataset, DATASET_PATH)
            print(f"Checkpoint: Saved {len(dataset)} trajectories.")

    torch.save(dataset, DATASET_PATH)
    print("Dataset generation completed and saved.")

def train_world_model(world_model, dataloader, epochs=100, early_stopping_patience=100, checkpoint_path=WM_DIR):
    """
    Trains the world model using the pre-generated dataset with early stopping and checkpoints.

    Args:
        world_model: The learned world model.
        dataset_path (str): Path to the saved dataset.
        batch_size (int): Number of samples per batch.
        early_stopping_patience (int): Stop training if no improvement in 'N' epochs.
        checkpoint_path (str): Where to save the best model.

    Returns:
        None
    """
    breakpoint()
    best_loss = float("inf")
    early_stopping_counter = 0
    losses = []

    for epoch in range(epochs):
        print(f"Training Epoch: {epoch}")
        total_loss = 0
        for actions, latent_states, next_latent_states, mask in dataloader:
            # Train world model
            loss = world_model.train_sequence(actions, latent_states, next_latent_states, mask)
            world_model.scheduler.step(loss)

            print(f"Loss: {loss:.5f}, Learning rate: {world_model.scheduler.get_last_lr()[0]}")
            losses.append(loss)
            plot_loss(losses)
            total_loss += loss.item()

            # Early Stopping & Checkpoints
            if loss < best_loss:
                best_loss = loss
                early_stopping_counter = 0
                print("Saving checkpoint...")
                torch.save(world_model.state_dict(), checkpoint_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered! Training stopped.")
                    break
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-data", action="store_true", help="Generate dataset before training")
    args = parser.parse_args()

    grid_worlds = [GridWorld(width=world.shape[0], height=world.shape[1], horizon=100, obstacles=np.argwhere(world==1)) for world in small_worlds]
    
    vae = VAE(input_dim=OBS_SIZE, latent_dim=LATENT_DIM, hidden_dim=256)
    world_model = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_SIZE, hidden_dim=HIDDEN_DIM, num_layers=5, num_gaussians=10, lr=1e-3)
    
    # Load models
    vae.load_state_dict(torch.load(VAE_DIR, weights_only=True))
    print("Loaded VAE")
    if PRELOAD_WM:
        world_model.load_state_dict(torch.load(WM_DIR, weights_only=True))
        print("Loaded world model")

    if args.generate_data:
        generate_dataset(vae, grid_worlds)

    dataset = torch.load(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate)

    train_world_model(world_model, dataloader)


