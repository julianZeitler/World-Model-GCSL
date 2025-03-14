import torch
import matplotlib.pylab as plt
import gradio as gr
import numpy as np

from VAE import VAE

VAE_DIR = "models/vae_small_02.pth"
OBS_SIZE = 5
LATENT_DIM = 4

vae = VAE(input_dim=OBS_SIZE, latent_dim=LATENT_DIM, hidden_dim=256)

torch.no_grad()
vae.load_state_dict(torch.load(VAE_DIR, weights_only=True))

def generate_observation(*latent_values):
    """Decode latent variables into a grid observation."""
    z = torch.tensor(latent_values, dtype=torch.float32).unsqueeze(0)
    obs = vae.decode(z).squeeze().detach().numpy()
    
    # Normalize to [0, 255] for visualization
    obs = (obs - obs.min()) / (obs.max() - obs.min())
    obs = obs.astype(np.float32)
    
    obs = np.kron(obs, np.ones((50, 50)))
    return obs

# Create Gradio interface
sliders = [gr.Slider(-3, 3, step=0.01, value=0, label=f'Latent {i+1}') for i in range(LATENT_DIM)]
interface = gr.Interface(fn=generate_observation, inputs=sliders, outputs=gr.Image(type="numpy"))


if __name__ == "__main__":
    print("launching")
    interface.launch()

