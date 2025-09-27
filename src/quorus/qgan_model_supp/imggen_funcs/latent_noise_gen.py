import torch
from pennylane import numpy as np

"""### Latent Noise Generation Function"""

def generate_latent_noise(latent_dim, n_qubits, device, min=0.0, max=(np.pi / 2), n_qubits_small=0):
    return (torch.rand(latent_dim, n_qubits, device=device) * (max - min)) + min