from quorus.logging.custom_slog import print_cust

"""### Definition of PCADiscriminator"""

import torch.nn as nn

import torch.nn.functional as F

# Maybe change model architecture??? (if it's not working well)
class PCADiscriminator(nn.Module):
    def __init__(self, input_size, scale_factor):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size * scale_factor),
            nn.ReLU(),
            nn.Linear(input_size * scale_factor, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )
        self.input_size = input_size
        self.scale_factor = scale_factor

    def __str__(self):
        return f"""
        PCADiscriminator
        self.net: {self.net}
        self.input_size: {self.input_size}
        self.state_dict(): {self.state_dict()}
        self.scale_factor: {self.scale_factor}
        """
    __repr__ = __str__

    def forward(self, x, alpha=1.0, generator_weights=None):
        """
        If x.shape[-1] < self.target_dim, pad zeros on the *right* so that the
        last dimension equals self.target_dim, then pass through self.net.
        """
        cur_dim = x.size(-1)             # length of the last (fast-changing) axis
        if cur_dim < self.input_size:
            pad_right = self.input_size - cur_dim
            # (left, right) for 1-D padding of the last dimension
            x = F.pad(x, (0, pad_right), mode="constant", value=0.0)
            print_cust(f"PCADiscriminator, forward, x.shape: {x.shape}")
            print_cust(f"PCADiscriminator, forward, after padding, x: {x}")
        return self.net(x)

    def get_data_components(self):
      return (self.input_size, self.scale_factor)