import torch
import torch.nn as nn
from typing import Tuple

Tensor = torch.Tensor


# <------------------------------- Vector In and Out  ------------------------------>

class VectorEncoder(nn.Module):
    '''A simple MLP to encode vector states and actions into a (d_model,) embedding 
    for the transformer.'''

    def __init__(self, state_dim: int, action_dim: int, d_model: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(state_dim + action_dim, d_model)

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        x = torch.cat([states, actions], dim=-1)
        batch, seq_len = x.shape[:2]
        x = x.view(batch * seq_len, -1)
        y = self.input_proj(x).view(batch, seq_len, -1)
        return y 


class VectorDecoder(nn.Module):
    '''A simple MLP to decode (d_model,) embeddings back into vector states for the transformer.'''

    def __init__(self, state_dim: int, d_model: int) -> None:
        super().__init__()

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, state_dim),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.output_proj(x)


# <------------------------------- Image In and Out  ------------------------------>

class ImageEncoder(nn.Module):
    '''A small CNN to encode images into a (d_model,) embedding for the transformer.'''

    def __init__(self, n_channels: int, action_dim: int, d_model: int) -> None:
        super().__init__()

        self.state_proj = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, d_model),
        )
        self.action_proj = nn.Linear(action_dim, d_model)
    
    def forward(self, states: Tensor, actions: Tensor) -> Tensor:

        # project state through cnn
        batch, seq_len, c, h, w = states.shape
        states = states.view(batch * seq_len, c, h, w)
        state_proj = self.state_proj(states).view(batch, seq_len, -1)

        # project action through MLP
        actions = actions.view(batch * seq_len, -1)
        action_proj = self.action_proj(actions).view(batch, seq_len, -1)

        # combine state and action projections in embedding space
        return state_proj + action_proj


class ImageDecoder(nn.Module):
    '''A small CNN to decode (d_model,) embeddings back into images for the 
    transformer output.'''

    def __init__(self, image_dims: Tuple[int, int, int], d_model: int) -> None:
        super().__init__()

        c, h, w = image_dims
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 128 * 8 * 8),
            nn.GELU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(size=(h, w), mode='bilinear', align_corners=False),
            nn.Conv2d(32, c, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.output_proj(x).clamp(1e-8, 1.0 - 1e-8)
