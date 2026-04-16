import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]

Shape = Tuple[int, ...]
ShapeDict = Dict[str, Shape]


# <------------------------------- Vector In and Out  ------------------------------>

class VectorEncoder(nn.Module):
    '''An MLP to encode vector states and actions into a (d_model,) embedding for the transformer.'''

    def __init__(self, input_dims: ShapeDict, d_model: int) -> None:
        super().__init__()

        # total input dimension is the sum of all state and action dimensions
        input_dim = sum(int(np.prod(shape, dtype=np.int64)) for shape in input_dims.values())

        # simple MLP to project concatenated state and action vector into (d_model,) embedding
        n_hidden = (input_dim + d_model) // 2
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, d_model)
        )

    def forward(self, input_dict: TensorDict) -> Tensor:

        # concatenate all state and action tensors into a single vector
        batch, seq_len = next(iter(input_dict.values())).shape[:2]
        x = [tensor.view(batch, seq_len, -1) for tensor in input_dict.values()]
        x = torch.cat(x, dim=-1)
        x = x.view(batch * seq_len, -1)

        # project concatenated vector into (d_model,) embedding
        x_emb = self.input_proj(x)
        x_emb = x_emb.view(batch, seq_len, -1)
        return x_emb


class VectorDecoder(nn.Module):
    '''An MLP to decode a (d_model,) embedding back into vector states for the transformer.'''

    condition_mode = 'last'

    def __init__(self, state_dims: ShapeDict, d_model: int) -> None:
        super().__init__()

        # total output dimension is the sum of all state dimensions
        self.state_dims = state_dims
        state_dim = sum(int(np.prod(shape, dtype=np.int64)) for shape in state_dims.values())

        # simple MLP to project (d_model,) embedding back into state dict
        n_hidden = (d_model + state_dim) // 2
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, state_dim),
        )
    
    def forward(self, x: Tensor) -> TensorDict:
        assert x.dim() == 2, "Decoder input should be (batch, d_model)."
        next_state = self.output_proj(x)

        # split output vector back into state dict
        output_dict = {}
        idx = 0
        for key, shape in self.state_dims.items():
            dim = int(np.prod(shape, dtype=np.int64))
            output_dict[key] = next_state[:, idx:idx+dim].view(x.size(0), *shape)
            idx += dim
        return output_dict


# <------------------------------- Image In and Out  ------------------------------>

class ImageEncoder(nn.Module):
    '''A CNN to encode images into a (d_model,) embedding for the transformer.'''

    def __init__(self, input_dims: ShapeDict, d_model: int, 
                 sizes: Tuple[int, ...]=(32, 64, 128)) -> None:
        super().__init__()

        if len(sizes) == 0:
            raise ValueError("ImageEncoder sizes must contain at least one channel width.")

        # for visual inputs, we expect a single key 'obs' in input_dims with shape (C, H, W)
        # we will use a CNN to project this image input into the (d_model,) embedding space
        input_dims = dict(input_dims)
        if 'obs' not in input_dims:
            raise ValueError("For visual inputs, input_dims should have key 'obs'.")
        obs_shape = input_dims.pop('obs')
        if len(obs_shape) != 3:
            raise ValueError("For visual inputs, input_dims['obs'] should be (C, H, W).")
        n_channels = obs_shape[0]

        # create conv layers
        conv_layers = []
        in_ch = n_channels
        for out_ch in sizes:
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            conv_layers.append(nn.GELU())
            in_ch = out_ch

        self.state_proj = nn.Sequential(
            *conv_layers,
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(sizes[-1] * 4 * 4, d_model),
        )

        # actions will be projected with an MLP and concatenated to the projected image embedding
        self.action_proj = VectorEncoder(input_dims, d_model)
    
    def forward(self, input_dict: TensorDict) -> Tensor:

        # project state through CNN
        input_dict = dict(input_dict)
        img = input_dict.pop('obs')
        batch, seq_len, c, h, w = img.shape
        img = img.view(batch * seq_len, c, h, w)
        state_emb = self.state_proj(img).view(batch, seq_len, -1)

        # project action through MLP
        action_emb = self.action_proj(input_dict)   # (batch, seq_len, d_model)

        # combine state and action projections in embedding space
        return state_emb + action_emb


class ImageDecoder(nn.Module):
    '''A CNN to decode a (d_model,) embedding back into images for the transformer output.'''

    condition_mode = 'last'

    def __init__(self, image_dims: Tuple[int, int, int], d_model: int,
                 sizes: Tuple[int, ...]=(128, 64, 32), min_value: float=1e-4) -> None:
        super().__init__()
        self.min_value = min_value

        if len(sizes) == 0:
            raise ValueError("ImageDecoder sizes must contain at least one channel width.")
        
        c, h, w = image_dims

        # create conv layers
        conv_layers = []
        in_ch = sizes[0]
        for out_ch in sizes[1:-1]:
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.GELU())
            conv_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            in_ch = out_ch
        out_ch = sizes[-1]
        conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        conv_layers.append(nn.GELU())
        conv_layers.append(nn.Upsample(size=(h, w), mode='bilinear', align_corners=False))
        conv_layers.append(nn.Conv2d(out_ch, c, kernel_size=3, stride=1, padding=1))

        n_hidden = (d_model + sizes[0] * 4 * 4) // 2
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, sizes[0] * 4 * 4),
            nn.GELU(),
            nn.Unflatten(1, (sizes[0], 4, 4)),
            *conv_layers,
            nn.Sigmoid(),
        )
    
    def forward(self, x: Tensor) -> TensorDict:
        assert x.dim() == 2, "Decoder input should be (batch, d_model)."
        img = self.output_proj(x)
        img = img.clamp(self.min_value, 1.0 - self.min_value)
        next_state = {'obs': img}
        return next_state
