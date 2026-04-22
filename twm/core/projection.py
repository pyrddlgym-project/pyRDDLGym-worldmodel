import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]
Shape = Tuple[int, ...]


# <------------------------------- Vector In and Out  ------------------------------>

class VectorEncoder(nn.Module):
    '''An MLP to encode vector states and actions into a (d_model,) embedding.'''

    def __init__(self, input_shape: Shape, d_model: int) -> None:
        super().__init__()

        # simple MLP to project state and action vector into (d_model,) embedding
        in_dim = int(np.prod(input_shape, dtype=np.int64))
        n_hidden = (in_dim + d_model) // 2
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len = x.shape[:2]
        x = x.view(batch * seq_len, -1)
        enc = self.encoder(x)
        enc = enc.view(batch, seq_len, -1)
        return enc


class VectorDecoder(nn.Module):
    '''An MLP to decode a (d_model,) embedding into vector states.'''
    condition_mode = 'last'

    def __init__(self, output_shape: Shape, d_model: int) -> None:
        super().__init__()
        self.output_shape = output_shape
        
        # simple MLP to project (d_model,) embedding back into state dict
        out_dim = int(np.prod(output_shape, dtype=np.int64))
        n_hidden = (d_model + out_dim) // 2
        self.decoder = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, out_dim),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 2, 'Decoder input should be (batch, d_model).'
        dec = self.decoder(x)
        dec = dec.view(-1, *self.output_shape)
        return dec


# <------------------------------- Image In and Out  ------------------------------>

class ImageEncoder(nn.Module):
    '''A CNN to encode images into a (d_model,) embedding.'''

    def __init__(self, image_shape: Shape, d_model: int, 
                 sizes: Tuple[int, ...]=(32, 64, 128)) -> None:
        super().__init__()

        if len(sizes) == 0:
            raise ValueError('sizes must not be empty.')
        if len(image_shape) != 3:
            raise ValueError('Image_shape should be (C, H, W).')
        
        # create conv layers
        conv_layers = []
        in_ch = image_shape[0]
        for out_ch in sizes:
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            conv_layers.append(nn.GELU())
            in_ch = out_ch

        self.encoder = nn.Sequential(
            *conv_layers,
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(sizes[-1] * 4 * 4, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, c, h, w = x.shape
        x = x.view(batch * seq_len, c, h, w)
        enc = self.encoder(x)
        enc = enc.view(batch, seq_len, -1)
        return enc


class ImageDecoder(nn.Module):
    '''A CNN to decode a (d_model,) embedding into images.'''
    condition_mode = 'last'

    def __init__(self, image_shape: Shape, d_model: int,
                 sizes: Tuple[int, ...]=(256, 128, 64, 32), 
                 min_value: float=1e-6) -> None:
        super().__init__()
        self.min_value = min_value

        if len(sizes) == 0:
            raise ValueError('sizes must not be empty.')
        if len(image_shape) != 3:
            raise ValueError('Image_shape should be (C, H, W).')
        
        # create conv layers
        c, h, w = image_shape
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

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, sizes[0] * 4 * 4),
            nn.GELU(),
            nn.Unflatten(1, (sizes[0], 4, 4)),
            *conv_layers,
            nn.Sigmoid(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 2, 'Decoder input should be (batch, d_model).'
        dec = self.decoder(x)
        dec = dec.clamp(self.min_value, 1.0 - self.min_value)
        return dec
    