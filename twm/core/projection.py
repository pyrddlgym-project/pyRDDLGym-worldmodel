import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

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

    condition_mode = 'last'

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

    def __init__(self, n_channels: int, action_dim: int, d_model: int,
                 sizes=(32, 64, 128)) -> None:
        super().__init__()

        if len(sizes) == 0:
            raise ValueError("ImageEncoder sizes must contain at least one channel width.")

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

    condition_mode = 'last'

    def __init__(self, image_dims: Tuple[int, int, int], d_model: int,
                 sizes=(256, 128, 64, 32)) -> None:
        super().__init__()

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

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, sizes[0] * 4 * 4),
            nn.GELU(),
            nn.Unflatten(1, (sizes[0], 4, 4)),
            *conv_layers,
            nn.Sigmoid(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.output_proj(x).clamp(1e-8, 1.0 - 1e-8)


# <------------------------------- Diffusion Out  ------------------------------>

class SinusoidalPosEmb(nn.Module):
    '''Generates sinusoidal positional embeddings for a given input tensor of timesteps.'''

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb_scale = np.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return emb


class ResidualBlock2D(nn.Module):
    '''A residual block with two 3x3 convolutions and FiLM conditioning on a (d_model,) embedding.'''

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int=8) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class CrossAttention2D(nn.Module):
    '''Cross-attention between UNet spatial features (B, C, H, W) and a context sequence (B, T, D).'''

    def __init__(self, query_dim: int, context_dim: int, heads: int=4, 
                 context_dropout: float=0.1) -> None:
        super().__init__()
        
        assert query_dim % heads == 0
        self.heads = heads
        self.head_dim = query_dim // heads
        self.norm = nn.GroupNorm(8, query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_dropout = nn.Dropout(context_dropout)
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

        # start as FiLM-only UNet and learn attention contribution gradually
        self.gate = nn.Parameter(torch.zeros(1))

    def process_for_attention(self, x: Tensor) -> Tensor:
        return x.unflatten(-1, (self.heads, self.head_dim)).permute(0, 2, 1, 3)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        context = self.context_dropout(self.context_norm(context))

        # compute queries from spatial features and keys/values from context sequence
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)
        q = self.to_q(h)
        k = self.to_k(context)
        v = self.to_v(context)

        # reshape for multi-head attention and compute scaled dot-product attention
        q = self.process_for_attention(q)
        k = self.process_for_attention(k)
        v = self.process_for_attention(v)

        # compute attention and reshape back to (B, C, H, W)
        out = F.scaled_dot_product_attention(q, k, v)   # (B, heads, HW, head_dim)
        out = out.permute(0, 2, 1, 3).flatten(-2)       # (B, HW, C)
        out = self.to_out(out).permute(0, 2, 1).view(B, C, H, W)

        # combine attention output with original features using a learnable gate
        return x + self.gate * out


class ConditionalUNet2D(nn.Module):
    '''A UNet for 2D image states, conditioned on a (d_model,) embedding.'''

    def __init__(self, in_channels: int, cond_dim: int, base_dim: int=32,
                 dim_mults: Tuple[int, ...]=(1, 2, 4), 
                 use_cross_attention: bool=True, nheads: int=4) -> None:
        super().__init__()

        # compute the embedding dimension for the conditioning MLPs
        emb_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # compute the channel dimensions for each level of the UNet
        dims = [base_dim * m for m in dim_mults]
        self.in_conv = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1)

        # create the downsampling and upsampling blocks of the UNet
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_blocks.append(
                nn.ModuleList([
                    ResidualBlock2D(dims[i], dims[i], emb_dim),
                    ResidualBlock2D(dims[i], dims[i], emb_dim),
                ])
            )
            self.downsample.append(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=4, stride=2, padding=1))

        # bottleneck residual blocks
        mid_dim = dims[-1]
        self.mid1 = ResidualBlock2D(mid_dim, mid_dim, emb_dim)
        self.cross_attn_mid = CrossAttention2D(mid_dim, cond_dim, heads=nheads) \
            if use_cross_attention else None
        self.mid2 = ResidualBlock2D(mid_dim, mid_dim, emb_dim)

        # create the upsampling blocks of the UNet
        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        rev_dims = list(reversed(dims))
        for i in range(len(rev_dims) - 1):
            in_ch = rev_dims[i] + rev_dims[i + 1]
            out_ch = rev_dims[i + 1]
            self.up_blocks.append(
                nn.ModuleList([
                    ResidualBlock2D(in_ch, out_ch, emb_dim),
                    ResidualBlock2D(out_ch, out_ch, emb_dim),
                ])
            )
            self.upsample.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        # final normalization and output convolution
        self.out_norm = nn.GroupNorm(8, dims[0])
        self.out_conv = nn.Conv2d(dims[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        
        # sequence conditioning uses the latest token for FiLM and full history for cross-attention
        if cond.ndim == 3:
            cond_last = cond[:, -1]
            context = cond
        else:
            cond_last = cond
            context = None

        # compute the conditioning embedding
        emb = self.time_mlp(t) + self.cond_mlp(cond_last)

        # forward pass through the UNet
        h = self.in_conv(x)
        skips = []
        for blocks, down in zip(self.down_blocks, self.downsample):
            h = blocks[0](h, emb)
            h = blocks[1](h, emb)
            skips.append(h)
            h = down(h)

        # bottleneck
        h = self.mid1(h, emb)
        if self.cross_attn_mid is not None and context is not None:
            h = self.cross_attn_mid(h, context)
        h = self.mid2(h, emb)

        # forward pass through upsampling path with skip connections
        for blocks, up in zip(self.up_blocks, self.upsample):
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            h = blocks[0](h, emb)
            h = blocks[1](h, emb)
            h = up(h)

        # final normalization and output convolution
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


class DiffusionDecoder(nn.Module):
    '''UNet-based conditional diffusion decoder for image states.'''

    condition_mode = 'last'

    def __init__(self, state_dim: Union[int, Tuple[int, ...]], d_model: int,
                 n_diffusion_steps: int=128, hidden_dim: int=32,
                 clamp_output: bool=True, use_cross_attention: bool=True) -> None:
        super().__init__()

        if not isinstance(state_dim, tuple):
            raise ValueError('DiffusionDecoder expects state_dim as tuple.')
        if len(state_dim) != 3:
            raise ValueError('DiffusionDecoder requires state_dim to be (C, H, W).')

        self.state_shape = tuple(state_dim)
        self.n_diffusion_steps = n_diffusion_steps
        self.clamp_output = clamp_output
        self.condition_mode = 'sequence' if use_cross_attention else 'last'

        self.denoiser = ConditionalUNet2D(
            in_channels=self.state_shape[0], cond_dim=d_model, base_dim=hidden_dim, 
            use_cross_attention=use_cross_attention)

        betas = self.cosine_beta_schedule(n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars))
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1.0 - alpha_bars))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float=0.008) -> Tensor:
        '''Generates a cosine schedule of betas for the diffusion process.'''
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
        return torch.tensor(betas_clipped, dtype=torch.float32)

    def extract(self, coeff: Tensor, t: Tensor, ndim: int) -> Tensor:
        '''Extracts the appropriate coefficients for a batch of timesteps and reshapes for broadcasting.'''
        return coeff.gather(0, t).view(t.size(0), *([1] * (ndim - 1)))

    def q_sample(self, x0: Tensor, t: Tensor, noise: Optional[Tensor]=None) -> Tensor:
        '''Adds noise to the input image x0 according to the diffusion process at timestep t.'''
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.extract(self.sqrt_alpha_bars, t, x0.ndim)
        sqrt_omb = self.extract(self.sqrt_one_minus_alpha_bars, t, x0.ndim)
        return sqrt_ab * x0 + sqrt_omb * noise

    def loss(self, cond: Tensor, target: Tensor) -> Tensor:
        '''Computes the MSE loss between the predicted noise and the true noise for a batch of 
        target images and conditions.'''
        batch = target.size(0)
        t = torch.randint(0, self.n_diffusion_steps, (batch,), device=target.device)

        noise = torch.randn_like(target)
        x_t = self.q_sample(target, t, noise)
        eps_pred = self.denoiser(x_t, t, cond)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def forward(self, cond: Tensor) -> Tensor:
        '''Generates an image by iteratively denoising from pure noise, conditioned on the input 
        embedding.'''
        # sample pure noise x_T
        batch = cond.size(0)
        x_t = torch.randn(batch, *self.state_shape, device=cond.device)
        
        # denoise until we reach x_0
        for i in reversed(range(self.n_diffusion_steps)):

            # sample noise prediction for current timestep
            t = torch.full((batch,), i, device=cond.device, dtype=torch.long)
            eps_pred = self.denoiser(x_t, t, cond)

            # compute the mean prediction for the denoised sample x_t
            beta_t = self.extract(self.betas, t, x_t.ndim)
            sqrt_omb_t = self.extract(self.sqrt_one_minus_alpha_bars, t, x_t.ndim)
            sqrt_inv_a_t = self.extract(self.sqrt_recip_alphas, t, x_t.ndim)
            x_t = sqrt_inv_a_t * (x_t - (beta_t / sqrt_omb_t) * eps_pred)

            # add noise for all but the final step
            if i > 0:
                x_t = x_t + torch.sqrt(beta_t) * torch.randn_like(x_t)

        if self.clamp_output:
            x_t = x_t.clamp(0.0, 1.0)
        return x_t

