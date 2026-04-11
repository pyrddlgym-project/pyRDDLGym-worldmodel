import os
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Dict, Tuple, Union

from twm.core.encoding import SinePositionalEncoding, RotaryTransformerEncoderLayer
from twm.core.projection import VectorEncoder, VectorDecoder, ImageEncoder, ImageDecoder

Tensor = torch.Tensor

PARENT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MODEL_PATH = os.path.join(PARENT_PATH, "models")


class EMA:
    '''Maintains an exponential moving average of model weights for evaluation stability.'''

    def __init__(self, model: nn.Module, decay: float=0.999) -> None:
        self.decay = decay
        self.weights = {k: v.clone().float().cpu() for k, v in model.state_dict().items()}
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.weights[k].mul_(self.decay).add_(v.float().cpu(), alpha=1. - self.decay)

    @property
    def state_dict(self) -> Dict[str, Tensor]:
        return {k: v.to(self.device) for k, v in self.weights.items()}


class WorldModel(nn.Module):
    '''A transformer-based world model that predicts the next state given a sequence of 
    past states and actions.'''

    def __init__(self, state_dim: Union[int, Tuple[int, ...]], action_dim: int, 
                 seq_len: int, visual: bool, d_model: int=64, 
                 nhead: int=4, num_layers: int=4, dim_feedforward: int=256, 
                 dropout: float=0.1, norm_first: bool=True, 
                 use_absolute_pe: bool=True, use_rope: bool=True) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.visual = visual
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.norm_first = norm_first

        self.use_absolute_pe = use_absolute_pe
        self.use_rope = use_rope

        # state and action -> (d_model,) embedding
        if self.visual:
            assert isinstance(state_dim, tuple), "For visual inputs, state_dim should be a tuple."
            assert len(state_dim) == 3, "For visual inputs, state_dim should be (C, H, W)."
            n_channels = state_dim[0]
            self.input_proj = ImageEncoder(n_channels, action_dim, d_model)
        else:
            assert isinstance(state_dim, int), "For vector inputs, state_dim should be an int."
            self.input_proj = VectorEncoder(state_dim, action_dim, d_model)
        
        # Keep input feature scaling stable regardless of transformer norm mode
        self.input_norm = nn.LayerNorm(d_model)

        # learnable special tokens in embedding space
        self.pad_token = nn.Parameter(torch.zeros(d_model))
        self.sos_token = nn.Parameter(torch.zeros(d_model))

        # positional encoding
        if use_absolute_pe:
            self.pe = SinePositionalEncoding(d_model, max_len=seq_len)
        else:
            self.pe = nn.Identity()
        self.embed_dropout = nn.Dropout(dropout)

        # transformer encoder
        if self.use_rope:
            encoder_layer = RotaryTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                max_len=seq_len,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
            )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # pre-norm stacks typically need a final normalization on the encoder output
        self.output_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        
        # (d_model,) embedding -> next state prediction
        if self.visual:
            self.output_proj = ImageDecoder(state_dim, d_model)
            self.loss_fn = nn.BCELoss()
        else:
            self.output_proj = VectorDecoder(state_dim, d_model)
            self.loss_fn = nn.HuberLoss()
        
        # placeholder normalizer buffers
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        if self.visual:
            self.register_buffer("obs_idx", None)
        else:
            self.register_buffer("obs_idx", torch.arange(state_dim, dtype=torch.long))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_dataset_stats(self, dataset) -> None:
        self.register_buffer("state_mean", dataset.state_mean.to(self.device))
        self.register_buffer("state_std", dataset.state_std.to(self.device))
        self.register_buffer("action_mean", dataset.action_mean.to(self.device))
        self.register_buffer("action_std", dataset.action_std.to(self.device))
        if dataset.obs_idx is None:
            obs_idx = None
        else:
            obs_idx = torch.tensor(dataset.obs_idx, dtype=torch.long, device=self.device)
        self.register_buffer("obs_idx", obs_idx)

    # <----------------------------------- prediction ----------------------------------->

    def forward(self, states: Tensor, actions: Tensor, pad_lens: Tensor) -> Tensor:
        
        # input processing
        states = self.normalize_states(states)
        actions = self.normalize_actions(actions)
        x = self.input_proj(states, actions)   # (batch, seq_len, d_model)
        x = self.input_norm(x)   
        x = x.clone()
        
        # replace padded positions with learnable PAD embedding
        batch, seq_len = actions.shape[:2]
        pad_mask = self.make_padding_mask(pad_lens, seq_len)  # (batch, seq_len)
        x[pad_mask] = self.pad_token
        
        # mark first timestep with a learnable SOS embedding
        x[:, 0] = x[:, 0] + self.sos_token

        # positional encoding and transformer
        x = self.pe(x)
        x = self.embed_dropout(x)
        mask = self.make_full_mask(pad_lens, seq_len, batch)
        x = self.transformer(x, mask=mask)

        # get last real output before padding
        x = self.output_norm(x)
        last_real_idx = (seq_len - pad_lens - 1).clamp(min=0)
        batch_idx = torch.arange(batch, device=self.device)
        x = x[batch_idx, last_real_idx]
        return self.output_proj(x)

    def make_padding_mask(self, pad_lens: Tensor, seq_len: int) -> Tensor:
        pos = torch.arange(seq_len, device=self.device)  # (seq_len,)
        return pos.unsqueeze(0) >= (seq_len - pad_lens).unsqueeze(1)  # (batch, seq_len)
    
    def make_full_mask(self, pad_lens: Tensor, seq_len: int, batch: int) -> Tensor:
        device = self.device

        # create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

        # add padding mask to causal mask
        pad_mask_ = self.make_padding_mask(pad_lens, seq_len)   # (batch, seq_len)
        padding_mask = torch.zeros(batch, 1, seq_len, device=device)
        padding_mask = padding_mask.masked_fill(pad_mask_.unsqueeze(1), float('-inf'))
        mask = causal_mask.unsqueeze(0) + padding_mask   # (batch, seq_len, seq_len)

        # ensure padded positions can attend to themselves to prevent NaNs in attention output
        eye = torch.eye(seq_len, dtype=torch.bool, device=device)
        mask = mask.masked_fill(pad_mask_.unsqueeze(-1) & eye, 0.0)
        mask = mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        mask = mask.reshape(batch * self.nhead, seq_len, seq_len)
        return mask

    def normalize_states(self, states: Tensor) -> Tensor:
        if self.visual:
            return states
        else:
            return (states - self.state_mean) / self.state_std
    
    def normalize_actions(self, actions: Tensor) -> Tensor:
        return (actions - self.action_mean) / self.action_std
    
    def unnormalize_states(self, states: Tensor) -> Tensor:
        if self.visual:
            return states
        else:
            return states * self.state_std + self.state_mean
    
    # <---------------------------- training and evaluation ----------------------------->

    def loss(self, states: Tensor, actions: Tensor, next_states: Tensor, 
             pad_lens: Tensor) -> Tensor:
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        pad_lens = pad_lens.to(self.device)
        
        targets = self.normalize_states(next_states)
        pred = self.forward(states, actions, pad_lens)
        return self.loss_fn(pred, targets)
    
    @torch.no_grad()
    def evaluate(self, data_loader) -> float:
        self.eval()
        
        if data_loader is None: 
            return float('nan')

        loss = 0.
        for states, actions, *_, next_states, pad_lens in tqdm(data_loader, desc="Evaluating"):
            loss += self.loss(states, actions, next_states, pad_lens).item()
        return loss / len(data_loader)

    def fit(self, train_data_loader, epochs: int, lr: float=1e-3, lr_decay: float=0.9, 
            test_data_loader=None, model_name: str='') -> None:
                
        self.set_dataset_stats(train_data_loader.dataset)

        optim = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, factor=lr_decay, patience=10, min_lr=1e-5)
        
        ema = EMA(self)

        for epoch in range(epochs):
            self.train()

            # training loop
            epoch_loss = 0.
            for states, actions, *_, next_states, pad_lens in tqdm(
                    train_data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                loss = self.loss(states, actions, next_states, pad_lens)
                optim.zero_grad()
                loss.backward()
                optim.step()
                ema.update(self)
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_data_loader)
            scheduler.step(avg_loss)

            # evaluation with EMA weights
            train_state = {k: v.clone() for k, v in self.state_dict().items()}
            self.load_state_dict(ema.state_dict)
            test_loss = self.evaluate(test_data_loader)
            current_lr = optim.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, "
                  f"Test Loss: {test_loss:.6f}, LR: {current_lr:.2e}")
            self.load_state_dict(train_state)
        
        # save the EMA weights as the final model
        self.load_state_dict(ema.state_dict)
        if model_name:
            self.save(model_name)

    # <---------------------------- loading and saving ----------------------------->

    def _config(self) -> Dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "seq_len": self.seq_len,
            "visual": self.visual,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "norm_first": self.norm_first,
            "use_absolute_pe": self.use_absolute_pe,
            "use_rope": self.use_rope,
        }

    def save(self, model_name: str) -> None:
        checkpoint = {
            "config": self._config(),
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, os.path.join(MODEL_PATH, model_name))

    @classmethod
    def load(cls, model_name: str, device: str="cuda") -> "WorldModel":
        checkpoint = torch.load(os.path.join(MODEL_PATH, model_name), map_location=device)
        config = checkpoint["config"]
        state_dict = checkpoint["state_dict"]
        
        model = cls(**config)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    

class RolloutContext:
    ''''Context manager for performing rollouts with a world model. 
    Maintains a sliding window of past states and actions to feed into the model for 
    next state prediction.'''

    def __init__(self, model: "WorldModel") -> None:
        self.model = model
        self.device = model.device
        self.seq_len = model.seq_len

    def make_padded(self, x: Tensor, req_len: int) -> Tensor:
        x = x.to(self.device).clone()
        batch, seq_len, *state_shape = x.shape
        if seq_len >= req_len:
            return x[:, -req_len:]
        pad_len = req_len - seq_len
        padding = torch.zeros(batch, pad_len, *state_shape, device=self.device)
        return torch.cat([x, padding], dim=1)
    
    @torch.no_grad()
    def reset(self, init_states: Tensor, init_actions: Tensor) -> None:
        batch, init_len = init_states.shape[:2]
        self.states = self.make_padded(init_states, self.seq_len)
        self.actions = self.make_padded(init_actions, self.seq_len)
        init_pad = max(0, self.seq_len - init_len)
        self.pad_lens = torch.full((batch,), init_pad, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def step(self, actions: Tensor) -> Tensor:
        self.model.eval()

        # set actions at the last real token position for each batch item
        last_real_idx = (self.seq_len - self.pad_lens - 1).clamp(min=0)
        batch_idx = torch.arange(actions.size(0), device=self.device)
        self.actions[batch_idx, last_real_idx] = actions

        # predict next state using the model
        pred = self.model.forward(self.states, self.actions, self.pad_lens)
        next_state = self.model.unnormalize_states(pred)
        
        # for indices with padding, write next index directly into the buffer
        has_pad = self.pad_lens > 0
        if torch.any(has_pad):
            append_idx = last_real_idx[has_pad] + 1
            self.states[has_pad, append_idx] = next_state[has_pad]
        
        # for indices without padding, shift left and append at the end
        if torch.any(~has_pad):
            self.states[~has_pad] = torch.roll(self.states[~has_pad], -1, dims=1)
            self.states[~has_pad, -1] = next_state[~has_pad]
            self.actions[~has_pad] = torch.roll(self.actions[~has_pad], -1, dims=1)
        
        self.pad_lens = (self.pad_lens - 1).clamp(min=0)

        return next_state
    
    @torch.no_grad()
    def rollout(self, init_states: Tensor, init_actions: Tensor, 
                vec_policy, max_steps: int) -> Tensor:
        device = self.device
        
        # reset rollout context with initial state from environment
        init_states = init_states.to(device)
        init_actions = init_actions.to(device)
        if self.model.obs_idx is not None:
            assert len(init_states.shape) == 3, "State must be 3D tensor of shape (B, T, D)."
            init_states = init_states[:, :, self.model.obs_idx]
        self.reset(init_states, init_actions)
                   
        # perform rollout using policy and model
        trajectories = []
        for _ in tqdm(range(max_steps), desc="Rollout"):
            last_real_idx = (self.seq_len - self.pad_lens - 1).clamp(min=0)
            batch_idx = torch.arange(self.states.size(0), device=device)
            last_states = self.states[batch_idx, last_real_idx].detach().cpu().numpy()
            actions = torch.tensor(vec_policy(last_states), dtype=torch.float32, device=device)
            states = self.step(actions)
            trajectories.append(states)

        return torch.stack(trajectories).transpose(0, 1)
    