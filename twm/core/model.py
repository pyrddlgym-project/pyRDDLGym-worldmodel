import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces

from twm.core.encoding import SinePositionalEncoding, RotaryTransformerEncoderLayer
from twm.core.projection import (
    VectorEncoder, VectorDecoder, ImageEncoder, ImageDecoder, DiffusionDecoder
)

Tensor = torch.Tensor

PARENT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MODEL_PATH = os.path.join(PARENT_PATH, "models")


class EMA:
    '''Maintains an exponential moving average of model weights for evaluation stability.'''

    def __init__(self, model: nn.Module, decay: float=0.995) -> None:
        self.decay = decay
        self.weights = {k: v.clone().float().cpu() for k, v in model.state_dict().items()}
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        '''Updates the EMA weights by blending them with the current model weights.'''
        for k, v in model.state_dict().items():
            self.weights[k].mul_(self.decay).add_(v.float().cpu(), alpha=1. - self.decay)

    @property
    def state_dict(self) -> Dict[str, Tensor]:
        '''Returns the EMA weights as a state dictionary, moved to the appropriate device.'''
        return {k: v.to(self.device) for k, v in self.weights.items()}


class WorldModel(nn.Module):
    '''A transformer-based world model that predicts the next state given a sequence of 
    past states and actions.'''

    def __init__(self, state_dim: Union[int, Tuple[int, ...]], action_dim: int, 
                 seq_len: int, visual: bool, d_model: int=64, 
                 nhead: int=4, num_layers: int=4, dim_feedforward: int=256, 
                 dropout: float=0.1, norm_first: bool=True, 
                 use_absolute_pe: bool=True, use_rope: bool=True, 
                 use_diffusion_decoder: bool=False) -> None:
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
        self.use_diffusion_decoder = use_diffusion_decoder

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
            self.pe = SinePositionalEncoding(d_model, max_len=seq_len + 1)
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
                max_len=seq_len + 1,
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
        if self.use_diffusion_decoder:
            self.output_proj = DiffusionDecoder(state_dim=state_dim, d_model=d_model)
            self.loss_fn = self.output_proj.loss
        elif self.visual:
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
        '''Convenience property to get the device of the model parameters.'''
        return next(self.parameters()).device

    def set_dataset_stats(self, dataset) -> None:
        '''Sets the normalization statistics for states and actions based on the provided dataset.'''
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

    def encode_history(self, states: Tensor, actions: Tensor, pad_lens: Tensor) -> Tensor:
        '''Encodes a history of states and actions into a sequence of latents using the transformer.'''
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

        # prepend a dedicated SOS timestep
        sos = self.sos_token.view(1, 1, -1).expand(batch, 1, -1)
        x = torch.cat([sos, x], dim=1)
        seq_len = seq_len + 1

        # positional encoding and transformer
        x = self.pe(x)
        x = self.embed_dropout(x)
        mask = self.make_full_mask(pad_lens, seq_len, batch)
        x = self.transformer(x, mask=mask)
        x = self.output_norm(x)
        return x

    def select_condition(self, latents: Tensor, pad_lens: Tensor) -> Tensor:
        '''Selects the appropriate latent from the transformer output to feed into the decoder.'''
        # if using sequence-level conditioning, return the full sequence of latents
        if getattr(self.output_proj, 'condition_mode', 'last') == 'sequence':
            return latents
            
        # otherwise, select the latents corresponding to the last real tokens
        batch, seq_len = latents.shape[:2]
        last_real_idx = (seq_len - pad_lens - 1).clamp(min=0)
        batch_idx = torch.arange(batch, device=self.device)
        return latents[batch_idx, last_real_idx]

    def forward(self, states: Tensor, actions: Tensor, pad_lens: Tensor,
                latent: bool=False, unnormalize: bool=False) -> Tensor:
        '''Predicts the next state or latent given a history of states and actions.'''
        latents = self.encode_history(states, actions, pad_lens)
        x = self.select_condition(latents, pad_lens)
        if latent:
            return x
        
        x = self.output_proj(x)
        if unnormalize:
            x = self.unnormalize_states(x)
        return x
    
    @torch.no_grad()
    def make_padding_mask(self, pad_lens: Tensor, seq_len: int) -> Tensor:
        '''Creates a boolean mask indicating which positions in the input sequence are padding.'''
        pos = torch.arange(seq_len, device=self.device)
        return pos.unsqueeze(0) >= (seq_len - pad_lens).unsqueeze(1)  # (batch, seq_len)
    
    @torch.no_grad()
    def make_full_mask(self, pad_lens: Tensor, seq_len: int, batch: int) -> Tensor:
        ''''Creates a combined mask that incorporates both causal masking and padding masking.'''
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
        '''Normalizes states using the dataset statistics, or returns raw states if visual.'''
        if self.visual:
            return states
        else:
            return (states - self.state_mean) / self.state_std
    
    def normalize_actions(self, actions: Tensor) -> Tensor:
        '''Normalizes actions using the dataset statistics.'''
        return (actions - self.action_mean) / self.action_std
    
    def unnormalize_states(self, states: Tensor) -> Tensor:
        '''Unnormalizes states using the dataset statistics, or returns raw states if visual.'''
        if self.visual:
            return states
        else:
            return states * self.state_std + self.state_mean
    
    # <---------------------------- training and evaluation ----------------------------->

    def loss(self, states: Tensor, actions: Tensor, next_states: Tensor, 
             pad_lens: Tensor) -> Tensor:
        '''Computes the loss between predicted and true next states for a batch of sequences.'''
        device = self.device
        states = states.to(device)
        actions = actions.to(device)
        next_states = next_states.to(device)
        pad_lens = pad_lens.to(device)
        
        targets = self.normalize_states(next_states)
        pred = self.forward(states, actions, pad_lens, latent=self.use_diffusion_decoder)
        return self.loss_fn(pred, targets)
    
    @torch.no_grad()
    def evaluate(self, data_loader) -> float:
        '''Evaluates the model on a test dataset and returns the average loss.'''
        self.eval()
        
        if data_loader is None: 
            return float('nan')

        loss = 0.
        for states, actions, *_, next_states, pad_lens in tqdm(data_loader, desc="Evaluating"):
            loss += self.loss(states, actions, next_states, pad_lens).item()
        return loss / len(data_loader)

    def fit(self, train_data_loader, epochs: int, 
            optimizer=torch.optim.Adam, lr: float=1e-3, lr_decay: float=0.9, 
            test_data_loader=None, model_name: str='') -> None:
        '''Trains the world model, optionally evaluating on a test dataset after each epoch.'''        
        self.set_dataset_stats(train_data_loader.dataset)

        # create optimizer and learning rate scheduler
        optim = optimizer(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, factor=lr_decay, patience=10, min_lr=1e-5)
        
        # maintain EMA of weights for more stable evaluation
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
        '''Returns a dictionary of the model configuration parameters for saving.'''
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
            "use_diffusion_decoder": self.use_diffusion_decoder,
        }

    def save(self, model_name: str) -> None:
        '''Saves the model configuration and weights to a file for later loading.'''
        checkpoint = {
            "config": self._config(),
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, os.path.join(MODEL_PATH, model_name))

    @classmethod
    def load(cls, model_name: str, device: str="cuda") -> "WorldModel":
        '''Loads a model from a file, reconstructing the architecture from the saved configuration.'''
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

    @torch.no_grad()
    def make_padded(self, x: Tensor, req_len: int) -> Tensor:
        '''Pads a sequence of states or actions to the required length for the model input.'''
        x = x.to(self.device).clone()
        batch, seq_len, *state_shape = x.shape
        if seq_len >= req_len:
            return x[:, -req_len:]
        pad_len = req_len - seq_len
        padding = torch.zeros(batch, pad_len, *state_shape, device=self.device)
        return torch.cat([x, padding], dim=1)
    
    @torch.no_grad()
    def reset(self, init_states: Tensor, init_actions: Optional[Tensor]) -> None:
        '''Resets the rollout context with initial states and actions.'''
        device = self.device

        # ensure batch dimension is present and get batch size and initial sequence length
        batch, init_len = init_states.shape[:2]
        assert init_len >= 1, 'RolloutContext.reset requires at least one initial timestep.'

        # extrcact only the observed state dimensions if obs_idx is set
        init_states = init_states.to(device)
        if self.model.obs_idx is not None:
            assert len(init_states.shape) == 3, "State must be 3D tensor of shape (B, T, D)."
            init_states = init_states[:, :, self.model.obs_idx]
        
        # pad initial states to required sequence length and store in context buffer
        self.states = self.make_padded(init_states, self.seq_len)

        # pad initial actions to required sequence length and store in context buffer
        if init_actions is None:
            assert init_len == 1, "Must pass single initial state."
            self.actions = torch.zeros(batch, self.seq_len, self.model.action_dim, device=device)
        else:
            assert init_actions.shape[1] == init_len - 1, \
                "Initial actions must have one less timestep than initial states."
            init_actions = init_actions.to(device)
            self.actions = self.make_padded(init_actions, self.seq_len)

        # calculate initial padding lengths based on the initial sequence length
        init_pad = max(0, self.seq_len - init_len)
        self.pad_lens = torch.full((batch,), init_pad, dtype=torch.long, device=device)

    @torch.no_grad()
    def step(self, actions: Tensor) -> Tensor:
        '''Performs a rollout step by feeding the current context into the model to predict 
        the next state, then updating the context with the new state and action.'''
        self.model.eval()

        # set actions at the last real token position for each batch item
        last_real_idx = (self.seq_len - self.pad_lens - 1).clamp(min=0)
        batch_idx = torch.arange(actions.size(0), device=self.device)
        self.actions[batch_idx, last_real_idx] = actions

        # predict next state using the model
        next_state = self.model.forward(
            self.states, self.actions, self.pad_lens, latent=False, unnormalize=True)
        
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
    def rollout(self, init_states: Tensor, init_actions: Optional[Tensor], 
                vec_policy, max_steps: int) -> Tensor:
        '''Performs a rollout using the world model and a given policy.'''
        device = self.device

        self.reset(init_states, init_actions)

        trajectories = []
        for _ in tqdm(range(max_steps), desc="Rollout"):
            last_real_idx = (self.seq_len - self.pad_lens - 1).clamp(min=0)
            batch_idx = torch.arange(self.states.size(0), device=device)
            last_states = self.states[batch_idx, last_real_idx].detach().cpu().numpy()
            actions = torch.tensor(vec_policy(last_states), dtype=torch.float32, device=device)
            states = self.step(actions)
            trajectories.append(states)
        return torch.stack(trajectories).transpose(0, 1)


class WorldModelEnv(gym.Env):
    '''A gymnasium environment that uses a world model for state transitions.'''

    def __init__(self, world_model: WorldModel, reward_fn,
                 initial_state: Union[Tensor, np.ndarray], 
                 initial_actions: Optional[Tensor] = None, 
                 min_action: float=-1.0, max_action: float=1.0,
                 max_steps: int=200) -> None:
        '''Initializes the environment with a world model and initial state.'''
        super().__init__()

        self.world_model = world_model
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.device = world_model.device
        
        # setup action and observation spaces
        state_dim = world_model.state_dim
        action_dim = world_model.action_dim
        if isinstance(state_dim, tuple):
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=state_dim, dtype=np.float32)
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=min_action, high=max_action, shape=(action_dim,), dtype=np.float32)
        
        # process and store initial state
        if isinstance(initial_state, np.ndarray):
            initial_state = torch.from_numpy(initial_state).float().to(self.device)
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)  # (1, state_dim)
        if initial_state.dim() == 2:
            initial_state = initial_state.unsqueeze(1)  # (1, 1, state_dim)
        self.initial_state = initial_state
        
        # process and store initial actions if provided
        if initial_actions is not None:
            if isinstance(initial_actions, np.ndarray):
                initial_actions = torch.from_numpy(initial_actions).float().to(self.device)
            if initial_actions.dim() == 1:
                initial_actions = initial_actions.unsqueeze(0)  # (1, action_dim)
            if initial_actions.dim() == 2:
                initial_actions = initial_actions.unsqueeze(1)  # (1, 1, action_dim)
        self.initial_actions = initial_actions 

        # Initialize rollout context
        self.rollout = RolloutContext(world_model)

    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        '''Resets the environment with a new initial state.'''
        self.rollout.reset(self.initial_state, self.initial_actions)
        last_real_idx = (self.rollout.seq_len - self.rollout.pad_lens - 1).clamp(min=0)
        batch_idx = torch.arange(self.rollout.states.size(0), device=self.device)
        self.obs = self.rollout.states[batch_idx, last_real_idx].squeeze().detach().cpu().numpy()
        self.step_num = 0
        return self.obs, {}

    def step(self, action: Union[np.ndarray, Tensor]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        '''Performs one step in the environment.'''
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        action = action.to(self.device)
        if action.dim() == 1:
            action = action.unsqueeze(0)  # (1, action_dim)
        
        # use rollout context to predict next state
        prev_obs = self.obs
        self.obs = self.rollout.step(action).squeeze().detach().cpu().numpy()
        
        # use reward function to evaluate reward
        reward = self.reward_fn(prev_obs, action.squeeze().detach().cpu().numpy(), self.obs)
        self.step_num += 1
        trunc = self.step_num >= self.max_steps
        return self.obs, reward, False, trunc, {}

    def render(self) -> None:
        '''Renders the environment (not implemented).'''
        pass
    