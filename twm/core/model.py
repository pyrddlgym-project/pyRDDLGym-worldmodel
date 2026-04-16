import os
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Dict, Union

from twm.core.encoding import SinePositionalEncoding, RotaryTransformerEncoderLayer
from twm.core.projection import (
    VectorEncoder, VectorDecoder, ImageEncoder, ImageDecoder, Tensor, TensorDict, ShapeDict
)

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
    def state_dict(self) -> TensorDict:
        '''Returns the EMA weights as a state dictionary, moved to the appropriate device.'''
        return {k: v.to(self.device) for k, v in self.weights.items()}


class WorldModel(nn.Module):
    '''A transformer-based world model that predicts the next state given a sequence of 
    past states and actions.'''

    def __init__(self, state_dims: ShapeDict, action_dims: ShapeDict, visual: bool, 
                 seq_len: int, d_model: int=64, 
                 nhead: int=4, num_layers: int=4, dim_feedforward: int=256, 
                 dropout: float=0.1, norm_first: bool=True, 
                 use_absolute_pe: bool=True, use_rope: bool=True) -> None:
        super().__init__()

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.visual = visual
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.norm_first = norm_first
        self.use_absolute_pe = use_absolute_pe
        self.use_rope = use_rope

        # (state, action) -> (d_model,) embedding -> next state prediction
        self.input_dims = {**state_dims, **action_dims}    
        if self.visual:
            assert len(state_dims) == 1, "For visual inputs, state_dims should have single key."
            assert 'obs' in state_dims, "For visual inputs, state_dims should have key 'obs'."
            c, h, w = state_dims['obs']
            self.input_proj = ImageEncoder(self.input_dims, d_model)
            self.output_proj = ImageDecoder((c, h, w), d_model)
            self.loss_fn = nn.BCELoss()
        else:
            self.input_proj = VectorEncoder(self.input_dims, d_model)
            self.output_proj = VectorDecoder(state_dims, d_model)
            self.loss_fn = nn.HuberLoss()
        
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
        
        # placeholder normalizer buffers
        for key, shape in self.input_dims.items():
            self.register_buffer(f"{key}_mean", torch.zeros(shape))
            self.register_buffer(f"{key}_std", torch.ones(shape))
            self.register_buffer(f"{key}_obs", torch.tensor(True, dtype=torch.bool))
            
    @property
    def device(self) -> torch.device:
        '''Convenience property to get the device of the model parameters.'''
        return next(self.parameters()).device

    # <----------------------------------- data handling ----------------------------------->

    def set_dataset_stats(self, dataset) -> None:
        '''Sets the normalization statistics for states and actions based on the provided dataset.'''
        for key, (mean, std) in dataset.normalizer_stats.items():
            self.register_buffer(f"{key}_mean", mean.to(self.device))
            self.register_buffer(f"{key}_std", std.to(self.device))
        for key in self.state_dims:
            is_obs = torch.tensor(key in dataset.obs_states, dtype=torch.bool, device=self.device)
            self.register_buffer(f"{key}_obs", is_obs)
        
    def observed_inputs(self, inputs: TensorDict) -> TensorDict:
        '''Removes unobservables from the input dict.'''
        result = {}
        for key, tensor in inputs.items():
            if getattr(self, f"{key}_obs"):
                result[key] = tensor
        return result

    def normalize_inputs(self, inputs: TensorDict) -> TensorDict:
        '''Normalizes inputs using the dataset statistics.'''
        result = {}
        for key, tensor in inputs.items():
            if self.visual and key in self.state_dims:
                result[key] = tensor
            else:
                mean = getattr(self, f"{key}_mean")
                std = getattr(self, f"{key}_std")
                mean = mean.view(*(1,) * (tensor.ndim - mean.ndim), *mean.shape)
                std = std.view(*(1,) * (tensor.ndim - std.ndim), *std.shape)
                result[key] = (tensor - mean) / std
        return result
        
    def unnormalize_inputs(self, inputs: TensorDict) -> TensorDict:
        '''Unnormalizes inputs using the dataset statistics.'''
        result = {}
        for key, tensor in inputs.items():
            if self.visual and key in self.state_dims:
                result[key] = tensor
            else:
                mean = getattr(self, f"{key}_mean")
                std = getattr(self, f"{key}_std")
                mean = mean.view(*(1,) * (tensor.ndim - mean.ndim), *mean.shape)
                std = std.view(*(1,) * (tensor.ndim - std.ndim), *std.shape)
                result[key] = tensor * std + mean
        return result

    # <----------------------------------- mask handling ----------------------------------->

    @torch.no_grad()
    def make_padding_mask(self, pad_lens: Tensor, seq_len: int) -> Tensor:
        '''Creates a boolean mask indicating which positions in the input sequence are padding.'''
        pos = torch.arange(seq_len, device=self.device)
        mask = pos.unsqueeze(0) >= (seq_len - pad_lens).unsqueeze(1)  # (batch, seq_len)
        return mask
    
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

    # <----------------------------------- transformer ----------------------------------->

    def encode_history(self, states: TensorDict, actions: TensorDict, pad_lens: Tensor) -> Tensor:
        '''Encodes a history of states and actions into a sequence of latents using the transformer.'''
        # prepare states and actions and map to latent dimension space
        states = self.observed_inputs(states)
        states = self.normalize_inputs(states)
        actions = self.normalize_inputs(actions)
        x = {**states, **actions}
        x = self.input_proj(x)   # (batch, seq_len, d_model)
        x = self.input_norm(x)
        x = x.clone()

        # replace padded positions with learnable PAD embedding
        batch, seq_len = x.shape[:2]
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
        latents = self.transformer(x, mask=mask)
        latents = self.output_norm(latents)
        return latents

    def select_condition(self, latents: Tensor, pad_lens: Tensor) -> Tensor:
        '''Selects the appropriate latent from the transformer output to feed into the decoder.'''
        # if using sequence-level conditioning, return the full sequence of latents
        if self.output_proj.condition_mode == 'sequence':
            return latents
            
        # otherwise, select the latents corresponding to the last real tokens
        batch, seq_len = latents.shape[:2]
        last_real_idx = (seq_len - pad_lens - 1).clamp(min=0)
        batch_idx = torch.arange(batch, device=self.device)
        last_latent = latents[batch_idx, last_real_idx]
        return last_latent

    def forward(self, states: TensorDict, actions: TensorDict, pad_lens: Tensor,
                return_latent: bool=False, unnormalize: bool=False) -> Union[Tensor, TensorDict]:
        '''Predicts the next state or latent given a history of states and actions.'''
        latents = self.encode_history(states, actions, pad_lens)
        latents = self.select_condition(latents, pad_lens)
        if return_latent:
            return latents
        
        next_states = self.output_proj(latents)
        if unnormalize:
            next_states = self.unnormalize_inputs(next_states)
        return next_states
    
    # <---------------------------- training and evaluation ----------------------------->

    def loss(self, states: TensorDict, actions: TensorDict, next_states: TensorDict, 
             pad_lens: Tensor) -> Tensor:
        '''Computes the loss between predicted and true next states for a batch of sequences.'''
        device = self.device
        states = {k: v.to(device) for k, v in states.items()}
        actions = {k: v.to(device) for k, v in actions.items()}
        next_states = {k: v.to(device) for k, v in next_states.items()}
        pad_lens = pad_lens.to(device)
        
        # normalize inputs as targets and get predictions
        targets = self.observed_inputs(next_states)
        targets = self.normalize_inputs(targets)
        preds = self.forward(states, actions, pad_lens, return_latent=False, unnormalize=False)
        assert isinstance(preds, dict), "Model output should be a dict of state predictions."

        # compute loss across all state keys and sum
        loss = torch.tensor(0., device=device)
        for key in self.state_dims.keys():
            loss += self.loss_fn(preds[key], targets[key])
        return loss
    
    @torch.no_grad()
    def evaluate(self, data_loader) -> float:
        '''Evaluates the model on a test dataset and returns the average loss.'''
        self.eval()
        
        if data_loader is None: 
            return float('nan')

        loss = 0.
        for batch_data in tqdm(data_loader, desc="Evaluating"):
            states = batch_data['states']
            actions = batch_data['actions']
            next_states = batch_data['next_states']
            pad_lens = batch_data['pad']
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
            for batch_data in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                states = batch_data['states']
                actions = batch_data['actions']
                next_states = batch_data['next_states']
                pad_lens = batch_data['pad']
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
            "state_dims": self.state_dims,
            "action_dims": self.action_dims,
            "visual": self.visual,
            "seq_len": self.seq_len,
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
    