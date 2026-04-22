import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple

from twm.core.encoding import SinePositionalEncoding, RotaryTransformerEncoderLayer
from twm.core.projection import (
    VectorEncoder, VectorDecoder, ImageEncoder, ImageDecoder, Tensor, TensorDict
)
from twm.core.spec import EnvSpec

Array = np.ndarray
ArrayDict = Dict[str, Array]

PARENT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MODEL_PATH = os.path.join(PARENT_PATH, 'models')


class EMA:
    '''Maintains an exponential moving average of model weights for stability.'''

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

    def __init__(self, env_spec: EnvSpec,  
                 seq_len: int, d_model: int=64, nhead: int=4, num_layers: int=4, 
                 dim_feedforward: int=256, dropout: float=0.1, norm_first: bool=True, 
                 use_absolute_pe: bool=True, use_rope: bool=True) -> None:
        super().__init__()

        self.env_spec = env_spec
        self.input_specs = {**env_spec.state_spec, **env_spec.action_spec}
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
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.loss_fns = nn.ModuleDict()
        for key, spec in self.input_specs.items():
            if spec.prange == 'pixel':
                self.encoders[key] = ImageEncoder(spec.shape, d_model)
                if key in env_spec.state_spec:
                    self.decoders[key] = ImageDecoder(spec.shape, d_model)
                    self.loss_fns[key] = nn.BCELoss()
            else:
                self.encoders[key] = VectorEncoder(spec.shape, d_model)
                if key in env_spec.state_spec:
                    self.decoders[key] = VectorDecoder(spec.shape, d_model)
                    self.loss_fns[key] = nn.HuberLoss()
        
        # combine encoder embeddings into a single latent
        self.input_proj = nn.Linear(len(self.encoders) * d_model, d_model)
        
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
        for key, spec in self.input_specs.items():
            self.register_buffer(f'{key}_mean', torch.zeros(spec.shape))
            self.register_buffer(f'{key}_std', torch.ones(spec.shape))
            self.register_buffer(f'{key}_norm', torch.tensor(True, dtype=torch.bool))
            
    @property
    def device(self) -> torch.device:
        '''Convenience property to get the device of the model parameters.'''
        return next(self.parameters()).device

    # <--------------------------------- data handling --------------------------------->

    def set_dataset_stats(self, dataset) -> None:
        '''Sets the normalization statistics for states and actions for the dataset.'''
        for key in self.input_specs:
            if key in dataset.normalizer_stats:
                mean, std = dataset.normalizer_stats[key]
                self.register_buffer(f'{key}_mean', mean.to(self.device))
                self.register_buffer(f'{key}_std', std.to(self.device))
                self.register_buffer(f'{key}_norm', torch.tensor(True, dtype=torch.bool))
            else:
                self.register_buffer(f'{key}_norm', torch.tensor(False, dtype=torch.bool))
        
    def normalize_inputs(self, inputs: TensorDict) -> TensorDict:
        '''Normalizes inputs using the dataset statistics.'''
        result = {}
        for key, tensor in inputs.items():
            if getattr(self, f'{key}_norm').item():
                mean = getattr(self, f'{key}_mean')
                std = getattr(self, f'{key}_std')
                mean = mean.view(*(1,) * (tensor.ndim - mean.ndim), *mean.shape)
                std = std.view(*(1,) * (tensor.ndim - std.ndim), *std.shape)
                result[key] = (tensor - mean) / std
            else:
                result[key] = tensor
        return result
        
    def unnormalize_inputs(self, inputs: TensorDict) -> TensorDict:
        '''Unnormalizes inputs using the dataset statistics.'''
        result = {}
        for key, tensor in inputs.items():
            if getattr(self, f'{key}_norm').item():
                mean = getattr(self, f'{key}_mean')
                std = getattr(self, f'{key}_std')
                mean = mean.view(*(1,) * (tensor.ndim - mean.ndim), *mean.shape)
                std = std.view(*(1,) * (tensor.ndim - std.ndim), *std.shape)
                result[key] = tensor * std + mean
            else:
                result[key] = tensor
        return result
    
    @torch.no_grad()
    def pad_with_zeros(self, x: Tensor) -> Tensor:
        '''Pads a sequence to the required context length.'''
        x = x.clone()
        batch, x_len, *shape = x.shape
        if x_len >= self.seq_len:
            return x[:, -self.seq_len:]
        pad_len = self.seq_len - x_len
        padding = torch.zeros(batch, pad_len, *shape, device=self.device)
        return torch.cat([x, padding], dim=1)

    # <--------------------------------- mask handling --------------------------------->

    @torch.no_grad()
    def make_padding_mask(self, pad_lens: Tensor, seq_len: int) -> Tensor:
        '''Creates a boolean mask indicating padding positions in the input sequence.'''
        pos = torch.arange(seq_len, device=self.device)
        mask = pos.unsqueeze(0) >= (seq_len - pad_lens).unsqueeze(1)  # (batch, seq_len)
        return mask
    
    @torch.no_grad()
    def make_full_mask(self, pad_lens: Tensor, seq_len: int, batch: int) -> Tensor:
        '''Creates a combined mask that incorporates causal and padding masking.'''
        device = self.device

        # create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device)

        # add padding mask to causal mask
        pad_mask_ = self.make_padding_mask(pad_lens, seq_len)   # (batch, seq_len)
        padding_mask = torch.zeros(batch, 1, seq_len, device=device)
        padding_mask = padding_mask.masked_fill(pad_mask_.unsqueeze(1), float('-inf'))
        mask = causal_mask.unsqueeze(0) + padding_mask   # (batch, seq_len, seq_len)

        # ensure padded positions can attend to themselves to prevent NaNs
        eye = torch.eye(seq_len, dtype=torch.bool, device=device)
        mask = mask.masked_fill(pad_mask_.unsqueeze(-1) & eye, 0.0)
        mask = mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        mask = mask.reshape(batch * self.nhead, seq_len, seq_len)
        return mask

    # <---------------------------------- transformer ---------------------------------->

    def embed_inputs(self, states: TensorDict, actions: TensorDict) -> Tensor:
        '''Projects states and actions into the transformer input space.'''
        # filter and normalize inputs
        states = {k: v for k, v in states.items() if k in self.env_spec.state_spec}
        actions = {k: v for k, v in actions.items() if k in self.env_spec.action_spec}
        states = self.normalize_inputs(states)
        actions = self.normalize_inputs(actions)
        
        # embed each state and action separately
        state_enc = [self.encoders[k](v) for k, v in states.items()]
        action_enc = [self.encoders[k](v) for k, v in actions.items()]

        # concatenate all embeddings and project to the embedding space
        x = torch.cat(state_enc + action_enc, dim=-1)  # (batch, seq_len, n_input * d_model)
        x = self.input_proj(x)   # (batch, seq_len, d_model)
        return x
    
    def encode_history(self, states: TensorDict, actions: TensorDict, pad_lens: Tensor
                       ) -> Tensor:
        '''Encodes a history of states and actions into latents using the transformer.'''
        # combine all state and action embeddings into a single sequence of latents
        x = self.embed_inputs(states, actions)   # (batch, seq_len, d_model)
        x = self.input_norm(x).clone()

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

    def select_condition(self, latents: Tensor, pad_lens: Tensor) -> TensorDict:
        '''Selects the appropriate latent from the transformer to feed to the decoder.'''
        # select the latents corresponding to the last real tokens
        batch, seq_len = latents.shape[:2]
        last_real_idx = (seq_len - pad_lens - 1).clamp(min=0)
        batch_idx = torch.arange(batch, device=self.device)
        last_latent = latents[batch_idx, last_real_idx]

        # choose latents depending on the decoder's conditioning mode
        result_latents = {}
        for key, decoder in self.decoders.items():
            if decoder.condition_mode == 'last':
                result_latents[key] = last_latent
            elif decoder.condition_mode == 'sequence':
                result_latents[key] = latents
            else:
                raise ValueError(f'Invalid condition mode: {decoder.condition_mode}')
        return result_latents

    def forward(self, states: TensorDict, actions: TensorDict, pad_lens: Tensor,
                return_latent: bool=False, unnormalize: bool=False) -> TensorDict:
        '''Predicts the next state or latent given a history of states and actions.'''
        # encode the history of states and actions into latents
        latents = self.encode_history(states, actions, pad_lens)
        latents = self.select_condition(latents, pad_lens)
        if return_latent:
            return latents
        
        # decode the latents into next state predictions for each observed key
        next_states = {}
        for key, decoder in self.decoders.items():
            if key not in self.env_spec.state_spec:
                raise ValueError(f'Decoder key not in state spec: {key}')
            next_states[key] = decoder(latents[key])
        if unnormalize:
            next_states = self.unnormalize_inputs(next_states)
        return next_states
    
    # <---------------------------- training and evaluation ----------------------------->

    def to_device(self, values: TensorDict) -> TensorDict:
        '''Moves a dictionary of tensors to the same device and dtype as the model.'''
        return {k: v.to(self.device) for k, v in values.items()}
    
    def loss(self, states: TensorDict, actions: TensorDict, next_states: TensorDict, 
             pad_lens: Tensor) -> Tensor:
        '''Computes the loss between predicted and true next states.'''
        # normalize inputs as targets
        targets = {k: v for k, v in next_states.items() if k in self.env_spec.state_spec}
        targets = self.normalize_inputs(targets)

        # predict next states from the model
        preds = self.forward(states, actions, pad_lens, unnormalize=False)
        assert isinstance(preds, dict), 'Model output must be a dict.'

        # compute loss across all state keys and sum
        loss = torch.tensor(0., device=self.device)
        for key in targets.keys():
            loss += self.loss_fns[key](preds[key], targets[key])
        return loss
    
    @torch.no_grad()
    def evaluate(self, data_loader) -> float:
        '''Evaluates the model on a test dataset and returns the average loss.'''
        self.eval()
        
        if data_loader is None: 
            return float('nan')

        loss = 0.
        for batch_data in tqdm(data_loader, desc='Evaluating'):
            states = self.to_device(batch_data['states'])
            actions = self.to_device(batch_data['actions'])
            next_states = self.to_device(batch_data['next_states'])
            pad_lens = batch_data['pad'].to(self.device)
            loss += self.loss(states, actions, next_states, pad_lens).item()
        return loss / len(data_loader)

    def fit(self, train_data_loader, epochs: int, 
            optimizer=torch.optim.Adam, lr: float=1e-3, lr_decay: float=0.9, 
            test_data_loader=None, model_name: str='') -> None:
        '''Trains the world model, optionally evaluating on a test dataset.'''        
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
            for batch_data in tqdm(train_data_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                states = self.to_device(batch_data['states'])
                actions = self.to_device(batch_data['actions'])
                next_states = self.to_device(batch_data['next_states'])
                pad_lens = batch_data['pad'].to(self.device)
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
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, '
                  f'Test Loss: {test_loss:.6f}, LR: {current_lr:.2e}')
            self.load_state_dict(train_state)
        
        # save the EMA weights as the final model
        self.load_state_dict(ema.state_dict)
        if model_name:
            self.save(model_name)

    # <---------------------------- loading and saving ----------------------------->

    def _config(self) -> Dict[str, Any]:
        '''Returns a dictionary of the model configuration parameters for saving.'''
        return {
            'env_spec': self.env_spec.serialize(),
            'seq_len': self.seq_len,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'norm_first': self.norm_first,
            'use_absolute_pe': self.use_absolute_pe,
            'use_rope': self.use_rope,
        }

    def save(self, model_name: str) -> None:
        '''Saves the model configuration and weights to a file for later loading.'''
        checkpoint = {
            'config': self._config(),
            'state_dict': self.state_dict(),
        }
        torch.save(checkpoint, os.path.join(MODEL_PATH, model_name))

    @classmethod
    def load(cls, model_name: str, device: str='cuda') -> 'WorldModel':
        '''Loads a model from a file, reconstructing the architecture from the config.'''
        checkpoint = torch.load(
            os.path.join(MODEL_PATH, model_name), map_location=device, weights_only=False)
        config = checkpoint['config']
        config['env_spec'] = EnvSpec.deserialize(config['env_spec'])
        state_dict = checkpoint['state_dict']
        
        model = cls(**config)
        model.load_state_dict(state_dict)
        model.eval()
        return model


class WorldModelEvaluator:
    '''Context manager for performing rollouts with a world model.'''

    def __init__(self, model: WorldModel) -> None:
        self.model = model
        self.device = model.device

    @torch.no_grad()
    def reset(self, init_states: TensorDict, init_actions: Optional[TensorDict]) -> None:
        '''Resets the rollout context with initial states and actions.'''
        device = self.device

        # ensure batch dimension and get batch size and initial sequence length
        batch, init_len = next(iter(init_states.values())).shape[:2]
        assert init_len >= 1, 'reset requires at least one initial timestep.'

        # extract only the observed states, pad to required length and store in buffer
        self.states = {}
        for key in self.model.env_spec.state_spec:  
            tensor = init_states[key].to(device)
            self.states[key] = self.model.pad_with_zeros(tensor)

        # pad initial actions to required sequence length and store in context buffer
        if init_actions is None:
            assert init_len == 1, 'Must pass single initial state.'
            self.actions = {
                k: torch.zeros(batch, self.model.seq_len, *spec.shape, device=device)
                for k, spec in self.model.env_spec.action_spec.items()
            }
        else:
            self.actions = {}
            for key in self.model.env_spec.action_spec:
                tensor = init_actions[key].to(device)
                self.actions[key] = self.model.pad_with_zeros(tensor)
                
        # calculate initial padding lengths based on the initial sequence length
        init_pad = max(0, self.model.seq_len - init_len)
        self.pad_lens = torch.full((batch,), init_pad, dtype=torch.long, device=device)

    def index_into_last_epoch(self) -> Tuple[Tensor, Tensor]:
        '''Calculates the indices into the last real state, accounting for padding.'''
        last_real_idx = (self.model.seq_len - self.pad_lens - 1).clamp(min=0)
        batch = next(iter(self.states.values())).size(0)
        batch_idx = torch.arange(batch, device=self.device)
        return batch_idx, last_real_idx

    def last_states(self, to_numpy: bool=False) -> ArrayDict | TensorDict:
        '''Extracts the last real states from the rollout context.'''
        batch_idx, last_real_idx = self.index_into_last_epoch()
        result = {}
        for key, tensor in self.states.items():
            value = tensor[batch_idx, last_real_idx]
            if to_numpy:
                value = value.detach().cpu().numpy()
            result[key] = value
        return result
    
    @torch.no_grad()
    def step(self, actions: TensorDict) -> TensorDict:
        '''Performs a rollout step by feeding the current context into the model to 
        predict the next state, then updating the context with the new state and action.'''
        self.model.eval()

        # set actions at the last real token position for each batch item
        batch_idx, last_real_idx = self.index_into_last_epoch()
        for key, tensor in self.actions.items():
            tensor[batch_idx, last_real_idx] = actions[key]

        # predict next state using the model
        next_states = self.model.forward(
            self.states, self.actions, self.pad_lens, unnormalize=True)
        assert isinstance(next_states, dict), 'Model output must be a dict.'

        # for indices with padding, write next index directly into the buffer
        has_pad = self.pad_lens > 0
        if torch.any(has_pad):
            append_idx = last_real_idx[has_pad] + 1
            for key, tensor in self.states.items():
                tensor[has_pad, append_idx] = next_states[key][has_pad]
        
        # for indices without padding, shift left and append at the end
        if torch.any(~has_pad):
            for key, tensor in self.states.items():
                tensor[~has_pad] = torch.roll(tensor[~has_pad], -1, dims=1)
                tensor[~has_pad, -1] = next_states[key][~has_pad]
            for key, tensor in self.actions.items():
                tensor[~has_pad] = torch.roll(tensor[~has_pad], -1, dims=1)
        
        # reduce padding lengths by 1, ensuring they don't go below 0
        self.pad_lens = (self.pad_lens - 1).clamp(min=0)

        return {key: next_states[key] for key in self.states}
    
    @torch.no_grad()
    def rollout(self, init_states: TensorDict, init_actions: Optional[TensorDict], 
                vec_policy, max_steps: int) -> TensorDict:
        '''Performs a rollout using the world model and a given policy.'''
        device = self.device
        self.reset(init_states, init_actions)
        
        trajectories = {}
        for _ in tqdm(range(max_steps), desc='Rollout'):

            # use the policy to select the next action based on the last real state
            last_states_np = self.last_states(to_numpy=True)
            actions = {
                k: torch.from_numpy(v).float().to(device) 
                for k, v in vec_policy(last_states_np).items()
            }
            
            # perform a rollout step with the selected actions to get the next states
            for key, tensor in self.step(actions).items():
                trajectories.setdefault(key, []).append(tensor.detach().cpu())

        return {k: torch.stack(vs, dim=1) for k, vs in trajectories.items()}
