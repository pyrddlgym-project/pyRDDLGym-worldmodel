import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple

from twm.core.encoding import SinePositionalEncoding, RotaryTransformerEncoderLayer
from twm.core.projection import VectorEncoder, VectorDecoder, ImageEncoder, ImageDecoder
from twm.core.spec import EnvSpec

Array = np.ndarray
ArrayDict = Dict[str, Array]
Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]

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
        '''Returns the EMA weights as a state dict, moved to the appropriate device.'''
        return {k: v.to(self.device) for k, v in self.weights.items()}


DEFAULT_ENCODER_TYPE = {
    'pixel': ImageEncoder,
    'real': VectorEncoder,
    'int': VectorEncoder,
    'bool': VectorEncoder,
}

DEFAULT_DECODER_TYPE = {
    'pixel': ImageDecoder,
    'real': VectorDecoder,
    'int': VectorDecoder,
    'bool': VectorDecoder,
}

DEFAULT_LOSS_FN = {
    'pixel': nn.BCEWithLogitsLoss(),
    'real': nn.HuberLoss(),
    'int': nn.CrossEntropyLoss(),
    'bool': nn.CrossEntropyLoss(),
}


class WorldModel(nn.Module):
    '''A transformer-based world model that predicts the next state given a sequence of 
    past states and actions.'''

    def __init__(self, env_spec: EnvSpec,  
                 seq_len: int, d_model: int=64, nhead: int=4, num_layers: int=4, 
                 dim_feedforward: int=256, dropout: float=0.1, norm_first: bool=True, 
                 use_absolute_pe: bool=True, use_rope: bool=True) -> None:
        super().__init__()

        self.env_spec = env_spec
        self.all_spec = env_spec.all_spec
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
        self._create_encoders_and_decoders_from_spec()
        self.input_proj = nn.Linear(len(self.encoders) * d_model, d_model)
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
        for key, spec in self.all_spec.items():
            self.register_buffer(f'{key}_mean', torch.zeros(spec.shape))
            self.register_buffer(f'{key}_std', torch.ones(spec.shape))
            self.register_buffer(f'{key}_norm', torch.tensor(False, dtype=torch.bool))

    def spec_bounds(self, key: str) -> Tuple[int, int]:
        '''Returns the (low, high) range of values for a given discrete key.'''
        spec = self.all_spec[key]
        values = (0, 1) if spec.prange == 'bool' else spec.values
        assert values is not None and len(values) == 2, \
            f'Expected 2 values for discrete spec {key}, got {values}'
        low, high = int(values[0]), int(values[1])
        return low, high

    def _create_encoders_and_decoders_from_spec(self):
        d_model = self.d_model
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.loss_fns = nn.ModuleDict()

        for key, spec in self.all_spec.items():
            prange = spec.prange
            encoder_type = DEFAULT_ENCODER_TYPE.get(prange, VectorEncoder)
            decoder_type = DEFAULT_DECODER_TYPE.get(prange, VectorDecoder)
            loss_fn = DEFAULT_LOSS_FN.get(prange, nn.HuberLoss())

            # use CNN layers for pixels, with binary cross-entropy loss, MLP for real
            if prange in ('pixel', 'real'):
                self.encoders[key] = encoder_type(spec.shape, d_model)
                if key in self.env_spec.state_spec:
                    self.decoders[key] = decoder_type(spec.shape, d_model)
                    self.loss_fns[key] = loss_fn
            
            # use one-hot encoding and cross entropy for discrete actions
            elif prange in ('int', 'bool'):
                low, high = self.spec_bounds(key)
                n_classes = high - low + 1
                new_shape = (*spec.shape, n_classes)
                self.encoders[key] = encoder_type(new_shape, d_model)
                if key in self.env_spec.state_spec:
                    self.decoders[key] = decoder_type(new_shape, d_model)
                    self.loss_fns[key] = loss_fn

            else:
                raise ValueError(f'Unknown prange for key {key}: {prange}')

    @property
    def device(self) -> torch.device:
        '''Convenience property to get the device of the model parameters.'''
        return next(self.parameters()).device

    # <--------------------------------- mask handling --------------------------------->

    @torch.no_grad()
    def make_padding_mask(self, pad_lens: Tensor, seq_len: int) -> Tensor:
        '''Creates a boolean mask indicating padding positions in the input sequence.'''
        pos = torch.arange(seq_len, device=pad_lens.device)
        mask = pos.unsqueeze(0) >= (seq_len - pad_lens).unsqueeze(1)  # (batch, seq_len)
        return mask
    
    @torch.no_grad()
    def make_full_mask(self, pad_lens: Tensor, seq_len: int, batch: int) -> Tensor:
        '''Creates a combined mask that incorporates causal and padding masking.'''
        # create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=pad_lens.device)

        # add padding mask to causal mask
        pad_mask_ = self.make_padding_mask(pad_lens, seq_len)   # (batch, seq_len)
        padding_mask = torch.zeros(batch, 1, seq_len, device=pad_lens.device)
        padding_mask = padding_mask.masked_fill(pad_mask_.unsqueeze(1), float('-inf'))
        mask = causal_mask.unsqueeze(0) + padding_mask   # (batch, seq_len, seq_len)

        # ensure padded positions can attend to themselves to prevent NaNs
        eye = torch.eye(seq_len, dtype=torch.bool, device=pad_lens.device)
        mask = mask.masked_fill(pad_mask_.unsqueeze(-1) & eye, 0.0)
        mask = mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        mask = mask.reshape(batch * self.nhead, seq_len, seq_len)
        return mask

    # <---------------------------------- prediction ---------------------------------->

    def one_hot(self, tensor: Tensor, key: str) -> Tensor:
        '''Converts an int discrete tensor to a one-hot float for grad-mode buffers.'''
        spec = self.all_spec[key]
        if spec.prange in ('int', 'bool'):
            low, high = self.spec_bounds(key)
            n_classes = high - low + 1
            return F.one_hot(tensor.long() - low, n_classes).float()
        else:
            return tensor.float()
    
    def prepare_inputs(self, inputs: TensorDict, grad: bool=False) -> TensorDict:
        '''Prepares inputs to feed to the transformer.'''
        result = {}
        for key, tensor in inputs.items():
            spec = self.all_spec[key]

            # pixel values fed directly
            if spec.prange == 'pixel':
                result[key] = tensor.float()

            # reals normalized with dataset stats
            elif spec.prange == 'real':
                mean = getattr(self, f'{key}_mean')
                std = getattr(self, f'{key}_std')
                mean = mean.view(*(1,) * (tensor.ndim - mean.ndim), *mean.shape)
                std = std.view(*(1,) * (tensor.ndim - std.ndim), *std.shape)
                result[key] = (tensor.float() - mean) / std
            
            # discrete converted to one-hot vectors
            elif spec.prange in ('int', 'bool'):
                result[key] = tensor.float() if grad else self.one_hot(tensor, key)
            
            else:
                raise ValueError(f'Unknown prange for key {key}: {spec.prange}')
                
        return result
        
    def embed_inputs(self, states: TensorDict, actions: TensorDict, 
                     grad: bool=False) -> Tensor:
        '''Projects states and actions into the transformer input space.'''
        # filter and normalize inputs
        states = {k: states[k] for k in self.env_spec.state_spec}
        actions = {k: actions[k] for k in self.env_spec.action_spec}
        states = self.prepare_inputs(states, grad=grad)
        actions = self.prepare_inputs(actions, grad=grad)
        
        # embed each state and action separately
        state_enc = [self.encoders[k](v) for k, v in states.items()]
        action_enc = [self.encoders[k](v) for k, v in actions.items()]

        # concatenate all embeddings and project to the embedding space
        x = torch.cat(state_enc + action_enc, dim=-1)  # (batch, seq_len, num * d_model)
        x = self.input_proj(x)   # (batch, seq_len, d_model)
        return x
    
    def encode_history(self, states: TensorDict, actions: TensorDict, pad_lens: Tensor,
                       grad: bool=False) -> Tensor:
        '''Encodes a history of states and actions into latents using the transformer.'''
        # combine all state and action embeddings into a single sequence of latents
        x = self.embed_inputs(states, actions, grad=grad)
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
        batch_idx = torch.arange(batch, device=latents.device)
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

    def prepare_outputs(self, inputs: TensorDict, grad: bool=False, 
                        stochastic: bool=False, temperature: float=1.0) -> TensorDict:
        '''Prepares outputs using the dataset statistics.'''
        if temperature <= 0:
            raise ValueError('temperature must be > 0.')

        result = {}
        for key, tensor in inputs.items():
            spec = self.all_spec[key]

            # pixel values fed directly
            if spec.prange == 'pixel':
                result[key] = torch.sigmoid(tensor).float()

            # real outputs rescaled back to original range
            elif spec.prange == 'real':
                mean = getattr(self, f'{key}_mean')
                std = getattr(self, f'{key}_std')
                mean = mean.view(*(1,) * (tensor.ndim - mean.ndim), *mean.shape)
                std = std.view(*(1,) * (tensor.ndim - std.ndim), *std.shape)
                result[key] = tensor.float() * std + mean
            
            # discrete outputs converted from one-hot vectors back to integer values
            elif spec.prange in ('int', 'bool'):
                if grad:
                    result[key] = F.softmax(tensor.float(), dim=-1)
                else:
                    if stochastic:
                        probs = F.softmax(tensor.float() / temperature, dim=-1)
                        flat_probs = probs.reshape(-1, probs.shape[-1])
                        idx = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
                        idx = idx.view(*probs.shape[:-1])
                    else:
                        idx = tensor.argmax(dim=-1)
                    low, high = self.spec_bounds(key)
                    assert idx.max() < (high - low + 1)
                    result[key] = idx + low
            
            else:
                raise ValueError(f'Unknown prange for key {key}: {spec.prange}')

        return result
    
    def forward(self, states: TensorDict, actions: TensorDict, pad_lens: Tensor,
                return_latent: bool=False, decode_output: bool=False, 
                grad: bool=False, **output_kwargs) -> TensorDict:
        '''Predicts the next state or latent given a history of states and actions.'''
        # encode the history of states and actions into latents
        latents = self.encode_history(states, actions, pad_lens, grad=grad)
        latents = self.select_condition(latents, pad_lens)
        if return_latent:
            return latents
        
        # decode the latents into next state predictions for each observed key
        next_states = {k: self.decoders[k](latents[k]) for k in self.env_spec.state_spec}
        if decode_output:
            next_states = self.prepare_outputs(next_states, grad=grad, **output_kwargs)
        return next_states
    
    # <---------------------------- training and evaluation ----------------------------->

    def set_dataset_stats(self, dataset) -> None:
        '''Sets the normalization statistics for states and actions for the dataset.'''
        device = self.device
        for key in self.all_spec:
            if key in dataset.normalizer_stats:
                mean, std = dataset.normalizer_stats[key]
                self.register_buffer(f'{key}_mean', mean.to(device))
                self.register_buffer(f'{key}_std', std.to(device))
                self.register_buffer(f'{key}_norm', torch.tensor(True, dtype=torch.bool))
            else:
                self.register_buffer(f'{key}_norm', torch.tensor(False, dtype=torch.bool))
                    
    def loss(self, states: TensorDict, actions: TensorDict, next_states: TensorDict, 
             pad_lens: Tensor) -> Tensor:
        '''Computes the loss between predicted and true next states.'''
        # normalize inputs as targets
        targets = {k: next_states[k] for k in self.env_spec.state_spec}
        targets = self.prepare_inputs(targets)

        # predict next states from the model
        preds = self.forward(states, actions, pad_lens, decode_output=False)

        # compute loss across all state keys and sum
        loss = torch.tensor(0., device=pad_lens.device)
        for key in targets:
            pred, target = preds[key], targets[key]
            if self.all_spec[key].prange in ('int', 'bool'):  # for cross-entropy
                pred = pred.movedim(-1, 1)
                target = target.movedim(-1, 1)
            loss += self.loss_fns[key](pred, target)
        return loss

    @staticmethod
    def dict_to_device(values: TensorDict, device: torch.device) -> TensorDict:
        '''Moves a dictionary of tensors to the same device and dtype as the model.'''
        return {k: v.to(device) for k, v in values.items()}
        
    @torch.no_grad()
    def evaluate(self, data_loader) -> float:
        '''Evaluates the model on a test dataset and returns the average loss.'''
        self.eval()
        device = self.device
        
        if data_loader is None: 
            return float('nan')

        loss = 0.
        for batch_data in tqdm(data_loader, desc='Evaluating'):
            states = self.dict_to_device(batch_data['states'], device)
            actions = self.dict_to_device(batch_data['actions'], device)
            next_states = self.dict_to_device(batch_data['next_states'], device)
            pad_lens = batch_data['pad'].to(device)
            loss += self.loss(states, actions, next_states, pad_lens).item()
        return loss / len(data_loader)

    def fit(self, train_data_loader, epochs: int, 
            optimizer=torch.optim.Adam, lr: float=1e-3, lr_decay: float=0.9, 
            test_data_loader=None, model_name: str='') -> None:
        '''Trains the world model, optionally evaluating on a test dataset.'''      
        device = self.device  
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
                states = self.dict_to_device(batch_data['states'], device)
                actions = self.dict_to_device(batch_data['actions'], device)
                next_states = self.dict_to_device(batch_data['next_states'], device)
                pad_lens = batch_data['pad'].to(device)
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
        self.seq_len = model.seq_len
        self.device = model.device

    def pad_with_zeros(self, x: Tensor) -> Tensor:
        '''Pads a sequence to the required context length.'''
        x = x.to(self.device).clone()
        batch, x_len, *shape = x.shape
        if x_len >= self.seq_len:
            return x[:, -self.seq_len:]
        pad_len = self.seq_len - x_len
        padding = torch.zeros(batch, pad_len, *shape, device=self.device)
        return torch.cat([x, padding], dim=1)

    def reset(self, init_states: TensorDict, init_actions: Optional[TensorDict],
              grad: bool=False) -> None:
        '''Resets the rollout context with initial states and actions.'''
        spec = self.model.env_spec
        
        # calculate initial padding lengths based on the length of the initial states
        self.batch, init_len = next(iter(init_states.values())).shape[:2]
        assert init_len >= 1, 'reset requires at least one initial timestep.'
        self.pad_len = max(self.seq_len - init_len, 0)

        # extract only the observed states, pad to required length and store in buffer;
        # in grad mode, discrete states are stored as one-hot floats rather than integers
        self.states = {
            k: self.pad_with_zeros(
                self.model.one_hot(init_states[k], k) if grad else init_states[k]) 
            for k in spec.state_spec
        }

        # pad initial actions to required sequence length and store in context buffer
        if init_actions is None:
            assert init_len == 1, 'Must pass single initial state.'
            self.actions = {}
            for key, spec in spec.action_spec.items():
                if grad and spec.prange in ('int', 'bool'):
                    low, high = self.model.spec_bounds(key)
                    n_classes = high - low + 1
                    shape = (self.batch, self.seq_len, *spec.shape, n_classes)
                else:
                    shape = (self.batch, self.seq_len, *spec.shape)
                self.actions[key] = torch.zeros(*shape, device=self.device)
        else:
            self.actions = {
                k: self.pad_with_zeros(
                    self.model.one_hot(init_actions[k], k) if grad else init_actions[k])
                for k in spec.action_spec
            }
        
    def index_of_last_epoch(self) -> int:
        '''Calculates the index into the last real state, accounting for padding.'''
        return max(self.seq_len - self.pad_len - 1, 0)
    
    def last_states(self, to_numpy: bool=False, squash: bool=False) -> ArrayDict | TensorDict:
        '''Extracts the last real states from the rollout context.'''
        last_idx = self.index_of_last_epoch()
        result = {}
        for key, tensor in self.states.items():
            value = tensor[:, last_idx]
            if squash:
                value = value.squeeze(0)
            if to_numpy:
                value = value.detach().cpu().numpy()
            result[key] = value
        return result
    
    def step(self, actions: TensorDict, grad: bool=False, **output_kwargs) -> TensorDict:
        '''Performs a rollout step by feeding the current context into the model to 
        predict the next state, then updating with the new state and action.'''
        self.model.eval()

        # set actions at the last real token position for each batch item
        last_idx = self.index_of_last_epoch()
        for key, tensor in actions.items():
            y = self.actions[key].clone()
            y[:, last_idx] = tensor
            self.actions[key] = y

        # predict next state using the model
        pad_lens = torch.full((self.batch,), self.pad_len, device=self.device)
        with (torch.enable_grad() if grad else torch.no_grad()):
            next_states = self.model.forward(
                self.states, self.actions, pad_lens, decode_output=True, 
                grad=grad, **output_kwargs)
        assert isinstance(next_states, dict), 'Model output must be a dict.'

        # if there is no padding, roll the state and action buffers
        if self.pad_len == 0:
            self.states = {k: torch.roll(v, -1, dims=1) for k, v in self.states.items()}
            self.actions = {k: torch.roll(v, -1, dims=1) for k, v in self.actions.items()}
            state_write_index = -1
        else:
            state_write_index = last_idx + 1

        # write the new states into the buffer at the appropriate position
        for key, tensor in next_states.items():
            y = self.states[key].clone()
            y[:, state_write_index] = tensor
            self.states[key] = y
        
        # reduce padding lengths by 1, ensuring they don't go below 0
        self.pad_len = max(self.pad_len - 1, 0)

        return next_states
    
    def rollout(self, init_states: TensorDict, init_actions: Optional[TensorDict], 
                policy, max_steps: int, grad: bool=False, **output_kwargs) -> TensorDict:
        '''Performs a rollout using the world model and a given policy.'''
        self.reset(init_states, init_actions, grad=grad)
        
        trajectories = {}
        for _ in tqdm(range(max_steps), desc='Rollout'):
            actions = policy(self.last_states())
            next_states = self.step(actions, grad=grad, **output_kwargs)
            for key, tensor in next_states.items():
                if not grad:
                    tensor = tensor.detach().cpu()
                trajectories.setdefault(key, []).append(tensor)
                
        return {k: torch.stack(vs, dim=1) for k, vs in trajectories.items()}
