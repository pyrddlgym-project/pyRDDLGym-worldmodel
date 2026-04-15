import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Dict, Tuple, Union
from tqdm import tqdm

from twm.core.model import WorldModel


class RolloutContext:
    ''''Context manager for performing rollouts with a world model. 
    Maintains a sliding window of past states and actions to feed into the model for 
    next state prediction.'''

    def __init__(self, model: WorldModel) -> None:
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
    