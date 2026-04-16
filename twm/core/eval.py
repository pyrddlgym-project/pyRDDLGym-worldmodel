import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Optional, Dict, Tuple
from tqdm import tqdm

from twm.core.model import WorldModel

Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]


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
    def reset(self, init_states: TensorDict, init_actions: Optional[TensorDict]) -> None:
        '''Resets the rollout context with initial states and actions.'''
        device = self.device

        # ensure batch dimension is present and get batch size and initial sequence length
        batch, init_len = next(iter(init_states.values())).shape[:2]
        assert init_len >= 1, 'RolloutContext.reset requires at least one initial timestep.'

        # extract only the observed states, pad to required length and store in buffer
        self.states = {}
        for key, tensor in init_states.items():
            if getattr(self.model, f'{key}_obs', False):
                tensor = tensor.to(device)
                self.states[key] = self.make_padded(tensor, self.seq_len)

        # pad initial actions to required sequence length and store in context buffer
        if init_actions is None:
            assert init_len == 1, "Must pass single initial state."
            self.actions = {k: torch.zeros(batch, self.seq_len, *shape, device=device)
                            for k, shape in self.model.action_dims.items()}
        else:
            self.actions = {}
            for key, shape in self.model.action_dims.items():
                if key in init_actions:
                    tensor = init_actions[key].to(device)
                    self.actions[key] = self.make_padded(tensor, self.seq_len)
                else:
                    self.actions[key] = torch.zeros(batch, self.seq_len, *shape, device=device)
                
        # calculate initial padding lengths based on the initial sequence length
        init_pad = max(0, self.seq_len - init_len)
        self.pad_lens = torch.full((batch,), init_pad, dtype=torch.long, device=device)

    def index_into_last_epoch(self) -> Tuple[Tensor, Tensor]:
        '''Calculates the indices into the last real state, accounting for padding.'''
        last_real_idx = (self.seq_len - self.pad_lens - 1).clamp(min=0)
        batch = next(iter(self.states.values())).size(0)
        batch_idx = torch.arange(batch, device=self.device)
        return batch_idx, last_real_idx

    @torch.no_grad()
    def step(self, actions: TensorDict) -> TensorDict:
        '''Performs a rollout step by feeding the current context into the model to predict 
        the next state, then updating the context with the new state and action.'''
        self.model.eval()

        # set actions at the last real token position for each batch item
        batch_idx, last_real_idx = self.index_into_last_epoch()
        for key, tensor in actions.items():
            self.actions[key][batch_idx, last_real_idx] = tensor

        # predict next state using the model
        next_states = self.model.forward(
            self.states, self.actions, self.pad_lens, return_latent=False, unnormalize=True)
        assert isinstance(next_states, dict), "Model output should be a dict of state predictions."

        # for indices with padding, write next index directly into the buffer
        has_pad = self.pad_lens > 0
        if torch.any(has_pad):
            append_idx = last_real_idx[has_pad] + 1
            for key, next_state in next_states.items():
                self.states[key][has_pad, append_idx] = next_state[has_pad]
        
        # for indices without padding, shift left and append at the end
        if torch.any(~has_pad):
            for key in self.states.keys():
                self.states[key][~has_pad] = torch.roll(self.states[key][~has_pad], -1, dims=1)
                self.states[key][~has_pad, -1] = next_states[key][~has_pad]
            for key in self.actions.keys():
                self.actions[key][~has_pad] = torch.roll(self.actions[key][~has_pad], -1, dims=1)
        
        # reduce padding lengths by 1, ensuring they don't go below 0
        self.pad_lens = (self.pad_lens - 1).clamp(min=0)

        return next_states
    
    @torch.no_grad()
    def rollout(self, init_states: TensorDict, init_actions: Optional[TensorDict], 
                vec_policy, max_steps: int) -> TensorDict:
        '''Performs a rollout using the world model and a given policy.'''
        device = self.device

        # reset context with initial states and actions
        self.reset(init_states, init_actions)
        
        # perform rollout steps, using the policy to select actions based on the last real state
        trajectories = {}
        for _ in tqdm(range(max_steps), desc="Rollout"):

            # extract the last real state for each batch item to feed into the policy
            batch_idx, last_real_idx = self.index_into_last_epoch()
            last_states_np = {k: v[batch_idx, last_real_idx].detach().cpu().numpy() 
                              for k, v in self.states.items()}
            
            # use the policy to select the next action based on the last real state
            actions = {k: torch.tensor(v, dtype=torch.float32, device=device) 
                       for k, v in vec_policy(last_states_np).items()}
            
            # perform a rollout step with the selected actions to get the next states
            for k, v in self.step(actions).items():
                trajectories.setdefault(k, []).append(v.detach().cpu())

        return {k: torch.stack(v, dim=1) for k, v in trajectories.items()}


class WorldModelEnv(gym.Env):
    '''A gymnasium environment that uses a world model for state transitions.'''

    def __init__(self, world_model: WorldModel, reward_fn,
                 initial_state: TensorDict, 
                 min_action: float=-1.0, max_action: float=1.0,
                 max_steps: int=200) -> None:
        '''Initializes the environment with a world model and initial state.'''
        super().__init__()

        self.world_model = world_model
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.device = world_model.device
        
        # set up observation space using only states marked as observed in the model
        state_dims = {k: v for k, v in world_model.state_dims.items()
                      if bool(getattr(world_model, f'{k}_obs'))}
        if len(state_dims) == 0:
            raise ValueError('WorldModelEnv requires at least one observed state key.')
        
        if world_model.visual:
            assert 'obs' in state_dims, "Visual world model must have 'obs' in state dims."
            assert len(state_dims) == 1, "Visual world model must only have 'obs' in state dims."
            observation_space = spaces.Dict({
                k: spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
                for k, shape in state_dims.items()
            })
        else:
            observation_space = spaces.Dict({
                k: spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
                for k, shape in state_dims.items()
            })
        self._observation_space = observation_space
        self.observation_space = spaces.flatten_space(self._observation_space)

        # set up action space
        action_dims = world_model.action_dims
        self._action_space = spaces.Dict({
            k: spaces.Box(low=min_action, high=max_action, shape=shape, dtype=np.float32)
            for k, shape in action_dims.items()
        })
        self.action_space = spaces.flatten_space(self._action_space)

        # process and store initial state
        self.initial_state = {}
        for key, value in initial_state.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()
            self.initial_state[key] = value.to(self.device)[None, None]   # (1, 1, state_dims...)
        self.initial_action = None

        # Initialize rollout context
        self.rollout = RolloutContext(world_model)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        '''Resets the environment with a new initial state.'''
        # reset the rollout context with the initial state and action
        self.rollout.reset(self.initial_state, self.initial_action)
        self.step_num = 0

        # extract the last real state for each batch item to return as initial obs
        batch_idx, last_real_idx = self.rollout.index_into_last_epoch()
        states_np = {key: tensor[batch_idx, last_real_idx][0].detach().cpu().numpy() 
                     for key, tensor in self.rollout.states.items()}
        self.states_np = states_np
        
        # flatten the observation dict into a single array for the observation space
        obs = spaces.flatten(self._observation_space, states_np)
        return obs, {}

    def step(self, action: np.ndarray):
        '''Performs one step in the environment.'''
        # unflatten the action array into a dict of action tensors
        action_dict = {}
        action_dict_np = {}
        for key, array in spaces.unflatten(self._action_space, action).items():
            if isinstance(array, np.ndarray):
                array = torch.from_numpy(array).float()
            action_dict[key] = array.to(self.device)[None]   # (1, action_dims...)
            action_dict_np[key] = array.detach().cpu().numpy()
        
        # use rollout context to predict next state
        prev_states_np = self.states_np
        states_np = {key: tensor[0].detach().cpu().numpy() 
                     for key, tensor in self.rollout.step(action_dict).items()}
        self.states_np = states_np
        obs = spaces.flatten(self._observation_space, states_np)
        
        # use reward function to evaluate reward
        reward = self.reward_fn(prev_states_np, action_dict_np, states_np)
        self.step_num += 1
        trunc = self.step_num >= self.max_steps
        return obs, reward, False, trunc, {}

    def render(self) -> None:
        '''Renders the environment (not implemented).'''
        pass
    