import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Any, List, Optional, Dict, Set, Tuple, Union
from tqdm import tqdm

from twm.core.model import WorldModel

Array = np.ndarray
ArrayDict = Dict[str, Array]

Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]


class RolloutContext:
    ''''Context manager for performing rollouts with a world model.'''

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
        for key in self.model.state_dims:   
            if key not in init_states:
                raise ValueError(f'Missing initial state for observed key: {key}')
            tensor = init_states[key].to(device)
            self.states[key] = self.make_padded(tensor, self.seq_len)

        # pad initial actions to required sequence length and store in context buffer
        if init_actions is None:
            assert init_len == 1, "Must pass single initial state."
            self.actions = {k: torch.zeros(batch, self.seq_len, *shape, device=device)
                            for k, shape in self.model.action_dims.items()}
        else:
            self.actions = {}
            for key in self.model.action_dims:
                if key not in init_actions:
                    raise ValueError(f'Missing initial action for key: {key}')
                tensor = init_actions[key].to(device)
                self.actions[key] = self.make_padded(tensor, self.seq_len)
                
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
        for key, tensor in self.actions.items():
            if key not in actions:
                raise ValueError(f'Missing action for key: {key}')
            tensor[batch_idx, last_real_idx] = actions[key]

        # predict next state using the model
        next_states = self.model.forward(
            self.states, self.actions, self.pad_lens, return_latent=False, unnormalize=True)
        assert isinstance(next_states, dict), "Model output should be a dict of state predictions."

        # for indices with padding, write next index directly into the buffer
        has_pad = self.pad_lens > 0
        if torch.any(has_pad):
            append_idx = last_real_idx[has_pad] + 1
            for key, tensor in self.states.items():
                if key not in next_states:
                    raise ValueError(f'Model output missing predicted state for key: {key}')
                tensor[has_pad, append_idx] = next_states[key][has_pad]
        
        # for indices without padding, shift left and append at the end
        if torch.any(~has_pad):
            for key, tensor in self.states.items():
                tensor[~has_pad] = torch.roll(tensor[~has_pad], -1, dims=1)
                if key not in next_states:
                    raise ValueError(f'Model output missing predicted state for key: {key}')
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

        # reset context with initial states and actions
        self.reset(init_states, init_actions)
        
        # perform rollout steps, using the policy to select actions based on the last real state
        trajectories = {}
        for _ in tqdm(range(max_steps), desc="Rollout"):

            # extract the last real state for each batch item to feed into the policy
            batch_idx, last_real_idx = self.index_into_last_epoch()
            last_states_np = {key: tensor[batch_idx, last_real_idx].detach().cpu().numpy() 
                              for key, tensor in self.states.items()}
            
            # use the policy to select the next action based on the last real state
            actions = {key: torch.tensor(tensor, dtype=torch.float32, device=device) 
                       for key, tensor in vec_policy(last_states_np).items()}
            
            # perform a rollout step with the selected actions to get the next states
            for key, tensor in self.step(actions).items():
                trajectories.setdefault(key, []).append(tensor.detach().cpu())

        return {key: torch.stack(tensors, dim=1) for key, tensors in trajectories.items()}


class WorldModelEnv(gym.Env):
    '''A gymnasium environment that uses a world model for state transitions.'''

    def __init__(self, world_model: WorldModel, reward_fn,
                 initial_state: Union[TensorDict, ArrayDict], 
                 min_action: float=-1.0, max_action: float=1.0,
                 max_steps: int=200,
                 discrete_action_keys: Optional[Set[str]]=None) -> None:
        '''Initializes the environment with a world model and initial state.'''
        super().__init__()

        self.world_model = world_model
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.device = world_model.device
        self.discrete_action_keys = set(discrete_action_keys or set())

        # determine which state keys are observed and build the observation space accordingly
        self._observation_space = self._build_observation_space()
        self.observation_space = spaces.flatten_space(self._observation_space)

        # build the action space based on the world model's action dimensions
        self.action_space, self._action_specs, self._action_mode = self._build_action_space(
            min_action, max_action)

        # prepare the initial state for the rollout context
        self.initial_state = self._prepare_initial_state(initial_state)
        self.initial_action = None

        # Initialize rollout context
        self.rollout = RolloutContext(world_model)

    # <----------------------------------- build spaces ----------------------------------->

    def _build_observation_space(self) -> spaces.Dict:
        '''Builds the observation space based on the observed state dimensions.'''
        if self.world_model.visual:
            assert 'obs' in self.world_model.state_dims, \
                "Visual world model must have 'obs' in state dims."
            assert len(self.world_model.state_dims) == 1, \
                "Visual world model must only have 'obs' in state dims."
            return spaces.Dict({
                key: spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
                for key, shape in self.world_model.state_dims.items()
            })
        else:
            return spaces.Dict({
                key: spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
                for key, shape in self.world_model.state_dims.items()
            })

    def _build_action_space(self, min_action: float, max_action: float
                            ) -> Tuple[spaces.Space, List[Dict[str, Any]], str]:
        '''Builds the action space based on the world model's action dimensions.'''
        action_dims = self.world_model.action_dims
        unknown_keys = self.discrete_action_keys.difference(action_dims.keys())
        if unknown_keys:
            raise ValueError(f'Unknown discrete action keys: {sorted(unknown_keys)}')

        action_specs = []

        # if all action keys are discrete, use Discrete or MultiDiscrete spaces
        if len(self.discrete_action_keys) == len(action_dims):

            # validate shapes and collect n-values per key
            ns, starts = [], []
            for key, shape in action_dims.items():
                if int(np.prod(shape, dtype=np.int64)) != 1:
                    raise ValueError(f'Discrete action key must be scalar, got {key}: {shape}')
                n = int(max_action - min_action + 1)
                if n < 1:
                    raise ValueError('Invalid action range: max_action must be >= min_action.')
                ns.append(n)
                starts.append(int(min_action))
                action_specs.append({
                    'key': key, 'shape': shape, 'is_discrete': True,
                    'discrete_n': n, 'discrete_start': int(min_action),
                })

            # if only one discrete action key, use Discrete; if multiple, use MultiDiscrete
            if len(ns) == 1:
                action_space = spaces.Discrete(n=ns[0])
                mode = 'single_discrete'
            else:
                # MultiDiscrete uses 0-based; we track starts separately.
                action_space = spaces.MultiDiscrete(ns)
                mode = 'multi_discrete'
        
        # if no discrete action keys, use a single Box for all continuous actions
        elif not self.discrete_action_keys:

            # all continuous: flatten into a single Box
            flat_dim = sum(int(np.prod(shape, dtype=np.int64)) for shape in action_dims.values())
            action_space = spaces.Box(
                low=min_action, high=max_action, shape=(flat_dim,), dtype=np.float32)
            
            # store flat start and end indices for each action key to decode later
            offset = 0
            for key, shape in action_dims.items():
                n_elem = int(np.prod(shape, dtype=np.int64))
                action_specs.append({
                    'key': key, 'shape': shape, 'is_discrete': False,
                    'flat_start': offset, 'flat_end': offset + n_elem,
                })
                offset += n_elem
            mode = 'continuous'
        
        # mixed discrete and continuous action keys are not supported
        else:
            raise ValueError(
                'Mixed discrete/continuous action keys are not supported. '
                'Mark all action keys as discrete or none.')

        return action_space, action_specs, mode

    def _prepare_initial_state(self, initial_state: Union[TensorDict, ArrayDict]) -> TensorDict:
        '''Prepares the initial state by ensuring it has the required keys and converting to tensors.'''
        result = {}
        for key in self.world_model.state_dims:
            if key not in initial_state:
                raise ValueError(f'Missing initial state for key: {key}')
            value = initial_state[key]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()
            result[key] = value.to(self.device)[None, None]   # (1, 1, state_dims...)
        return result

    # <----------------------------------- sampling ----------------------------------->

    def _extract_current_states_np(self) -> ArrayDict:
        '''Extracts the current states as numpy arrays from the rollout context.'''
        batch_idx, last_real_idx = self.rollout.index_into_last_epoch()
        return {
            key: tensor[batch_idx, last_real_idx][0].detach().cpu().numpy()
            for key, tensor in self.rollout.states.items()
        }

    def _decode_action(self, action) -> Tuple[TensorDict, ArrayDict]:
        '''Decodes an action from the gym action space into model-ready tensors.'''
        action_dict, action_dict_np = {}, {}

        # discrete action from SB3 is 0-based
        if self._action_mode == 'single_discrete':
            spec = self._action_specs[0]
            value = float(spec['discrete_start'] + int(action))
            array_np = np.asarray(value, dtype=np.float32).reshape(spec['shape'])
            action_dict[spec['key']] = torch.from_numpy(array_np).to(self.device)[None]
            action_dict_np[spec['key']] = array_np

        # MultiDiscrete is 0-based; add discrete_start to recover original value
        elif self._action_mode == 'multi_discrete':
            action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
            for i, spec in enumerate(self._action_specs):
                value = float(spec['discrete_start'] + int(action_arr[i]))
                array_np = np.asarray(value, dtype=np.float32).reshape(spec['shape'])
                action_dict[spec['key']] = torch.from_numpy(array_np).to(self.device)[None]
                action_dict_np[spec['key']] = array_np

        # continuous
        else: 
            action_flat = np.asarray(action, dtype=np.float32).reshape(-1)
            for spec in self._action_specs:
                array_np = action_flat[spec['flat_start']:spec['flat_end']].reshape(spec['shape'])
                action_dict[spec['key']] = torch.from_numpy(array_np).to(self.device)[None]
                action_dict_np[spec['key']] = array_np

        return action_dict, action_dict_np

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        '''Resets the environment with a new initial state.'''
        super().reset(seed=seed)

        # reset the rollout context with the initial state and action
        self.rollout.reset(self.initial_state, self.initial_action)
        self.step_num = 0

        # extract the current states as numpy arrays for reward calculation
        self.states_np = self._extract_current_states_np()
        
        # flatten the observation dict into a single array for the observation space
        obs = spaces.flatten(self._observation_space, self.states_np)
        return obs, {}

    def step(self, action: np.ndarray):
        '''Performs one step in the environment.'''
        action_dict, action_dict_np = self._decode_action(action)
        
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
    