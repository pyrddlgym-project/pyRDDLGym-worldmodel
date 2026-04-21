import gymnasium as gym
from gymnasium import spaces
import itertools
import numpy as np
import torch
from typing import Any, List, Optional, Dict, Set, Tuple, Union

from twm.core.model import WorldModel, WorldModelEvaluator

Array = np.ndarray
ArrayDict = Dict[str, Array]
Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]


class WorldModelEnv(gym.Env):
    '''A gymnasium environment that uses a world model for state transitions.'''

    def __init__(self, world_model: WorldModel, reward_fn,
                 initial_state: Union[TensorDict, ArrayDict], max_steps: int,
                 min_action: float=-1.0, max_action: float=1.0,
                 discrete_action_keys: Optional[Set[str]]=None) -> None:
        '''Initializes the environment with a world model and initial state.'''
        super().__init__()

        self.world_model = world_model
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.device = world_model.device
        self.discrete_action_keys = set(discrete_action_keys or set())

        # determine which state keys are observed and build the obs space accordingly
        self._observation_space = self._build_observation_space()
        self.observation_space = spaces.flatten_space(self._observation_space)

        # build the action space based on the world model's action dimensions
        self.action_space, self._action_specs, self._action_mode = self._build_action_space(
            min_action, max_action)

        # prepare the initial state for the rollout context
        self.initial_state = self._prepare_initial_state(initial_state)
        self.initial_action = None

        # Initialize rollout context
        self.rollout = WorldModelEvaluator(world_model)

    # <--------------------------------- build spaces --------------------------------->

    def _build_observation_space(self) -> spaces.Dict:
        '''Builds the observation space based on the observed state dimensions.'''
        if self.world_model.visual:
            assert 'obs' in self.world_model.state_dims, \
                'Visual world model must have \'obs\' in state dims.'
            assert len(self.world_model.state_dims) == 1, \
                'Visual world model must only have \'obs\' in state dims.'
            return spaces.Dict({
                k: spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
                for k, shape in self.world_model.state_dims.items()
            })
        else:
            return spaces.Dict({
                k: spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
                for k, shape in self.world_model.state_dims.items()
            })

    def _build_action_space(self, min_action: float, max_action: float
                            ) -> Tuple[spaces.Space, List[Dict[str, Any]], str]:
        '''Builds the action space based on the world model's action dimensions.'''
        # validate discrete action keys
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
                    raise ValueError(f'Discrete key must be scalar, got {key}: {shape}')
                n = int(max_action - min_action + 1)
                if n < 1:
                    raise ValueError('max_action must be >= min_action.')
                ns.append(n)
                starts.append(int(min_action))
                action_specs.append({
                    'key': key, 'shape': shape, 'is_discrete': True,
                    'discrete_n': n, 'discrete_start': int(min_action),
                })

            # if only one discrete action key use Discrete, otherwise use MultiDiscrete
            if len(ns) == 1:
                action_space = spaces.Discrete(n=ns[0])
                mode = 'single_discrete'
            else:
                action_space = spaces.MultiDiscrete(ns)
                mode = 'multi_discrete'
        
        # if no discrete action keys, use a single Box for all continuous actions
        elif not self.discrete_action_keys:

            # store flat start and end indices for each action key to decode later
            offset = 0
            for key, shape in action_dims.items():
                n_elem = int(np.prod(shape, dtype=np.int64))
                action_specs.append({
                    'key': key, 'shape': shape, 'is_discrete': False,
                    'flat_start': offset, 'flat_end': offset + n_elem,
                })
                offset += n_elem
            action_space = spaces.Box(
                low=min_action, high=max_action, shape=(offset,), dtype=np.float32)
            mode = 'continuous'
        
        # mixed discrete and continuous action keys are not supported
        else:
            raise ValueError('Mixed discrete/continuous action keys are not supported.')

        return action_space, action_specs, mode

    def _prepare_initial_state(self, initial_state: Union[TensorDict, ArrayDict]
                               ) -> TensorDict:
        '''Prepares the initial state by converting to tensors.'''
        result = {}
        for key in self.world_model.state_dims:
            if key not in initial_state:
                raise ValueError(f'Missing initial state for key: {key}')
            value = initial_state[key]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()
            result[key] = value.to(self.device)[None, None]   # (1, 1, state_dims...)
        return result
    
    @property
    def enumerated_actions(self) -> List:
        '''Return a list of all discrete actions from this environment's action space.'''
        action_space = self.action_space
        if isinstance(action_space, spaces.Discrete):
            return list(range(int(action_space.n)))
        elif isinstance(action_space, spaces.MultiDiscrete):
            return list(itertools.product(*[range(int(n)) for n in action_space.nvec]))
        else:
            raise ValueError(f'Cannot enumerate {type(action_space).__name__}.')

    # <----------------------------------- sampling ----------------------------------->

    def obs_to_state_dict(self, obs: Array) -> ArrayDict:
        '''Split a flat real-env observation into a {key: array} state dict.'''
        state_dict = {}
        offset = 0
        for key, space in self._observation_space.spaces.items():
            n = int(np.prod(space.shape, dtype=np.int64))
            state_dict[key] = obs[offset:offset + n].reshape(space.shape)
            offset += n
        return state_dict
    
    def decode_action(self, action) -> Tuple[TensorDict, ArrayDict]:
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
            flat_act = np.asarray(action, dtype=np.float32).reshape(-1)
            for spec in self._action_specs:
                array_np = flat_act[spec['flat_start']:spec['flat_end']].reshape(spec['shape'])
                action_dict[spec['key']] = torch.from_numpy(array_np).to(self.device)[None]
                action_dict_np[spec['key']] = array_np

        return action_dict, action_dict_np

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple:
        '''Resets the environment with a new initial state.'''
        super().reset(seed=seed)
        rollout = self.rollout

        # reset the rollout context with the initial state and action
        rollout.reset(self.initial_state, self.initial_action)
        self.step_num = 0

        # extract the current states as numpy arrays for reward calculation
        self.states_np = {k: v[0] for k, v in rollout.last_states(to_numpy=True).items()}
        obs = spaces.flatten(self._observation_space, self.states_np)
        return obs, {}

    def step(self, action: Array) -> Tuple:
        '''Performs one step in the environment.'''
        action_dict, action_dict_np = self.decode_action(action)
        
        # use rollout context to predict next state
        prev_states_np = self.states_np
        self.states_np = {
            k: v[0].detach().cpu().numpy() 
            for k, v in self.rollout.step(action_dict).items()
        }
        obs = spaces.flatten(self._observation_space, self.states_np)
        
        # use reward function to evaluate reward
        reward = self.reward_fn(prev_states_np, action_dict_np, self.states_np)
        self.step_num += 1
        trunc = self.step_num >= self.max_steps
        return obs, reward, False, trunc, {}

    def render(self) -> None:
        '''Renders the environment (not implemented).'''
        pass
    