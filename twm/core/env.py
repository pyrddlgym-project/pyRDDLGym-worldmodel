import itertools
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Callable, Dict, List, Optional, Tuple

from twm.core.model import WorldModel, WorldModelEvaluator
from twm.core.spec import FluentSpec

Array = np.ndarray
ArrayDict = Dict[str, Array]
Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]


class WorldModelEnv(gym.Env):
    '''A gymnasium environment that uses a world model for state transitions.'''

    def __init__(self, world_model: WorldModel, reward_fn: Callable,
                 initial_state: ArrayDict | TensorDict, max_steps: int) -> None:
        '''Initializes the environment with a world model and initial state.'''
        super().__init__()

        self.world_model = world_model
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.device = world_model.device

        # build the observation space from spec
        self.observation_space = spaces.Dict({
            k: self.build_space_from_spec(spec)
            for k, spec in self.world_model.env_spec.state_spec.items()
        })

        # build the action space from spec
        self.action_space = spaces.Dict({
            k: self.build_space_from_spec(spec)
            for k, spec in self.world_model.env_spec.action_spec.items()
        })

        # prepare the initial state and evaluator
        self.set_initial_state(initial_state)
        self.rollout = WorldModelEvaluator(world_model)

    # <--------------------------------- build spaces --------------------------------->

    @staticmethod
    def build_space_from_spec(spec: FluentSpec) -> spaces.Space:
        '''Builds a gym space from a FluentSpec.'''
        # images are always Box with values in [0, 1]
        if spec.prange == 'pixel':
            return spaces.Box(low=0.0, high=1.0, shape=spec.shape, dtype=np.float32)

        # real variables use the ranges provided in spec
        if spec.prange == 'real':
            low, high = (-np.inf, np.inf) if spec.values is None else spec.values
            return spaces.Box(low=low, high=high, shape=spec.shape, dtype=np.float32)

        # integer variables must have specified values
        if spec.prange == 'int':
            if spec.values is None:
                raise ValueError('Integer variable must have specified values.')
            n = len(spec.values)
            if n < 1:
                raise ValueError('Integer variable must provide at least one value.')
            if spec.size == 1:
                return spaces.Discrete(n)
            else:
                return spaces.MultiDiscrete(np.full(spec.shape, n, dtype=np.int64))

        # boolean variables are represented as Discrete(2) or MultiBinary
        if spec.prange == 'bool':
            if spec.size == 1:
                return spaces.Discrete(2)
            else:
                return spaces.MultiBinary(spec.shape)

        raise ValueError(f'Unknown prange: {spec.prange}')
        
    def set_initial_state(self, initial_state: ArrayDict | TensorDict) -> None:
        '''Sets the initial state of the environment.'''
        result = {}
        for key in self.world_model.env_spec.state_spec:
            value = initial_state[key]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()
            result[key] = value.to(self.device)[None, None]   # (1, 1, state_dims...)
        self.initial_state = result
        self.initial_action = None
    
    # <----------------------------------- sampling ----------------------------------->
    
    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict]=None) -> Tuple:
        '''Resets the environment with a new initial state.'''
        super().reset(seed=seed)
        self.rollout.reset(self.initial_state, self.initial_action)
        self.obs = {k: v[0] for k, v in self.rollout.last_states(to_numpy=True).items()}
        self.step_num = 0
        return self.obs, {}

    def decode_action(self, action: ArrayDict) -> ArrayDict:
        '''Maps a gym action dict to the action values expected by the world model.'''
        decoded = {}
        for key, spec in self.world_model.env_spec.action_spec.items():

            # decode the raw action value from the dict and ensure shape
            raw = np.asarray(action[key])
            if raw.shape == () and spec.size == 1:
                raw = raw.reshape(spec.shape)
            if raw.shape != spec.shape:
                raise ValueError(f'Action {key} has shape {raw.shape} != {spec.shape}.')

            # decode int action according to prange
            if spec.prange == 'int':
                values = np.asarray(spec.values)
                idx = raw.astype(np.int64)
                if np.any((idx < 0) | (idx >= len(values))):
                    raise ValueError(f'Action {key} index out of range.')
                decoded[key] = values[idx].astype(np.float32)
            
            # decode bool action
            elif spec.prange == 'bool':
                arr = raw.astype(np.float32)
                if np.any((arr != 0.0) & (arr != 1.0)):
                    raise ValueError(f'Boolean action {key} must contain only 0/1.')
                decoded[key] = arr
            
            # decode real or pixel action (assumed to already be in correct range)
            elif spec.prange in ('real', 'pixel'):
                decoded[key] = raw.astype(np.float32)
            
            else:
                raise ValueError(f'Unknown prange for action {key}: {spec.prange}')

        return decoded

    def step(self, action: ArrayDict) -> Tuple:
        '''Performs one step in the environment.'''
        # decode the action dict to get the raw action values
        action_np = self.decode_action(action)
        action_tensor = {
            k: torch.from_numpy(v).float().to(self.device)[None]
            for k, v in action_np.items()
        }
        
        # use rollout context to predict next state
        next_obs = {
            k: v[0].detach().cpu().numpy() 
            for k, v in self.rollout.step(action_tensor).items()
        }
        reward = self.reward_fn(self.obs, action_np, next_obs)
        self.obs = next_obs
        self.step_num += 1
        trunc = self.step_num >= self.max_steps
        return self.obs, reward, False, trunc, {}

    def render(self) -> None:
        '''Renders the environment (not implemented).'''
        pass


class DiscreteActionWrapper(gym.ActionWrapper):
    '''Expose a single Discrete action and map it to action dicts.'''

    def __init__(self, env: WorldModelEnv) -> None:
        super().__init__(env)

        self._action_lut = self.build_action_lut(env)
        self.action_space = spaces.Discrete(len(self._action_lut))

    @staticmethod
    def build_action_lut(env: WorldModelEnv) -> List[ArrayDict]:
        '''Enumerates all valid finite joint actions from the action spec.'''
        keys, per_key_choices = [], []

        for key, spec in env.world_model.env_spec.action_spec.items():
            
            # determine the number of discrete choices for this action key
            if spec.prange == 'int':
                if spec.values is None:
                    raise ValueError(f'Int action {key} must define values.')
                n = len(spec.values)

            # boolean actions have 2 choices (0 or 1)
            elif spec.prange == 'bool':
                n = 2
            
            # real and pixel actions are not discrete and cannot be enumerated
            else:
                raise ValueError(
                    f'Only supports discrete actions, got {key} of range {spec.prange}.')

            # generate the discrete choices for this action key
            size, shape = spec.size, spec.shape
            if size == 1:
                choices = [np.asarray(i).reshape(shape) for i in range(n)]
            else:
                flat_grids = np.meshgrid(*[np.arange(n)] * size, indexing='ij')
                flat_grids = np.array(flat_grids).reshape(size, -1).T
                choices = [g.reshape(shape).astype(np.int64) for g in flat_grids]
            
            keys.append(key)
            per_key_choices.append(choices)

        # build the joint action LUT by taking the Cartesian product of per-key choices
        action_lut = []
        for joint in itertools.product(*per_key_choices):
            action_lut.append({
                k: v.item() if np.asarray(v).shape == () else v
                for k, v in zip(keys, joint)
            })

        if not action_lut:
            raise ValueError('Action spec produced no valid actions.')
        return action_lut

    def action(self, action: int) -> ArrayDict:
        '''Maps a discrete action index to the corresponding action dict from the LUT.'''
        idx = int(action)
        if idx < 0 or idx >= len(self._action_lut):
            raise ValueError(
                f'Action index {idx} out of bounds [0, {len(self._action_lut) - 1}].')
        return self._action_lut[idx]
