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

    @staticmethod
    def build_space_from_spec(spec: FluentSpec) -> spaces.Space:
        '''Builds a gym space from a FluentSpec.'''
        # images are always Box with values in [0, 1]
        if spec.prange == 'pixel':
            return spaces.Box(low=0.0, high=1.0, shape=spec.shape, dtype=np.float32)

        # real variables use the ranges provided in spec
        elif spec.prange == 'real':
            low, high = (-np.inf, np.inf) if spec.values is None else spec.values
            return spaces.Box(low=low, high=high, shape=spec.shape, dtype=np.float32)

        # integer variables must have specified values
        elif spec.prange == 'int':
            if spec.values is None or len(spec.values) != 2:
                raise ValueError('Integer variable must have low and high values.')
            low, high = int(spec.values[0]), int(spec.values[1])
            n = high - low + 1
            if spec.size == 1:
                return spaces.Discrete(n, start=low)
            else:
                return spaces.MultiDiscrete(
                    np.full(spec.shape, n, dtype=np.int64), 
                    start=np.full(spec.shape, low, dtype=np.int64)
                )

        # boolean variables are represented as Discrete(2) or MultiBinary
        elif spec.prange == 'bool':
            if spec.size == 1:
                return spaces.Discrete(2)
            else:
                return spaces.MultiBinary(spec.shape)

        else:
            raise ValueError(f'Unknown prange {spec.prange}.')
        
    def set_initial_state(self, initial_state: ArrayDict | TensorDict) -> None:
        '''Sets the initial state of the environment.'''
        result = {}
        for key in self.world_model.env_spec.state_spec:
            value = torch.as_tensor(initial_state[key]).float().to(self.device) 
            result[key] = value[None, None]   # (1, 1, state_dims...)
        self.initial_state = result
        self.initial_action = None

    # <----------------------------------- sampling ----------------------------------->
    
    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict]=None) -> Tuple:
        '''Resets the environment with a new initial state.'''
        super().reset(seed=seed)
        self.rollout.reset(self.initial_state, self.initial_action)
        self.obs = self.rollout.last_states(to_numpy=False, squash=True)
        self.step_num = 0
        return self.obs, {}

    def step(self, action: ArrayDict) -> Tuple:
        '''Performs one step in the environment.'''
        action_torch = {k: torch.as_tensor(v).to(self.device) for k, v in action.items()}
        action_batched = {k: v[None] for k, v in action_torch.items()}
        self.rollout.step(action_batched)
        next_obs = self.rollout.last_states(to_numpy=False, squash=True)
        reward = self.reward_fn(self.obs, action_torch, next_obs)
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
            if spec.prange in ('int', 'bool'):
                values = spec.values
                if spec.prange == 'bool':
                    values = (0, 1)
                if values is None or len(values) != 2:
                    raise ValueError(f'Int action {key} must have values.')
                low, high = int(values[0]), int(values[1])
                choice = np.arange(low, high + 1, dtype=np.int64)
            else:
                raise ValueError(
                    f'Only supports discrete actions, got {key} of range {spec.prange}.')

            # generate the discrete choices for this action key
            size, shape = spec.size, spec.shape
            if size == 1:
                choices = [np.full(shape, i) for i in choice]
            else:
                flat_grids = np.meshgrid(*[choice] * size, indexing='ij')
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
        n = len(self._action_lut)
        if idx < 0 or idx >= n:
            raise ValueError(f'Action index {idx} out of bounds [0, {n - 1}].')
        return self._action_lut[idx]
