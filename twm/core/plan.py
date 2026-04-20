from gymnasium import spaces
import itertools
import numpy as np
import os
import torch
from typing import List
from tqdm import tqdm

from twm.core.data import PLOTS_PATH


class SimpleMPC:
    """Model-predictive control using a WorldModelEnv for action selection."""

    def __init__(self, rollout_env, real_env, lookahead: int) -> None:
        self.rollout_env = rollout_env
        self.real_env = real_env
        self.lookahead = lookahead
        self.actions = self._enumerate_actions(rollout_env.action_space)

        self._obs_history = []     
        self._action_history = []  

    @staticmethod
    def _enumerate_actions(action_space) -> List:
        """Return a list of all discrete actions from a Discrete or MultiDiscrete space."""
        if isinstance(action_space, spaces.Discrete):
            return list(range(int(action_space.n)))
        if isinstance(action_space, spaces.MultiDiscrete):
            return list(itertools.product(*[range(int(n)) for n in action_space.nvec]))
        raise ValueError(f"Cannot enumerate action space {type(action_space).__name__}.")

    def _obs_to_state_dict(self, obs: np.ndarray) -> dict:
        """Split a flat real-env observation into a {key: array} state dict."""
        state_dict = {}
        offset = 0
        for key, space in self.rollout_env._observation_space.spaces.items():
            n = int(np.prod(space.shape))
            state_dict[key] = obs[offset:offset + n].reshape(space.shape)
            offset += n
        return state_dict

    def _build_init_tensors(self):
        """Build init_states and init_actions tensors from the history buffers."""
        wm = self.rollout_env
        seq_len = wm.world_model.seq_len
        device = wm.device

        # cap to seq_len most-recent observations (and corresponding actions)
        obs_hist = self._obs_history[-seq_len:]
        act_hist = self._action_history[-(seq_len - 1):] if self._action_history else []

        # build init_states
        init_states = {}
        for key in wm._observation_space.spaces:
            stacked = np.stack([self._obs_to_state_dict(o)[key] for o in obs_hist], axis=0)
            init_states[key] = torch.from_numpy(stacked).float().unsqueeze(0).to(device)
            
        # build init_actions: T-1 real actions + one zero pad
        init_actions = {}
        for key, shape in wm.world_model.action_dims.items():
            if act_hist:
                decoded = np.stack(
                    [self.rollout_env._decode_action(a)[1][key] for a in act_hist], axis=0)
            else:
                decoded = np.zeros((0, *shape), dtype=np.float32)
            zero_pad = np.zeros((1, *shape), dtype=np.float32)
            combined = np.concatenate([decoded, zero_pad], axis=0)
            init_actions[key] = torch.from_numpy(combined).float().unsqueeze(0).to(device)
            
        return init_states, init_actions

    def _align_world_model(self) -> None:
        """Seed the WorldModelEnv's RolloutContext with the full real-env history."""
        wm = self.rollout_env
        init_states, init_actions = self._build_init_tensors()

        # bypass WorldModelEnv.reset() and set the rollout context directly
        wm.rollout.reset(init_states, init_actions)
        wm.step_num = 0
        wm.states_np = self._obs_to_state_dict(self._obs_history[-1])

    def select_action(self) -> int:
        """Return the best action via random-shooting in the world model."""
        best_action = self.actions[0]
        best_return = -np.inf

        for action in self.actions:

            # align the world model's rollout context with the real env's history before 
            # each candidate action
            self._align_world_model()

            # first step: candidate action
            _, reward, term, trunc, _ = self.rollout_env.step(action)
            total_return = float(reward)

            # remaining steps: uniform-random policy
            for _ in range(self.lookahead - 1):
                if term or trunc:
                    break
                rand_action = self.rollout_env.action_space.sample()
                _, reward, term, trunc, _ = self.rollout_env.step(rand_action)
                total_return += float(reward)

            if total_return > best_return:
                best_return = total_return
                best_action = action

        return best_action

    def reset(self) -> None:
        obs, _ = self.real_env.reset()
        self._obs_history = [obs]
        self._action_history = []
        self.obs = obs

        self.frames = []

    def step(self, save_frames: bool=True):
        action = self.select_action()
        obs, reward, term, trunc, info = self.real_env.step(action)
        if save_frames:
            self.frames.append(self.real_env.render())

        # update history before returning
        self._action_history.append(action)
        self._obs_history.append(obs)
        self.obs = obs

        return obs, reward, term, trunc, info

    def run(self, plot_name: str, max_steps: int=200, episodes: int=1, 
            save_frames: bool=True) -> float:
        """Run full episodes and return average reward."""
        avg = 0.0
        for _ in range(episodes):
            total = 0.0
            self.reset()
            for _ in (pbar := tqdm(range(max_steps), desc="Running MPC")):
                _, reward, term, trunc, _ = self.step(save_frames=save_frames)
                total += reward
                done = term or trunc
                if done:
                    break
                pbar.set_postfix({"Cuml Return": f"{total:.3f}"})
            avg += total / episodes
            
        if save_frames:
            self.frames[0].save(
                fp=os.path.join(PLOTS_PATH, plot_name), 
                format='GIF', append_images=self.frames[1:], save_all=True, duration=100)
            
        return avg
