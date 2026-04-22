import gymnasium as gym
import numpy as np
import os
import torch
from tqdm import tqdm
from typing import Tuple

from twm.core.data import PLOTS_PATH
from twm.core.env import DiscreteActionWrapper, WorldModelEnv


class RandomShootingMPC:
    '''Model-predictive controller for discrete action problems using random shooting.'''

    def __init__(self, rollout_env: WorldModelEnv, real_env: gym.Env, 
                 lookahead: int, num_parallel_evals: int=32) -> None:
        self.rollout_env = rollout_env
        self.real_env = real_env
        self.lookahead = lookahead
        self.num_parallel_evals = num_parallel_evals
        
        self.actions = DiscreteActionWrapper.build_action_lut(self.rollout_env)
        self._obs_history = []     
        self._action_history = []  

    def _repeat_tensor_dict(self, tensors):
        '''Repeat a batched tensor dict along the batch dimension.'''
        if self.num_parallel_evals == 1:
            return tensors
        return {
            k: v.repeat(self.num_parallel_evals, *([1] * (v.ndim - 1)))
            for k, v in tensors.items()
        }

    def _align_world_model(self):
        '''Seed the WorldModelEnv with the full real-env history.'''
        wm = self.rollout_env
        device = wm.device

        # cap to seq_len most-recent observations (and corresponding actions)
        seq_len = wm.world_model.seq_len
        obs_hist = self._obs_history[-seq_len:]
        act_hist = self._action_history[-(seq_len - 1):] if self._action_history else []

        # build init_states
        init_states = {}
        for key in wm.world_model.env_spec.state_spec:
            stacked = np.stack([obs[key] for obs in obs_hist], axis=0)
            init_states[key] = torch.from_numpy(stacked).float().to(device)[None]
            
        # build init_actions: T-1 real actions + one zero pad
        init_actions = {}
        for key, spec in wm.world_model.env_spec.action_spec.items():
            if act_hist:
                decoded = np.stack([wm.decode_action(a)[key] for a in act_hist], axis=0)
            else:
                decoded = np.zeros((0, *spec.shape), dtype=np.float32)
            zero_pad = np.zeros((1, *spec.shape), dtype=np.float32)
            combined = np.concatenate([decoded, zero_pad], axis=0)
            init_actions[key] = torch.from_numpy(combined).float().to(device)[None]
            
        # repeat tensors along the batch dimension for parallel rollouts
        init_states = self._repeat_tensor_dict(init_states)
        init_actions = self._repeat_tensor_dict(init_actions)

        # bypass WorldModelEnv and set the rollout context directly
        wm.rollout.reset(init_states, init_actions)
        wm.step_num = 0

    def _decode_action_batch(self, actions):
        '''Decode a list of gym actions into batched model tensors and numpy arrays.'''
        wm = self.rollout_env
        decoded_np = [wm.decode_action(action) for action in actions]
        action_tensors, action_arrays = {}, {}
        for key in wm.world_model.env_spec.action_spec:
            action = np.stack([item[key] for item in decoded_np], axis=0)
            action_arrays[key] = action
            action_tensors[key] = torch.from_numpy(action).float().to(wm.device)
        return action_tensors, action_arrays

    def _batched_reward(self, obs_dict, action_dict, next_obs_dict):
        '''Evaluate the reward function independently for each rollout in the batch.'''
        rewards = np.zeros(self.num_parallel_evals, dtype=np.float32)
        for i in range(self.num_parallel_evals):
            rewards[i] = float(self.rollout_env.reward_fn(
                {k: v[i] for k, v in obs_dict.items()},
                {k: v[i] for k, v in action_dict.items()},
                {k: v[i] for k, v in next_obs_dict.items()},
            ))
        return rewards

    def _estimate_action_return(self, action):
        '''Estimate an action value with one or more rollout continuations.'''
        # align the world model's rollout context with the real environment's history
        self._align_world_model()

        actions = [action] * self.num_parallel_evals
        returns = np.zeros(self.num_parallel_evals, dtype=np.float32)
        horizon = min(self.lookahead, self.rollout_env.max_steps)

        for step_idx in range(horizon):

            # get the batch of last observations from the rollout history
            obs_dict = self.rollout_env.rollout.last_states(to_numpy=True)
        
            # apply a batched step in the world model with the current batch of actions
            action_dict, action_dict_np = self._decode_action_batch(actions)
            next_obs_dict = {
                k: v.detach().cpu().numpy()
                for k, v in self.rollout_env.rollout.step(action_dict).items()
            }
            returns += self._batched_reward(obs_dict, action_dict_np, next_obs_dict)

            if step_idx == horizon - 1:
                break
            
            # for the next step, sample a new batch of random actions
            idx = np.random.randint(len(self.actions), size=self.num_parallel_evals)
            actions = [self.actions[i] for i in idx]

        return float(returns.mean())

    def _select_action(self):
        '''Return the best action via random-shooting in the world model.'''
        best_action = self.actions[0]
        best_return = -np.inf

        for action in self.actions:
            estimated_return = self._estimate_action_return(action)

            if estimated_return > best_return:
                best_return = estimated_return
                best_action = action

        return best_action

    def reset(self) -> None:
        '''Reset the real environment and clear the history buffers.'''
        obs, _ = self.real_env.reset()
        self._obs_history = [obs]
        self._action_history = []
        self.frames = []

    def step(self, save_frames: bool=True) -> Tuple:
        '''Take a step in the real environment using the selected action.'''
        action = self._select_action()
        obs, reward, term, trunc, info = self.real_env.step(action)

        if save_frames:
            self.frames.append(self.real_env.render())

        self._action_history.append(action)
        self._obs_history.append(obs)

        return obs, reward, term, trunc, info

    def run(self, plot_name: str, max_steps: int=200, episodes: int=1, 
            save_frames: bool=True) -> float:
        '''Run full episodes and return average reward.'''
        # run multiple episodes and average the total reward
        avg = 0.0
        for _ in range(episodes):
            total = 0.0
            self.reset()
            for _ in (pbar := tqdm(range(max_steps), desc='Running MPC')):
                _, reward, term, trunc, _ = self.step(save_frames=save_frames)
                total += reward
                done = term or trunc
                if done:
                    break
                pbar.set_postfix({'Cuml Return': f'{total:.3f}'})
            avg += total / episodes
            
        # save a GIF of the last episode if requested
        if save_frames:
            self.frames[0].save(
                fp=os.path.join(PLOTS_PATH, plot_name), 
                format='GIF', append_images=self.frames[1:], save_all=True, duration=100)
            
        return avg
