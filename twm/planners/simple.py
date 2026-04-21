import gymnasium as gym
import numpy as np
import os
import torch
from tqdm import tqdm
from typing import Tuple

from twm.core.data import PLOTS_PATH
from twm.core.env import WorldModelEnv


class SimpleMPC:
    '''Model-predictive control using a WorldModelEnv for action selection.'''

    def __init__(self, rollout_env: WorldModelEnv, real_env: gym.Env, 
                 lookahead: int, num_parallel_evals: int=32) -> None:
        self.rollout_env = rollout_env
        self.real_env = real_env
        self.lookahead = lookahead
        self.num_parallel_evals = int(num_parallel_evals)
        self.actions = rollout_env.enumerated_actions

        self._obs_history = []     
        self._action_history = []  

    def _build_init_tensors(self):
        '''Build init_states and init_actions tensors from the history buffers.'''
        wm = self.rollout_env
        device = wm.device

        # cap to seq_len most-recent observations (and corresponding actions)
        seq_len = wm.world_model.seq_len
        obs_hist = self._obs_history[-seq_len:]
        act_hist = self._action_history[-(seq_len - 1):] if self._action_history else []

        # build init_states
        init_states = {}
        for key in wm._observation_space.spaces:
            stacked = np.stack([wm.obs_to_state_dict(o)[key] for o in obs_hist], axis=0)
            init_states[key] = torch.from_numpy(stacked).float().unsqueeze(0).to(device)
            
        # build init_actions: T-1 real actions + one zero pad
        init_actions = {}
        for key, shape in wm.world_model.action_dims.items():
            if act_hist:
                decoded = np.stack([wm.decode_action(a)[1][key] for a in act_hist], axis=0)
            else:
                decoded = np.zeros((0, *shape), dtype=np.float32)
            zero_pad = np.zeros((1, *shape), dtype=np.float32)
            combined = np.concatenate([decoded, zero_pad], axis=0)
            init_actions[key] = torch.from_numpy(combined).float().unsqueeze(0).to(device)
            
        # repeat tensors along the batch dimension for parallel rollouts
        init_states = self._repeat_tensor_dict(init_states, self.num_parallel_evals)
        init_actions = self._repeat_tensor_dict(init_actions, self.num_parallel_evals)
        return init_states, init_actions

    @staticmethod
    def _repeat_tensor_dict(tensors, batch_size):
        '''Repeat a batched tensor dict along the batch dimension.'''
        if batch_size == 1:
            return tensors
        return {
            k: v.repeat(batch_size, *([1] * (v.ndim - 1)))
            for k, v in tensors.items()
        }

    def _decode_action_batch(self, actions):
        '''Decode a list of gym actions into batched model tensors and numpy arrays.'''
        decoded = [self.rollout_env.decode_action(action) for action in actions]
        action_tensors, action_arrays = {}, {}
        for key in self.rollout_env.world_model.action_dims:
            action_tensors[key] = torch.cat([item[0][key] for item in decoded], dim=0)
            action_arrays[key] = np.stack([item[1][key] for item in decoded], axis=0)
        return action_tensors, action_arrays

    def _batched_reward(self, prev_states_np, action_dict_np, states_np):
        '''Evaluate the reward function independently for each rollout in the batch.'''
        rewards = np.empty(self.num_parallel_evals, dtype=np.float32)
        for i in range(self.num_parallel_evals):
            rewards[i] = float(self.rollout_env.reward_fn(
                {k: v[i] for k, v in prev_states_np.items()},
                {k: v[i] for k, v in action_dict_np.items()},
                {k: v[i] for k, v in states_np.items()},
            ))
        return rewards

    def _align_world_model(self):
        '''Seed the WorldModelEnv's RolloutContext with the full real-env history.'''
        wm = self.rollout_env
        init_states, init_actions = self._build_init_tensors()

        # bypass WorldModelEnv.reset() and set the rollout context directly
        wm.rollout.reset(init_states, init_actions)
        wm.step_num = 0
        wm.states_np = wm.obs_to_state_dict(self._obs_history[-1])

    def _estimate_action_return(self, action):
        '''Estimate an action value with one or more rollout continuations.'''
        wm = self.rollout_env
        self._align_world_model()

        actions = [action] * self.num_parallel_evals
        returns = np.zeros(self.num_parallel_evals, dtype=np.float32)
        horizon = min(self.lookahead, wm.max_steps)

        for step_idx in range(horizon):

            # get the batch of last observations from the rollout history
            prev_states_np = wm.rollout.last_states(to_numpy=True)
        
            # apply a batched step in the world model with the current batch of actions
            action_dict, action_dict_np = self._decode_action_batch(actions)
            next_states_np = {
                k: v.detach().cpu().numpy()
                for k, v in wm.rollout.step(action_dict).items()
            }
            returns += self._batched_reward(prev_states_np, action_dict_np, next_states_np)

            if step_idx == horizon - 1:
                break
            
            # for the next step, sample a new batch of random actions
            actions = [wm.action_space.sample() for _ in range(self.num_parallel_evals)]

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
