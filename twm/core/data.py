import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from PIL import Image
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Callable, Dict, Generator, List, Tuple

from twm.core.spec import EnvSpec

Array = np.ndarray
ArrayDict = Dict[str, Array]
Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]

PARENT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(PARENT_PATH, 'data')
PLOTS_PATH = os.path.join(PARENT_PATH, 'plots')


# <------------------------------- Data collection  ------------------------------>

def image_to_tensor(img: Image.Image, size: Tuple[int, int]=(64, 64)) -> Array:
    '''Converts a PIL image into a normalized tensor.'''
    x = img.resize(size, Image.Resampling.BILINEAR)
    x = np.asarray(x, dtype=np.float16) / 255.0
    return np.transpose(x, (2, 0, 1))


def _dict_append(src: Dict[str, Any], dest: Dict[str, List[Any]]) -> None:
    '''Appends src arrays to dest arrays along the first dimension.'''
    for key, value in src.items():
        dest.setdefault(key, []).append(np.asarray(value))


def _create_obs(state, env, env_spec, image_map):
    '''Creates an observation dict from the environment state.'''
    obs = {}
    for key, spec in env_spec.state_spec.items():
        if spec.prange == 'pixel':
            obs[key] = image_map(env.render(), spec.shape[1:])
        else:
            obs[key] = state[key]
    return obs


def _create_action(action, env_spec):
    '''Creates an action dict from the policy action.'''
    return {k: action[k] for k in env_spec.action_spec}


def create_data(env: gym.Env, env_spec: EnvSpec, policy, episodes: int, max_steps: int, 
                data_name: str, image_map: Callable=image_to_tensor) -> None:
    '''Collects data by running a policy in an environment and saves it to file.'''
    states, actions, next_states, rewards, dones = {}, {}, {}, [], []
    
    for _ in (pbar := tqdm(range(episodes), desc='Collecting data')):
        state, _ = env.reset()
        obs = _create_obs(state, env, env_spec, image_map)
        total_reward = 0.

        for step in range(max_steps):
            action = policy.sample_action(state)
            action_obs = _create_action(action, env_spec)
            next_state, reward, term, trunc, _ = env.step(action)
            next_obs = _create_obs(next_state, env, env_spec, image_map)
            total_reward += reward
            done = term or trunc or step == max_steps - 1
            
            _dict_append(obs, states)
            _dict_append(action_obs, actions)
            _dict_append(next_obs, next_states)
            rewards.append(float(reward))
            dones.append(done)
            
            state, obs = next_state, next_obs
            if done:
                break
        pbar.set_postfix({'Episode Reward': total_reward})
    
    # save data
    data = {
        'states':      {k: np.asarray(v) for k, v in states.items()},
        'actions':     {k: np.asarray(v) for k, v in actions.items()},
        'next_states': {k: np.asarray(v) for k, v in next_states.items()},
        'rewards':     np.array(rewards),
        'dones':       np.array(dones),
        'spec':        env_spec,
    }
    with open(os.path.join(DATA_PATH, data_name), 'wb') as f:
        pickle.dump(data, f)


# <------------------------------- Data preparation  ------------------------------>

def load_episodic_data(data_name: str) -> Generator[Dict[str, Any], None, None]:
    '''Loads data from a file and yields it as individual episode dicts.'''
    with open(os.path.join(DATA_PATH, data_name), 'rb') as f:
        data = pickle.load(f)

    # find episode boundaries based on termination flags
    ep_ends = np.where(data['dones'])[0] + 1
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    
    # split data into episodes and yield as dicts
    for start, end in zip(ep_starts, ep_ends):
        dones_ep = data['dones'][start:end]  
        assert dones_ep[-1] and np.all(~dones_ep[:-1])
        yield {
            'states':      {k: v[start:end] for k, v in data['states'].items()},
            'actions':     {k: v[start:end] for k, v in data['actions'].items()},
            'next_states': {k: v[start:end] for k, v in data['next_states'].items()},
            'rewards':     data['rewards'][start:end],
            'dones':       dones_ep,
            'len':         end - start,
            'spec':        data['spec'],
        }


class SequenceDataset(torch.utils.data.Dataset):
    '''A PyTorch Dataset that takes episodic data and returns padded sequences of a 
    specified length.'''
    
    def __init__(self, episodes: List[Dict[str, Any]], seq_len: int, 
                 augment_starts: bool=False, min_frames: int=2) -> None:
        self.seq_len = seq_len
        self.augment_starts = augment_starts
        self.min_frames = max(1, int(min_frames))
        self.env_spec = episodes[0]['spec']
              
        self._episodes, self._index = self.make_episode_index(episodes)
        self.normalizer_stats = self.init_stats(self._episodes)

    # <------------------------------- Data preparation  ------------------------------>

    @staticmethod
    def make_episode_index(episodes: List[Dict[str, Any]]) -> Tuple:
        '''Builds a flat (ep_idx, t) sample index to index into the dataset.'''
        new_episodes = list(episodes)
        index = []
        for ep_idx, ep in enumerate(new_episodes):
            for t in range(ep['len']):
                index.append((ep_idx, t))
        return new_episodes, index

    @staticmethod
    def init_stats(episodes: List[Dict[str, Any]]) -> Dict[str, Tuple[Tensor, Tensor]]:
        '''Calculates mean and std for states and actions across the entire dataset.'''
        # gather all state and action values across episodes into lists per key
        values = {}
        for ep in episodes:
            _dict_append(ep['next_states'], values)
            _dict_append(ep['actions'], values)

        # calculate data set stats for state
        stats = {}
        for key, spec in episodes[0]['spec'].all_spec.items():
            if spec.prange == 'real':
                all_vals = torch.as_tensor(np.concatenate(values[key], axis=0)).float()
                mean = all_vals.mean(dim=0).reshape(spec.shape)
                std = all_vals.std(dim=0).clamp(min=1e-8).reshape(spec.shape)
                stats[key] = (mean, std)
        return stats

    # <------------------------------- Data sampling  ------------------------------>

    def make_padded(self, x: Array, t: int) -> Tuple[Array, int]:
        '''Returns a padded sequence of length seq_len ending at time t, along with the 
        pad length.'''
        start = max(0, t - self.seq_len + 1)
        hist = x[start:t + 1]
        pad_len = self.seq_len - hist.shape[0]
        pad = np.zeros((pad_len,) + x.shape[1:], dtype=x.dtype)
        new_x = np.concatenate([hist, pad], axis=0)
        return new_x, pad_len

    @staticmethod
    def increase_pad(x: Array, old_pad: int, new_pad: int) -> Array:
        '''Increases the padding of a sequence tensor from old_pad to new_pad.'''
        seq_len = x.shape[0]
        old_real = seq_len - old_pad
        new_real = seq_len - new_pad
        y = np.zeros_like(x)
        y[:new_real] = x[old_real - new_real:old_real]
        return y

    def __len__(self) -> int:
        '''Returns the total number of samples in the dataset.'''
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Tensor | TensorDict]:
        '''Returns a padded sequence sample from the dataset at the given index.'''
        ep_idx, t = self._index[idx]
        ep = self._episodes[ep_idx]
        T = self.seq_len
        
        # get padded sequences of states, actions, rewards, dones, and next_states
        states      = {k: self.make_padded(v, t)[0] for k, v in ep['states'].items()}
        actions     = {k: self.make_padded(v, t)[0] for k, v in ep['actions'].items()}            
        next_states = {k: v[t] for k, v in ep['next_states'].items()}
        rewards, _  = self.make_padded(ep['rewards'], t)
        dones, pad  = self.make_padded(ep['dones'], t)
        
        # HER-style: pick a random virtual start in [current_pad_len, seq_len-1]
        if self.augment_starts:
            seq_len = rewards.shape[0]
            max_pad = max(seq_len - self.min_frames, pad)
            new_pad = np.random.randint(pad, max_pad + 1)
            if new_pad > pad:
                states  = {k: self.increase_pad(v, pad, new_pad) for k, v in states.items()}
                actions = {k: self.increase_pad(v, pad, new_pad) for k, v in actions.items()}
                rewards = self.increase_pad(rewards, pad, new_pad)
                dones   = self.increase_pad(dones, pad, new_pad)
                pad     = new_pad

        # convert to PyTorch tensors and return
        return {
            'states':      {k: torch.as_tensor(v) for k, v in states.items()},       
            # (seq_len, state_dims...)
            'actions':     {k: torch.as_tensor(v) for k, v in actions.items()},      
            # (seq_len, action_dims...)
            'next_states': {k: torch.as_tensor(v) for k, v in next_states.items()},  
            # (state_dims...)
            'rewards':     torch.as_tensor(rewards).float(),   # (seq_len,)
            'dones':       torch.as_tensor(dones).bool(),      # (seq_len,)
            'pad':         torch.as_tensor(pad).long(),        # ()
        }
    

def get_dataloader(data_name: str, seq_len: int, batch_size: int=64, 
                   test_split: float=0.1, seed: int=0, **dataset_kwargs
                   ) -> Tuple[DataLoader, DataLoader]:
    '''Loads episodic data from a file, splits into train/test sets.'''
    # load episodic data
    episodes = list(load_episodic_data(data_name))
    n_episodes = len(episodes)

    # split into train/test sets
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_test = min(max(1, int(round(test_split * n_episodes))), n_episodes - 1)
    train_eps = [episodes[i] for i in perm[n_test:]]
    test_eps = [episodes[i] for i in perm[:n_test]]

    # create PyTorch Datasets and DataLoaders
    train_set = SequenceDataset(train_eps, seq_len, **dataset_kwargs)
    test_set = SequenceDataset(test_eps, seq_len, **dataset_kwargs)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader


# <------------------------------- Data visualization  ------------------------------>

def plot_trajectories(trajectories: List[TensorDict], plot_name: str) -> None:
    '''Plots trajectories of state variables over time and saves the plot to a file.'''
    n_states = sum(
        int(np.prod(v.shape[1:], dtype=np.int64)) 
        for v in trajectories[0].values()
    )
    fig, axs = plt.subplots(1, n_states, figsize=(4 * n_states, 4), squeeze=False)
    axs = axs.flat

    for traj in trajectories:
        idx = 0
        for key, value in traj.items():
            if isinstance(value, Tensor):
                value = value.detach().cpu().numpy()
            value = value.reshape(value.shape[0], -1)
            for j in range(value.shape[1]):
                axs[idx].plot(value[:, j])
                idx += 1

    plt.savefig(os.path.join(PLOTS_PATH, plot_name))
    plt.close(fig)


def plot_data_trajectories(data_name: str, limit: int, plot_name: str) -> None:
    '''Plots trajectories of state variables over time from a data file.'''
    episode_data = list(load_episodic_data(data_name))
    random.shuffle(episode_data)
    trajectories = [episode_data[i]['states'] for i in range(limit)]
    plot_trajectories(trajectories, plot_name)


def save_video(render_fn: Callable, trajectories: List[TensorDict], plot_name: str) -> None:
    '''Saves a video of trajectories rendered by a provided function.'''
    frames = []
    for trajectory in trajectories:
        trajectory_np = {}
        for key, value in trajectory.items():
            if isinstance(value, Tensor):
                value = value.detach().cpu().numpy()
            trajectory_np[key] = value

        n_steps = next(iter(trajectory_np.values())).shape[0]
        for i in range(n_steps):
            state = {k: v[i] for k, v in trajectory_np.items()}
            frames.append(render_fn(state))
    
    frames[0].save(
        fp=os.path.join(PLOTS_PATH, plot_name), 
        format='GIF', append_images=frames, save_all=True, duration=100
    )
