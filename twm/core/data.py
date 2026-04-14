import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from PIL import Image
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Dict, Generator, Tuple

Array = np.ndarray
ArrayDict = Dict[str, Array]


PARENT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(PARENT_PATH, 'data')
PLOTS_PATH = os.path.join(PARENT_PATH, 'plots')


# <------------------------------- Data collection  ------------------------------>

def dict_to_tensor(state_dict: ArrayDict) -> Array:
    '''Converts a dict of arrays into a single concatenated array.'''
    state_vec = []
    for value in state_dict.values():
        value = np.asarray(value, dtype=np.float32)
        state_vec.append(np.ravel(value))
    return np.concatenate(state_vec)


def image_to_tensor(img: Image.Image, size: Tuple[int, int]=(64, 64)) -> Array:
    '''Converts a PIL image into a normalized tensor.'''
    img = img.resize(size, Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float16) / 255.0
    return np.transpose(x, (2, 0, 1))


def create_vector_data(env, policy, episodes: int, max_steps: int, data_name: str, 
                       state_map: Callable[[ArrayDict], Array]=dict_to_tensor, 
                       action_map: Callable[[ArrayDict], Array]=dict_to_tensor, 
                       render: bool=False) -> None:
    '''Collects vector data by running a policy in an environment, and saves it to a file.'''
    episodes = int(episodes)
    max_steps = int(max_steps)

    # collect data
    states, actions, rewards, dones, next_states = [], [], [], [], []
    
    for _ in (pbar := tqdm(range(episodes), desc="Collecting data")):
        state, _ = env.reset()
        total_reward = 0.
        for step in range(max_steps):
            if render:
                env.render()

            action = policy.sample_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            done = term or trunc or step == max_steps - 1
            
            states.append(state_map(state))
            actions.append(action_map(action))
            rewards.append(float(reward))
            dones.append(done)
            next_states.append(state_map(next_state))
            
            state = next_state
            if done:
                break
        pbar.set_postfix({"Episode Reward": total_reward})
    
    # save data
    data = {
        "states":      np.array(states),
        "actions":     np.array(actions),
        "rewards":     np.array(rewards),
        "dones":       np.array(dones),
        "next_states": np.array(next_states),
    }
    
    with open(os.path.join(DATA_PATH, data_name), "wb") as f:
        pickle.dump(data, f)


def create_image_data(env, policy, episodes: int, max_steps: int, data_name: str, 
                      image_map: Callable[[Image.Image], Array]=image_to_tensor, 
                      action_map: Callable[[ArrayDict], Array]=dict_to_tensor) -> None:
    '''Collects image data by running a policy in an environment, and saves it to a file.
    Uses the environment's render function to get image observations.'''
    episodes = int(episodes)
    max_steps = int(max_steps)

    # collect data
    states, actions, rewards, dones, next_states = [], [], [], [], []
    
    for _ in (pbar := tqdm(range(episodes), desc="Collecting data")):
        state, _ = env.reset()
        obs = image_map(env.render())
        total_reward = 0.
        for step in range(max_steps):
            action = policy.sample_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            next_obs = image_map(env.render())
            total_reward += reward
            done = term or trunc or step == max_steps - 1

            states.append(obs)
            actions.append(action_map(action))
            rewards.append(float(reward))
            dones.append(done)
            next_states.append(next_obs)
            
            state, obs = next_state, next_obs
            if done:
                break
        pbar.set_postfix({"Episode Reward": total_reward})
    
    # save data
    data = {
        "states":      np.array(states),
        "actions":     np.array(actions),
        "rewards":     np.array(rewards),
        "dones":       np.array(dones),
        "next_states": np.array(next_states),
    }
    
    with open(os.path.join(DATA_PATH, data_name), "wb") as f:
        pickle.dump(data, f)


# <------------------------------- Data preparation  ------------------------------>

def load_episodic_data(data_name: str) -> Generator[ArrayDict, None, None]:
    '''Loads data from a file and yields it as individual episode dicts.'''
    with open(os.path.join(DATA_PATH, data_name), "rb") as f:
        data = pickle.load(f)

    # find episode boundaries based on dones
    ep_ends = np.where(data['dones'])[0] + 1
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    
    # split data into episodes
    for start, end in zip(ep_starts, ep_ends):
        states_ep = data['states'][start:end]
        actions_ep = data['actions'][start:end]
        rewards_ep = data['rewards'][start:end]
        dones_ep = data['dones'][start:end]  
        next_states_ep = data['next_states'][start:end] 
        assert dones_ep[-1] and np.all(~dones_ep[:-1])
        yield {
            "states":      states_ep,
            "actions":     actions_ep,
            "rewards":     rewards_ep,
            "dones":       dones_ep,
            "next_states": next_states_ep,
            'len':         end - start,
        }


class SequenceDataset(torch.utils.data.Dataset):
    '''A PyTorch Dataset that takes episodic data and returns padded sequences of a specified length.'''

    def __init__(self, episodes, seq_len: int, obs_idx=None,
                 augment_starts: bool=False, min_frames: int=2) -> None:
        self.seq_len = seq_len
        self.augment_starts = augment_starts
        self.min_frames = max(1, int(min_frames))
        self.obs_idx = np.array(obs_idx) if obs_idx is not None else None

        # store raw per-episode arrays and build a flat (ep_idx, t) sample index        
        self._episodes, self._index = self.init_episodes(episodes, obs_idx)
        sample_states = self._episodes[0]['states']
        if sample_states.ndim > 2:
            self.state_dim = tuple(sample_states.shape[1:])
        else:
            self.state_dim = sample_states.shape[1]
        self.action_dim = self._episodes[0]['actions'].shape[-1]

        # calculate data set stats for normalization
        self.state_mean, self.state_std, self.action_mean, self.action_std = self.init_stats(
            self._episodes)

    # <------------------------------- Data preparation  ------------------------------>

    @staticmethod
    def init_episodes(episodes, obs_idx=None):
        '''Processes raw episode data, optionally selecting a subset of state dimensions, and builds 
        a flat (ep_idx, t) sample index.'''
        new_episodes, index = [], []

        for ep_idx, ep in enumerate(episodes):
            states = ep['states']
            next_states = ep['next_states']

            # optionally select only a subset of state dimensions to use as model input/targets
            if obs_idx is not None:
                states = np.take(states, obs_idx, axis=1)
                next_states = np.take(next_states, obs_idx, axis=1)
            
            new_episodes.append({
                'states':      states,
                'actions':     ep['actions'],
                'rewards':     ep['rewards'],
                'dones':       ep['dones'],
                'next_states': next_states,
                'len':         ep['len'],
            })
            
            # build a flat (ep_idx, t) sample index for the dataset
            for t in range(ep['len']):
                index.append((ep_idx, t))

        return new_episodes, index

    @staticmethod
    def init_stats(episodes):
        '''Calculates mean and std for states and actions across the entire dataset, for normalization.'''
        states, actions = [], []
        for ep in episodes:
            states.append(ep['next_states'])
            actions.append(ep['actions'])

        # calculate data set stats for state
        states = torch.tensor(np.concatenate(states, axis=0), dtype=torch.float32)
        state_mean = states.mean(dim=0)
        state_std = states.std(dim=0).clamp(min=1e-8)

        # calculate data set stats for action
        actions = torch.tensor(np.concatenate(actions, axis=0), dtype=torch.float32)
        action_mean = actions.mean(dim=0)
        action_std = actions.std(dim=0).clamp(min=1e-8)

        return state_mean, state_std, action_mean, action_std

    # <------------------------------- Data sampling  ------------------------------>

    @staticmethod
    def make_padded(x: Array, t: int, seq_len: int) -> Tuple[Array, int]:
        '''Returns a padded sequence of length seq_len ending at time t, along with the pad length.'''
        start = max(0, t - seq_len + 1)
        hist = x[start:t + 1]
        pad_len = seq_len - hist.shape[0]
        pad = np.zeros((pad_len,) + x.shape[1:], dtype=x.dtype)
        new_x = np.concatenate([hist, pad], axis=0)
        return new_x, pad_len

    @staticmethod
    def increase_padding(x: Array, old_pad: int, new_pad: int) -> Array:
        '''Increases the padding of a sequence tensor from old_pad to new_pad.'''
        seq_len = x.shape[0]
        old_real = seq_len - old_pad
        new_real = seq_len - new_pad
        y = np.zeros_like(x)
        y[:new_real] = x[old_real - new_real:old_real]
        return y

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        ep_idx, t = self._index[idx]
        ep = self._episodes[ep_idx]
        
        # get padded sequences of states, actions, rewards, dones, and next_states
        states, pad = self.make_padded(ep['states'], t, self.seq_len)
        actions, _ = self.make_padded(ep['actions'], t, self.seq_len)
        rewards, _ = self.make_padded(ep['rewards'], t, self.seq_len)
        dones, _ = self.make_padded(ep['dones'], t, self.seq_len)
        next_states = ep['next_states'][t]

        # HER-style: pick a random virtual start in [current_pad_len, seq_len-1]
        if self.augment_starts:
            seq_len = states.shape[0]
            max_pad = max(seq_len - self.min_frames, pad)
            new_pad = np.random.randint(pad, max_pad + 1)
            if new_pad > pad:
                states = SequenceDataset.increase_padding(states, pad, new_pad)
                actions = SequenceDataset.increase_padding(actions, pad, new_pad)
                rewards = SequenceDataset.increase_padding(rewards, pad, new_pad)
                dones = SequenceDataset.increase_padding(dones, pad, new_pad)
                pad = new_pad

        return (
            torch.tensor(states, dtype=torch.float32),          # (context_len, state_dim...)
            torch.tensor(actions, dtype=torch.float32),         # (context_len, action_dim,)
            torch.tensor(rewards, dtype=torch.float32),         # (context_len,)
            torch.tensor(dones, dtype=torch.bool),              # (context_len,)
            torch.tensor(next_states, dtype=torch.float32),     # (state_dim...)
            torch.tensor(pad, dtype=torch.long),                # (,)
        )
    

def get_dataloader(data_name: str, seq_len: int, batch_size: int=64, test_split: float=0.1, 
                   seed: int=0, **dataset_kwargs):
    '''Loads episodic data from a file, splits into train/test sets, and returns PyTorch DataLoaders 
    for each.'''
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

def plot_trajectories(trajectories, plot_name: str) -> None:
    '''Plots trajectories of state variables over time and saves the plot to a file.'''
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.detach().cpu().numpy()

    n_states = trajectories[0].shape[-1]
    fig, axs = plt.subplots(1, n_states, figsize=(4 * n_states, 4), squeeze=False)
    for traj in trajectories:
        for j in range(n_states):
            axs[0, j].plot(traj[:, j])
    plt.savefig(os.path.join(PLOTS_PATH, plot_name))
    plt.close(fig)


def plot_data_trajectories(data_name: str, limit: int, plot_name: str) -> None:
    '''Plots trajectories of state variables over time from a data file.'''
    episode_data = list(load_episodic_data(data_name))
    random.shuffle(episode_data)
    trajectories = [episode_data[i]['states'] for i in range(limit)]
    plot_trajectories(trajectories, plot_name)


def save_video(render_fn, trajectories, plot_name: str) -> None:
    '''Saves a video of trajectories rendered by a provided function.'''
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.detach().cpu().numpy()
    
    frames = []
    for trajectory in trajectories:
        for state in trajectory:
            frames.append(render_fn(state))

    frames[0].save(
        fp=os.path.join(PLOTS_PATH, plot_name), 
        format='GIF', append_images=frames, save_all=True, duration=100)
