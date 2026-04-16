import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from PIL import Image
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]

Array = np.ndarray
ArrayDict = Dict[str, Array]


PARENT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(PARENT_PATH, 'data')
PLOTS_PATH = os.path.join(PARENT_PATH, 'plots')


# <------------------------------- Data collection  ------------------------------>

def image_to_tensor(img: Image.Image, size: Tuple[int, int]=(64, 64)) -> Array:
    '''Converts a PIL image into a normalized tensor.'''
    img = img.resize(size, Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float16) / 255.0
    return np.transpose(x, (2, 0, 1))


def dict_append(src: Dict[str, Any], dest: Dict[str, List[Any]]) -> None:
    '''Appends src arrays to dest arrays along the first dimension.'''
    for key, value in src.items():
        dest.setdefault(key, []).append(np.asarray(value))


def create_vector_data(env, policy, episodes: int, max_steps: int, data_name: str, 
                       render: bool=False) -> None:
    '''Collects vector data by running a policy in an environment, and saves it to a file.'''
    episodes = int(episodes)
    max_steps = int(max_steps)

    # collect data
    states, actions, next_states, rewards, dones = {}, {}, {}, [], []
    
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
            
            dict_append(state, states)
            dict_append(action, actions)
            dict_append(next_state, next_states)
            rewards.append(float(reward))
            dones.append(done)
            
            state = next_state
            if done:
                break
        pbar.set_postfix({"Episode Reward": total_reward})
    
    # save data
    data = {
        "states":      {k: np.array(v) for k, v in states.items()},
        "actions":     {k: np.array(v) for k, v in actions.items()},
        "next_states": {k: np.array(v) for k, v in next_states.items()},
        "rewards":     np.array(rewards),
        "dones":       np.array(dones),
    }
    
    with open(os.path.join(DATA_PATH, data_name), "wb") as f:
        pickle.dump(data, f)


def create_image_data(env, policy, episodes: int, max_steps: int, data_name: str, 
                      image_map: Callable[[Image.Image], Array]=image_to_tensor) -> None:
    '''Collects image data by running a policy in an environment, and saves it to a file.
    Uses the environment's render function to get image observations.'''
    episodes = int(episodes)
    max_steps = int(max_steps)

    # collect data
    states, actions, next_states, rewards, dones = {}, {}, {}, [], []
    
    for _ in (pbar := tqdm(range(episodes), desc="Collecting data")):
        state, _ = env.reset()
        obs = {'obs': image_map(env.render())}
        total_reward = 0.

        for step in range(max_steps):
            action = policy.sample_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            next_obs = {'obs': image_map(env.render())}
            total_reward += reward
            done = term or trunc or step == max_steps - 1

            dict_append(obs, states)
            dict_append(action, actions)
            dict_append(next_obs, next_states)
            rewards.append(float(reward))
            dones.append(done)
            
            state, obs = next_state, next_obs
            if done:
                break
        pbar.set_postfix({"Episode Reward": total_reward})
    
    # save data
    data = {
        "states":      {k: np.array(v) for k, v in states.items()},
        "actions":     {k: np.array(v) for k, v in actions.items()},
        "next_states": {k: np.array(v) for k, v in next_states.items()},
        "rewards":     np.array(rewards),
        "dones":       np.array(dones),
    }
    
    with open(os.path.join(DATA_PATH, data_name), "wb") as f:
        pickle.dump(data, f)


# <------------------------------- Data preparation  ------------------------------>

def load_episodic_data(data_name: str) -> Generator[Dict[str, Any], None, None]:
    '''Loads data from a file and yields it as individual episode dicts.'''
    with open(os.path.join(DATA_PATH, data_name), "rb") as f:
        data = pickle.load(f)

    # find episode boundaries based on dones
    ep_ends = np.where(data['dones'])[0] + 1
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    
    # split data into episodes
    for start, end in zip(ep_starts, ep_ends):
        dones_ep = data['dones'][start:end]  
        assert dones_ep[-1] and np.all(~dones_ep[:-1])
        yield {
            "states":      {k: v[start:end] for k, v in data['states'].items()},
            "actions":     {k: v[start:end] for k, v in data['actions'].items()},
            "next_states": {k: v[start:end] for k, v in data['next_states'].items()},
            "rewards":     data['rewards'][start:end],
            "dones":       dones_ep,
            'len':         end - start,
        }


class SequenceDataset(torch.utils.data.Dataset):
    '''A PyTorch Dataset that takes episodic data and returns padded sequences of a specified length.'''

    def __init__(self, episodes, seq_len: int, obs_states: Optional[Set[str]]=None,
                 augment_starts: bool=False, min_frames: int=2) -> None:
        self.seq_len = seq_len
        if obs_states is None:
            obs_states = set(episodes[0]['states'].keys())
        self.obs_states = set(obs_states)
        self.augment_starts = augment_starts
        self.min_frames = max(1, int(min_frames))
        
        # store raw per-episode arrays and build a flat (ep_idx, t) sample index        
        self._episodes, self._index = SequenceDataset.init_episodes(episodes, self.obs_states)
        self.state_dims = {k: v.shape[1:] for k, v in self._episodes[0]['states'].items()}
        self.action_dims = {k: v.shape[1:] for k, v in self._episodes[0]['actions'].items()}
        self.input_dims = {**self.state_dims, **self.action_dims}

        # calculate data set stats for normalization
        self.normalizer_stats = SequenceDataset.init_stats(self._episodes)

    # <------------------------------- Data preparation  ------------------------------>

    @staticmethod
    def init_episodes(episodes, obs_states: Set[str]):
        '''Processes raw episode data, optionally selecting a subset of states, and builds a flat 
        (ep_idx, t) sample index.'''
        new_episodes, index = [], []

        for ep_idx, ep in enumerate(episodes):

            # optionally select only a subset of states to be returned by the dataset
            new_episodes.append({
                'states':      {k: v for k, v in ep['states'].items() if k in obs_states},
                'actions':     ep['actions'],
                'next_states': {k: v for k, v in ep['next_states'].items() if k in obs_states},
                'rewards':     ep['rewards'],
                'dones':       ep['dones'],
                'len':         ep['len'],
            })
            
            # build a flat (ep_idx, t) sample index for the dataset
            for t in range(ep['len']):
                index.append((ep_idx, t))

        return new_episodes, index

    @staticmethod
    def init_stats(episodes):
        '''Calculates mean and std for states and actions across the entire dataset.'''
        states, actions = {}, {}
        for ep in episodes:
            dict_append(ep['next_states'], states)
            dict_append(ep['actions'], actions)

        # calculate data set stats for state
        stats = {}
        for key, values in states.items():
            states_array = torch.tensor(np.concatenate(values, axis=0), dtype=torch.float32)
            state_mean = states_array.mean(dim=0)
            state_std = states_array.std(dim=0).clamp(min=1e-8)
            stats[key] = (state_mean, state_std)
        
        # calculate data set stats for action       
        for key, values in actions.items():
            actions_array = torch.tensor(np.concatenate(values, axis=0), dtype=torch.float32)
            action_mean = actions_array.mean(dim=0)
            action_std = actions_array.std(dim=0).clamp(min=1e-8)
            stats[key] = (action_mean, action_std)

        return stats

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

    @staticmethod
    def dict_of_float_tensor(value_dict: ArrayDict) -> TensorDict:
        '''Converts a dict of numpy arrays into a dict of PyTorch tensors.'''
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in value_dict.items()}
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        '''Returns a padded sequence sample from the dataset at the given index.'''
        ep_idx, t = self._index[idx]
        ep = self._episodes[ep_idx]
        
        # get padded sequences of states, actions, rewards, dones, and next_states
        states = {key: self.make_padded(arr, t, self.seq_len)[0]
                  for key, arr in ep['states'].items()}
        actions = {key: self.make_padded(arr, t, self.seq_len)[0]
                   for key, arr in ep['actions'].items()}            
        next_states = {k: v[t] for k, v in ep['next_states'].items()}
        rewards, _ = self.make_padded(ep['rewards'], t, self.seq_len)
        dones, pad = self.make_padded(ep['dones'], t, self.seq_len)
        
        # HER-style: pick a random virtual start in [current_pad_len, seq_len-1]
        if self.augment_starts:
            seq_len = rewards.shape[0]
            max_pad = max(seq_len - self.min_frames, pad)
            new_pad = np.random.randint(pad, max_pad + 1)
            if new_pad > pad:
                states = {k: SequenceDataset.increase_padding(v, pad, new_pad) 
                          for k, v in states.items()}
                actions = {k: SequenceDataset.increase_padding(v, pad, new_pad) 
                           for k, v in actions.items()}
                rewards = SequenceDataset.increase_padding(rewards, pad, new_pad)
                dones = SequenceDataset.increase_padding(dones, pad, new_pad)
                pad = new_pad

        # convert to PyTorch tensors and return
        return {
            "states":      SequenceDataset.dict_of_float_tensor(states),       # (seq_len, state_dims...)
            "actions":     SequenceDataset.dict_of_float_tensor(actions),      # (seq_len, action_dims...)
            "next_states": SequenceDataset.dict_of_float_tensor(next_states),  # (state_dims...)
            "rewards":     torch.tensor(rewards, dtype=torch.float32),         # (seq_len,)
            "dones":       torch.tensor(dones, dtype=torch.bool),              # (seq_len,)
            "pad":         torch.tensor(pad, dtype=torch.long),                # ()
        }
    

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

def plot_trajectories(trajectories: List[TensorDict], plot_name: str) -> None:
    '''Plots trajectories of state variables over time and saves the plot to a file.'''
    n_states = sum(int(np.prod(v.shape[1:], dtype=np.int64)) for v in trajectories[0].values())
    fig, axs = plt.subplots(1, n_states, figsize=(4 * n_states, 4), squeeze=False)
    axs = axs.flat

    for traj in trajectories:
        idx = 0
        for key, value in traj.items():
            if isinstance(value, torch.Tensor):
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


def save_video(render_fn, trajectories: List[TensorDict], plot_name: str) -> None:
    '''Saves a video of trajectories rendered by a provided function.'''
    frames = []
    for trajectory in trajectories:
        trajectory_np = {}
        for k, v in trajectory.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            trajectory_np[k] = v

        n_steps = next(iter(trajectory_np.values())).shape[0]
        for i in range(n_steps):
            state = {k: v[i] for k, v in trajectory_np.items()}
            frames.append(render_fn(state))
    
    frames[0].save(
        fp=os.path.join(PLOTS_PATH, plot_name), 
        format='GIF', append_images=frames, save_all=True, duration=100)
