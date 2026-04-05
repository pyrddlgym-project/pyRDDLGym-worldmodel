import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


# <------------------------------- Data collection  ------------------------------>

def to_tensor(state_dict):
    state_vec = []
    for value in state_dict.values():
        value = np.asarray(value, dtype=np.float32)
        state_vec.append(np.ravel(value))
    return np.concatenate(state_vec)


def to_state_dict(state_vec, state_keys):
    return dict(zip(state_keys, state_vec))


def create_data(env, policy, episodes, max_steps, save_path, 
                state_map=to_tensor, action_map=to_tensor, render=False):
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
    
    with open(save_path, "wb") as f:
        pickle.dump(data, f)


# <------------------------------- Data preparation  ------------------------------>

def load_and_episode_split_data(data_path):
    with open(data_path, "rb") as f:
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


def load_episodes(data_path):
    return list(load_and_episode_split_data(data_path))


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, episodes, seq_len: int, obs_idx=None,
                 augment_starts: bool=False, min_frames: int=2):
        self.augment_starts = augment_starts
        self.min_frames = max(1, int(min_frames))

        # create episode data but with padding
        sequences = self.load_and_episode_split_pad_data(episodes, seq_len)
        
        # filter observed components of state
        states = sequences["states"]
        next_states = sequences["next_states"]
        if obs_idx is not None:
            obs_idx = np.array(obs_idx)
            states = np.take(states, obs_idx, axis=2)
            next_states = np.take(next_states, obs_idx, axis=1)
        sequences["states"] = states
        sequences["next_states"] = next_states
        self.obs_idx = obs_idx
        
        self.state_shape = tuple(sequences["states"].shape[2:])
        self.state_dim = int(np.prod(self.state_shape, dtype=np.int64))
        self.action_dim = sequences["actions"].shape[-1]

        self.states = torch.tensor(sequences["states"], dtype=torch.float32)
        self.actions = torch.tensor(sequences["actions"], dtype=torch.float32)
        self.rewards = torch.tensor(sequences["rewards"], dtype=torch.float32)
        self.dones = torch.tensor(sequences["dones"], dtype=torch.bool)
        self.next_states = torch.tensor(sequences["next_states"], dtype=torch.float32)
        self.pad_lens = torch.tensor(sequences["pad_lens"], dtype=torch.long)

        # compute state normalization stats
        self.state_mean = self.next_states.mean(dim=0)   # (state_dim,)
        self.state_std = self.next_states.std(dim=0).clamp(min=1e-8)
        
        # compute action normalization stats
        N, T = self.actions.shape[:2]
        non_pad_idx = (T - self.pad_lens - 1).clamp(min=0)
        a_last = self.actions[torch.arange(N), non_pad_idx]
        self.action_mean = a_last.mean(dim=0)   # (action_dim,)
        self.action_std = a_last.std(dim=0).clamp(min=1e-8)

    @staticmethod
    def make_padded(x, t, seq_len):
        start = max(0, t - seq_len + 1)
        hist = x[start:t + 1]
        pad_len = seq_len - hist.shape[0]
        pad = np.zeros((pad_len,) + x.shape[1:], dtype=x.dtype)
        new_x = np.concatenate([hist, pad], axis=0)
        return new_x, pad_len

    @staticmethod
    def load_and_episode_split_pad_data(episodes, seq_len: int) -> dict:

        states, actions, rewards, dones, next_states, pad_lens = [], [], [], [], [], []
        
        for eps_data in episodes:
            
            for t in range(eps_data['len']):
                s_win, pad = SequenceDataset.make_padded(eps_data['states'], t, seq_len)
                a_win, _ = SequenceDataset.make_padded(eps_data['actions'], t, seq_len)
                r_win, _ = SequenceDataset.make_padded(eps_data['rewards'], t, seq_len)
                d_win, _ = SequenceDataset.make_padded(eps_data['dones'], t, seq_len)

                states.append(s_win)
                actions.append(a_win)
                rewards.append(r_win)
                dones.append(d_win)
                next_states.append(eps_data['next_states'][t])
                pad_lens.append(pad)

        return {
            "states":      np.stack(states),       # (T_total, seq_len, state_dim)
            "actions":     np.stack(actions),      # (T_total, seq_len, action_dim)
            "rewards":     np.stack(rewards),      # (T_total, seq_len)
            "dones":       np.stack(dones),        # (T_total, seq_len)
            "next_states": np.stack(next_states),  # (T_total, state_dim)
            "pad_lens":    np.stack(pad_lens),     # (T_total,)
        }

    @staticmethod
    def increase_padding(x, old_pad, new_pad):
        seq_len = x.shape[0]
        old_real = seq_len - old_pad
        new_real = seq_len - new_pad
        y = torch.zeros_like(x)
        y[:new_real] = x[old_real - new_real:old_real]
        return y

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]
        next_states = self.next_states[idx]
        pad_lens = self.pad_lens[idx]

        # HER-style: pick a random virtual start in [current_pad_len, seq_len-1]
        if self.augment_starts:
            seq_len = states.shape[0]
            old_pad = int(pad_lens.item())
            max_pad = max(seq_len - self.min_frames, old_pad)
            new_pad = int(torch.randint(old_pad, max_pad + 1, (1,)).item())
            if new_pad > old_pad:
                states = SequenceDataset.increase_padding(states, old_pad, new_pad)
                actions = SequenceDataset.increase_padding(actions, old_pad, new_pad)
                rewards = SequenceDataset.increase_padding(rewards, old_pad, new_pad)
                dones = SequenceDataset.increase_padding(dones, old_pad, new_pad)
                pad_lens = torch.tensor(new_pad, dtype=torch.long)

        return (
            states,          # (context_len, state_dim)
            actions,         # (context_len, action_dim)
            rewards,         # (context_len,)
            dones,           # (context_len,)
            next_states,     # (state_dim,)
            pad_lens,        # (,)
        )
    

def get_dataloader(data_path, context_len, batch_size=64, test_split=0.1, seed=0, 
                   **dataset_kwargs):
    episodes = list(load_and_episode_split_data(data_path))
    n_episodes = len(episodes)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)

    n_test = min(max(1, int(round(test_split * n_episodes))), n_episodes - 1)

    train_eps = [episodes[i] for i in perm[n_test:]]
    test_eps = [episodes[i] for i in perm[:n_test]]
    train_set = SequenceDataset(train_eps, context_len, **dataset_kwargs)
    test_set = SequenceDataset(test_eps, context_len, **dataset_kwargs)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


# <------------------------------- Data visualization  ------------------------------>

def plot_trajectories(trajectories, save_path):
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.detach().cpu().numpy()

    n_states = trajectories[0].shape[-1]
    fig, axs = plt.subplots(1, n_states, figsize=(4 * n_states, 4), squeeze=False)
    for traj in trajectories:
        for j in range(n_states):
            axs[0, j].plot(traj[:, j])
    plt.savefig(save_path)
    plt.close(fig)


def plot_data_trajectories(data_path, limit, save_path):
    episode_data = list(load_and_episode_split_data(data_path))
    random.shuffle(episode_data)
    trajectories = [episode_data[i]['states'] for i in range(limit)]
    plot_trajectories(trajectories, save_path)


def save_video(state_keys, viz, trajectories, save_path):
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.detach().cpu().numpy()
    
    frames = []
    for trajectory in trajectories:
        for state in trajectory:
            state_dict = to_state_dict(state, state_keys)
            image = viz.render(state_dict)
            frames.append(image)
    frames[0].save(
        fp=save_path, format='GIF', append_images=frames, save_all=True, duration=100)
