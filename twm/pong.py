import numpy as np
import torch
import pyRDDLGym
from pyRDDLGym.core.policy import BaseAgent

from .data import create_data, get_dataloader, to_tensor, \
    plot_data_trajectories, plot_trajectories, save_video
from .model import RolloutContext, WorldModel


class PongEnvWithRandomStarts:

    def __init__(self):
        self.env = pyRDDLGym.make("Pong_arcade", '1')
        self._visualizer = self.env._visualizer
    
    def reset(self):
        state, info = self.env.reset()
        self.env.sampler.subs['vel-x'] = np.random.choice([-0.03, 0.03], size=(1,))
        self.env.sampler.subs['vel-y'] = np.random.choice([-0.01, 0.01], size=(1,))
        self.env.sampler.states['vel-x'] = self.env.sampler.subs['vel-x'][0]
        self.env.sampler.states['vel-y'] = self.env.sampler.subs['vel-y'][0]
        self.env.state = self.env.sampler.states
        state['vel-x___b1'] = self.env.sampler.subs['vel-x'][0]
        state['vel-y___b1'] = self.env.sampler.subs['vel-y'][0]
        return state, info
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
        
    
class PongPolicy(BaseAgent):

    def sample_action(self, state):
        ball_y = state['ball-y___b1']
        paddle_y = state['paddle-y']
        if ball_y < paddle_y + 0.05:    
            return {'move': -1 if np.random.rand() < 0.9 else 1}
        elif ball_y > paddle_y + 0.05:  
            return {'move': 1 if np.random.rand() < 0.9 else -1}
        else:                    
            return {'move': 0}


def vec_policy(states):
    policy = PongPolicy()
    actions = []
    for state in states:
        state_dict = {'ball-x___b1': state[0], 'ball-y___b1': state[1], 'paddle-y': state[2]}
        action = policy.sample_action(state_dict)
        actions.append(np.array([action['move']]))
    return np.array(actions)


def create_pong_data(episodes=300, max_steps=200, save_path='pong_data.pkl'):
    env = PongEnvWithRandomStarts()
    policy = PongPolicy()
    create_data(env, policy, episodes, max_steps, save_path)


def init_torch(*xs, batch_size):
    new_xs = [torch.tensor(to_tensor(x))[None, None, :] for x in xs]
    return torch.cat(new_xs, dim=1).expand(batch_size, -1, -1)


def plot_model_rollouts(model, batch_size=4):
    env = PongEnvWithRandomStarts()
    init_state = env.reset()[0]
    init_action = PongPolicy().sample_action(init_state)
    next_state = env.step(init_action)[0]

    init_states = init_torch(init_state, next_state, batch_size=batch_size)
    init_actions = init_torch(init_action, batch_size=batch_size)

    rollout_context = RolloutContext(model)
    trajectories = rollout_context.rollout(
        init_states, init_actions, vec_policy, max_steps=200)
    
    state_keys = ['ball-x___b1', 'ball-y___b1', 'paddle-y']
    plot_data_trajectories('pong_data.pkl', batch_size, 'pong_data_rollouts.png')
    plot_trajectories(trajectories, save_path='pong_model_rollouts.png')
    save_video(state_keys, env._visualizer, trajectories, 'pong_model_rollout.gif')


if __name__ == "__main__":
    #create_pong_data()
    seq_len = 10
    fit = False

    if fit:
        train_loader, test_loader = get_dataloader(
            'pong_data.pkl', seq_len, batch_size=64, obs_idx=[0, 1, 4], augment_starts=False)
        state_dim = train_loader.dataset.dataset.state_dim
        action_dim = train_loader.dataset.dataset.action_dim
   
        model = WorldModel(state_dim, action_dim, seq_len).to('cuda')
        model.fit(train_loader, epochs=500, test_data_loader=test_loader, path='pong_world_model.pth')
    
    else:
        model = WorldModel.load('pong_world_model.pth').to('cuda')
        
        plot_model_rollouts(model)
    