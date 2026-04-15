import numpy as np
import torch
import pyRDDLGym
from pyRDDLGym.core.policy import BaseAgent

from twm.core.data import create_vector_data, get_dataloader, dict_to_tensor, \
    plot_data_trajectories, plot_trajectories, save_video
from twm.core.model import WorldModel
from twm.core.eval import RolloutContext


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


def create_pong_data(episodes=200, max_steps=200, save_path='pong_data.pkl'):
    env = PongEnvWithRandomStarts()
    policy = PongPolicy()
    create_vector_data(env, policy, episodes, max_steps, save_path)


def init_torch(*xs, batch_size):
    new_xs = [torch.tensor(dict_to_tensor(x))[None, None, :] for x in xs]
    return torch.cat(new_xs, dim=1).expand(batch_size, -1, -1)


def plot_model_rollouts(model, batch_size=4):
    env = PongEnvWithRandomStarts()

    rollout_context = RolloutContext(model)
    init_states = init_torch(env.reset()[0], batch_size=batch_size)
    trajectories = rollout_context.rollout(init_states, None, vec_policy, max_steps=200)
    
    def render_fn(state_vec):
        state_keys = ['ball-x___b1', 'ball-y___b1', 'paddle-y']
        return env._visualizer.render(dict(zip(state_keys, state_vec)))
    
    plot_data_trajectories('pong_data.pkl', batch_size, 'pong_data_rollouts.png')
    plot_trajectories(trajectories, 'pong_model_rollouts.png')
    save_video(render_fn, trajectories, 'pong_model_rollout.gif')


if __name__ == "__main__":
    #create_pong_data()
    seq_len = 10
    fit = False

    if fit:
        train_loader, test_loader = get_dataloader(
            'pong_data.pkl', seq_len, batch_size=64, obs_idx=[0, 1, 4], augment_starts=True)
        state_dim = train_loader.dataset.state_dim
        action_dim = train_loader.dataset.action_dim
   
        model = WorldModel(state_dim, action_dim, seq_len, visual=False).to('cuda')
        model.fit(train_loader, lr=0.0007, epochs=600, test_data_loader=test_loader, 
                  model_name='pong_world_model.pth')
    
    else:
        model = WorldModel.load('pong_world_model.pth').to('cuda')
        
        plot_model_rollouts(model)
    