from PIL import Image
import numpy as np
import torch
import pyRDDLGym
from pyRDDLGym.core.policy import BaseAgent

from twm.core.data import create_image_data, get_dataloader, image_to_tensor, save_video
from twm.core.model import RolloutContext, WorldModel


class PongEnvWithRandomStarts:

    def __init__(self):
        self.env = pyRDDLGym.make("Pong_arcade", '1')
        self._visualizer = self.env._visualizer
        self._visualizer._ball_radius *= 2.5
    
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
    actions = []
    for state in states:
        action = {'move': np.random.choice([-1, 0, 1])}
        actions.append(np.array([action['move']]))
    return np.array(actions)


def create_pong_data(episodes=120, max_steps=200, save_path='pong_image_data.pkl'):
    env = PongEnvWithRandomStarts()
    policy = PongPolicy()
    create_image_data(env, policy, episodes, max_steps, save_path)


def init_torch_state(*xs, batch_size):
    new_xs = [torch.tensor(x)[None, None, :, :, :] for x in xs]
    return torch.cat(new_xs, dim=1).expand(batch_size, -1, -1, -1, -1)


def init_torch_action(*xs, batch_size):
    new_xs = [torch.tensor(x['move'])[None, None] for x in xs]
    return torch.cat(new_xs, dim=1).expand(batch_size, -1, -1)


def plot_model_rollouts(model, batch_size=4):
    env = PongEnvWithRandomStarts()
    env.reset()
    init_state = image_to_tensor(env.render())
    init_action = {'move': 0}
    env.step(init_action)
    next_state = image_to_tensor(env.render())

    init_states = init_torch_state(init_state, next_state, batch_size=batch_size)
    init_actions = init_torch_action(init_action, batch_size=batch_size)

    rollout_context = RolloutContext(model)
    trajectories = rollout_context.rollout(
        init_states, init_actions, vec_policy, max_steps=200)

    def render_fn(state_img):
        state_img = state_img.transpose(1, 2, 0)
        return Image.fromarray((state_img * 255).astype(np.uint8))
    
    save_video(render_fn, trajectories, 'pong_model_rollout.gif')


if __name__ == "__main__":
    #create_pong_data()
    seq_len = 10
    fit = True

    if fit:
        train_loader, test_loader = get_dataloader(
            'pong_image_data.pkl', seq_len, batch_size=64, augment_starts=True)
        state_dim = train_loader.dataset.state_dim
        action_dim = train_loader.dataset.action_dim
   
        model = WorldModel(state_dim, action_dim, seq_len, visual=True, use_rope=True,
                           use_diffusion_decoder=True).to('cuda')
        model.fit(train_loader, epochs=1000, lr=0.0005, test_data_loader=test_loader, 
                  model_name='pong_image_world_model.pth')
    
    else:
        model = WorldModel.load('pong_image_world_model.pth').to('cuda')
        
        plot_model_rollouts(model)
    