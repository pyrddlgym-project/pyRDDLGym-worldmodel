from PIL import Image
import numpy as np
import torch
import pyRDDLGym
from pyRDDLGym.core.policy import BaseAgent

from twm.core.data import create_image_data, get_dataloader, image_to_tensor, save_video
from twm.core.model import WorldModel
from twm.core.eval import RolloutContext


class PongEnvWithRandomStarts:

    def __init__(self):
        self.env = pyRDDLGym.make("Pong_arcade", '1', vectorized=True)
        self._visualizer = self.env._visualizer
    
    def reset(self):
        vel_x = np.random.choice([-0.03, 0.03], size=(1,))
        vel_y = np.random.choice([-0.01, 0.01], size=(1,))
        state, info = self.env.reset()
        self.env.sampler.subs['vel-x'] = vel_x
        self.env.sampler.subs['vel-y'] = vel_y
        self.env.sampler.states['vel-x'] = vel_x
        self.env.sampler.states['vel-y'] = vel_y
        self.env.state = self.env.sampler.states
        state['vel-x'] = vel_x
        state['vel-y'] = vel_y
        return state, info
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    

class PongPolicy(BaseAgent):

    def sample_action(self, state):
        ball_y = state['ball-y'][0]
        paddle_y = state['paddle-y']
        if ball_y < paddle_y + 0.05:    
            return {'move': -1 if np.random.rand() < 0.85 else 1}
        elif ball_y > paddle_y + 0.05:  
            return {'move': 1 if np.random.rand() < 0.85 else -1}
        else:                    
            return {'move': 0}


def vec_policy(states):
    return {'move': np.random.choice([-1, 0, 1])}    


def create_pong_data(episodes=200, max_steps=200, save_path='pong_image_data.pkl'):
    env = PongEnvWithRandomStarts()
    policy = PongPolicy()
    create_image_data(env, policy, episodes, max_steps, save_path)


def plot_rollouts(model):
    env = PongEnvWithRandomStarts()
    env.reset()
    init_image = image_to_tensor(env.render())
    init_state = {'obs': torch.from_numpy(init_image).float().to('cuda')[None, None]}
    
    rollout_context = RolloutContext(model)
    trajectories = rollout_context.rollout(init_state, None, vec_policy, max_steps=200)
    trajectories = [{k: v[0].detach().cpu() for k, v in trajectories.items()}]

    def render_fn(state_dict):
        state_img = state_dict['obs'].transpose(1, 2, 0)
        return Image.fromarray((state_img * 255).astype(np.uint8))
    save_video(render_fn, trajectories, 'pong_model_rollout.gif')


if __name__ == "__main__":
    # create_pong_data()
    seq_len = 10
    fit = True

    if fit:
        train_loader, test_loader = get_dataloader(
            'pong_image_data.pkl', seq_len, batch_size=64, augment_starts=False)
        state_dims = train_loader.dataset.state_dims
        action_dims = train_loader.dataset.action_dims
   
        model = WorldModel(state_dims, action_dims, visual=True, seq_len=seq_len).to('cuda')
        model.fit(train_loader, epochs=300, lr=0.00001, test_data_loader=test_loader, 
                  model_name='pong_image_world_model.pth')
    
    else:
        model = WorldModel.load('pong_image_world_model.pth').to('cuda')
        
        plot_rollouts(model)
    