import numpy as np
import torch
import pyRDDLGym
from pyRDDLGym.core.policy import BaseAgent

from twm.core.data import create_data, get_dataloader, \
    plot_data_trajectories, plot_trajectories, save_video
from twm.core.model import WorldModel, WorldModelEvaluator
from twm.core.spec import EnvSpec, FluentSpec


state_spec = {
    'ball-x': FluentSpec(shape=(1,), prange='real'),
    'ball-y': FluentSpec(shape=(1,), prange='real'),
    'paddle-y': FluentSpec(shape=(1,), prange='real'),
}
action_spec = {
    'move': FluentSpec(shape=(), prange='int', values=(-1, +1)),
}
env_spec = EnvSpec(state_spec=state_spec, action_spec=action_spec)


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
    ball_y = states['ball-y'][:, 0].detach().cpu().numpy()
    paddle_y = states['paddle-y'][:, 0].detach().cpu().numpy()
    actions = np.zeros((len(ball_y),), dtype=np.int32)
    actions[ball_y < paddle_y + 0.05] = np.where(
        np.random.rand((ball_y < paddle_y + 0.05).sum()) < 0.85, -1, 1)
    actions[ball_y > paddle_y + 0.05] = np.where(
        np.random.rand((ball_y > paddle_y + 0.05).sum()) < 0.85, 1, -1)
    return {'move': torch.from_numpy(actions)}    


def create_pong_data(episodes=500, max_steps=200, save_path='pong_data.pkl'):
    env = PongEnvWithRandomStarts()
    policy = PongPolicy()
    create_data(env, env_spec, policy, episodes, max_steps, save_path)


def plot_rollouts(model, batch_size=4):
    env = PongEnvWithRandomStarts()

    # rollout trajectories from the world model
    eval = WorldModelEvaluator(model)
    init_states = {k: torch.from_numpy(v).float().to('cuda')[None, None]
                   for k, v in env.reset()[0].items()}
    trajectories = eval.rollout(init_states, None, vec_policy, max_steps=200)
    trajectories = [{k: v[0].detach().cpu() for k, v in trajectories.items()}]
    
    # plot rollouts
    plot_data_trajectories('pong_data.pkl', batch_size, 'pong_data_rollouts.png')
    plot_trajectories(trajectories, 'pong_model_rollouts.png')

    # save rollout video
    def render_fn(state_dict):
        state = {'ball-x___b1': state_dict['ball-x'][0].item(),
                 'ball-y___b1': state_dict['ball-y'][0].item(), 
                 'paddle-y': state_dict['paddle-y'][0].item()}
        return env._visualizer.render(state)
    save_video(render_fn, trajectories, 'pong_model_rollout.gif')
    

if __name__ == "__main__":
    # create_pong_data()
    seq_len = 8
    fit = False
    
    if fit:
        train_loader, test_loader = get_dataloader(
            'pong_data.pkl', seq_len, batch_size=64, augment_starts=False)
   
        model = WorldModel(env_spec=env_spec, seq_len=seq_len).to('cuda')
        model.fit(train_loader, lr=0.001, epochs=600, test_data_loader=test_loader, 
                  model_name=f'pong_world_model_{seq_len}.pth')
    
    else:
        model = WorldModel.load(f'pong_world_model_{seq_len}.pth').to('cuda')     
        plot_rollouts(model)
    