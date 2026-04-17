import pyRDDLGym
import numpy as np
from stable_baselines3 import PPO

from twm.core.model import WorldModel
from twm.core.eval import WorldModelEnv
    

def create_world_model_env():
    world_model = WorldModel.load('pong_world_model.pth').to('cuda')
    init_state = {'ball-x': np.array([0.5]), 'ball-y': np.array([0.5]), 
                  'vel-x': np.array([0.0]), 'vel-y': np.array([0.0]), 'paddle-y': np.array([0.4])}
    reward_fn = lambda s, a, ns: -ns['ball-x'][0]
    return WorldModelEnv(world_model, reward_fn=reward_fn, initial_state=init_state, 
                         min_action=-1.0, max_action=1.0, max_steps=200, 
                         discrete_action_keys={'move'})


class PongVecEnv:

    def __init__(self):
        self.env = pyRDDLGym.make("Pong_arcade", '1', vectorized=True)
        self._visualizer = self.env._visualizer
    
    def state_to_vec(self, state):
        return np.concatenate([state['ball-x'], state['ball-y'], state['paddle-y']])
    
    def reset(self):
        state, info = self.env.reset()
        state = self.state_to_vec(state)
        return state, info
        
    def step(self, action):
        action = {'move': action - 1}
        state, *etc = self.env.step(action)
        state = self.state_to_vec(state)
        return state, *etc
    
    def render(self):
        return self.env.render()
    

def eval_rl_agent_in_env(env, model, episodes=10):
    avg_reward = 0.0
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            done = term or trunc
        avg_reward += total_reward
    avg_reward /= episodes
    print(f"Average reward over {episodes} episodes: {avg_reward:.3f}")


def train_rl_agent(env, steps=10000, iters=20):
    model = PPO("MlpPolicy", env, verbose=1)
    for _ in range(iters):
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        eval_rl_agent_in_env(env, model)
        eval_rl_agent_in_env(PongVecEnv(), model)
        

if __name__ == "__main__":
    env = create_world_model_env()
    train_rl_agent(env)
