from PIL import Image
import numpy as np
import torch

from twm.core.data import create_data, get_dataloader, image_to_tensor, save_video
from twm.core.model import WorldModel, WorldModelEvaluator
from twm.core.spec import FluentSpec, EnvSpec

from twm.examples.pong_train import PongEnvWithRandomStarts, PongPolicy


state_spec = { 'image': FluentSpec(shape=(3, 64, 64), prange='pixel') }
action_spec = { 'move': FluentSpec(shape=(), prange='int', values=(-1, +1)) }
env_spec = EnvSpec(state_spec=state_spec, action_spec=action_spec)


def vec_policy(states):
    return {'move': torch.as_tensor(np.random.choice([-1, 0, 1]))}    


def create_pong_data(episodes=300, max_steps=200, save_path='pong_image_data.pkl'):
    env = PongEnvWithRandomStarts()
    policy = PongPolicy()
    create_data(env, env_spec, policy, episodes, max_steps, save_path)


def plot_rollouts(model):
    env = PongEnvWithRandomStarts()
    env.reset()
    init_image = image_to_tensor(env.render())
    init_action = {'move': 0}
    env.step(init_action)
    next_state = image_to_tensor(env.render())

    images = [init_image, next_state]
    stacked_images = np.stack(images, axis=0)[None]
    init_state = {'image': torch.from_numpy(stacked_images).float().to('cuda')}
    init_action = {'move': torch.as_tensor([[0]], dtype=torch.float32).to('cuda')}
    
    rollout_context = WorldModelEvaluator(model)
    trajectories = rollout_context.rollout(init_state, init_action, vec_policy, max_steps=200)
    trajectories = [{k: v[0].detach().cpu() for k, v in trajectories.items()}]

    def render_fn(state_dict):
        state_img = state_dict['image'].transpose(1, 2, 0)
        return Image.fromarray((state_img * 255).astype(np.uint8))
    save_video(render_fn, trajectories, 'pong_model_rollout.gif')


if __name__ == "__main__":
    # create_pong_data()
    seq_len = 8
    fit = False

    if fit:
        train_loader, test_loader = get_dataloader(
            'pong_image_data.pkl', seq_len, batch_size=64, augment_starts=False)
        
        model = WorldModel(env_spec, seq_len).to('cuda')
        model.fit(train_loader, epochs=200, lr=0.0003, test_data_loader=test_loader,
                  model_name=f'pong_image_world_model_{seq_len}.pth')
    else:
        model = WorldModel.load(f'pong_image_world_model_{seq_len}.pth').to('cuda')    
        plot_rollouts(model)
    