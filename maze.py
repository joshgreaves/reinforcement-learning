import torch
from torch import nn

import rl
from environments import MazeWorldEnvironmentFactory
import networks as nets


def train_ppo():
    factory = MazeWorldEnvironmentFactory()
    conv = nets.ConvNetwork128(128, 128, input_channels=4)
    policy = nn.Sequential(
        nets.FourLayerMlp(conv.get_output_dim(), 4, hidden_dim=100),
        nets.MultinomialNetwork()
    )
    value = nets.FourLayerMlp(conv.get_output_dim(), 1, hidden_dim=100)

    ppo = rl.PPO(factory, policy, value, embedding_network=conv, device=torch.device('cuda'), gamma=0.999,
                 experiment_name='holodeck_maze_basic', gif_epochs=5)
    ppo.train(1000, rollouts_per_epoch=10, max_episode_length=600, environment_threads=10, data_loader_threads=10)


if __name__ == '__main__':
    train_ppo()
