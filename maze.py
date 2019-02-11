import torch
from torch import nn

import rl
from environments.holodeck import MazeWorldEnvironmentFactory
import networks as nets


def train_ppo():
    num_mixtures = 5  # [weight, mean, variance]
    factory = MazeWorldEnvironmentFactory()
    conv = nets.ConvNetwork128(128, 128, input_channels=4)
    policy = nn.Sequential(
        nets.FourLayerMlp(conv.get_output_dim(), 4, hidden_dim=100),
        nets.MultinomialNetwork()
    )
    value = nets.FourLayerMlp(conv.get_output_dim(), num_mixtures * 3, hidden_dim=100)

    ppo = rl.PPO(factory, policy, value, embedding_network=conv, device=torch.device('cuda'), gamma=0.99,
                 experiment_name='holodeck_maze_distribution', gif_epochs=5, epsilon=0.4)
    ppo.train(1000, rollouts_per_epoch=36, max_episode_length=450, environment_threads=9, data_loader_threads=10,
              policy_epochs=5)


if __name__ == '__main__':
    train_ppo()
