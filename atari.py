import torch
from torch import nn

import rl
from environments import AtariEnvironmentFactory
import networks as nets


def train_ppo():
    factory = AtariEnvironmentFactory('SpaceInvaders-v0')
    conv = nets.ConvNetwork128(240, 160, input_channels=4)
    policy = nn.Sequential(
        nets.FourLayerMlp(conv.get_output_dim(), 6, hidden_dim=100),
        nets.MultinomialNetwork()
    )
    value = nets.FourLayerMlp(conv.get_output_dim(), 1, hidden_dim=100)

    ppo = rl.POPO(factory, policy, value, embedding_network=conv, device=torch.device('cuda'), gamma=0.999,
                 experiment_name='space_invaders_popo_10', gif_epochs=2)
    ppo.train(1000, rollouts_per_epoch=10, max_episode_length=1500, environment_threads=10, data_loader_threads=10,
              batch_size=100)


def train_dqn():
    factory = AtariEnvironmentFactory('SpaceInvaders-v0')
    conv = nets.ConvNetwork128(240, 160, input_channels=4)
    network = nn.Sequential(
        conv,
        nets.FourLayerMlp(conv.get_output_dim(), 6, hidden_dim=100),
        nets.MultinomialNetwork()
    )

    dqn = rl.DQN(factory, network, device=torch.device('cuda'), gamma=0.999,
                 experiment_name='space_invaders_dqn', gif_epochs=5, exp_replay_size=50000,
                 target_network_copy_epochs=1)
    dqn.train(10000, rollouts_per_epoch=25, max_episode_length=1500, environment_threads=10, data_loader_threads=10)


if __name__ == '__main__':
    train_ppo()
