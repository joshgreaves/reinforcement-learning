import torch
from torch import nn

import rl
from environments import DoomHallwayFactory, DoomDefendLineFactory
import networks as nets


# def main():
#     factory = DoomHallwayFactory()
#     conv = nets.ConvNetwork128(120, 160, 4)
#     policy = nets.DiscreteActionSpaceNetwork(nets.FourLayerMlp(conv.get_output_dim(), 6))
#     value = nets.FourLayerMlp(conv.get_output_dim(), 1)
#     ppo(factory, policy, value, multinomial_likelihood, embedding_net=conv, epochs=1000, rollouts_per_epoch=500,
#         max_episode_length=3000, gamma=0.999, policy_epochs=4, batch_size=256, epsilon=0.2, environment_threads=5,
#         data_loader_threads=10, device=torch.device('cuda'), lr=1e-3, weight_decay=0.0, gif_epochs=5,
#         experiment_name='doom_hallway')


def train_dqn():
    factory = DoomDefendLineFactory(render=False)
    conv = nets.ConvNetwork128(120, 160, input_channels=4)
    network = nn.Sequential(
        conv,
        nets.FourLayerMlp(conv.get_output_dim(), 3, hidden_dim=100),
        nets.MultinomialNetwork()
    )

    dqn = rl.DQN(factory, network, device=torch.device('cuda'), gamma=0.995,
                 experiment_name='doom_dqn_luminance', gif_epochs=1, exp_replay_size=125000,
                 target_network_copy_epochs=1, betas=(0.1, 0.5), epsilon_decay_epochs=200)
    dqn.train(1000, rollouts_per_epoch=50, policy_epochs=2, batch_size=256, max_episode_length=1500, environment_threads=10,
              data_loader_threads=10)


if __name__ == '__main__':
    train_dqn()
