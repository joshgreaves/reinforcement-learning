import torch

from rl import *
from environments.doom import DoomHallwayFactory
import networks as nets


def main():
    factory = DoomHallwayFactory()
    conv = nets.ConvNetwork128(120, 160, 4)
    policy = nets.DiscreteActionSpaceNetwork(nets.FourLayerMlp(conv.get_output_dim(), 6))
    value = nets.FourLayerMlp(conv.get_output_dim(), 1)
    ppo(factory, policy, value, multinomial_likelihood, embedding_net=conv, epochs=1000, rollouts_per_epoch=500,
        max_episode_length=3000, gamma=0.999, policy_epochs=4, batch_size=256, epsilon=0.2, environment_threads=5,
        data_loader_threads=10, device=torch.device('cuda'), lr=1e-3, weight_decay=0.0, gif_epochs=5,
        experiment_name='doom_hallway')


if __name__ == '__main__':
    main()
