import torch

from rl import *
from environments import MazeWorldEnvironmentFactory
import networks as nets


def main():
    factory = MazeWorldEnvironmentFactory()
    conv = nets.ConvNetwork128(128, 128, input_channels=4)
    policy = nets.DiscreteActionSpaceNetwork(nets.FourLayerMlp(conv.get_output_dim(), 4, hidden_dim=100))
    value = nets.FourLayerMlp(conv.get_output_dim(), 1, hidden_dim=100)
    ppo(factory, policy, value, multinomial_likelihood, embedding_net=conv, epochs=1000, rollouts_per_epoch=10,
        max_episode_length=450, gamma=0.999, policy_epochs=4, batch_size=256, epsilon=0.2, environment_threads=10,
        data_loader_threads=10, device=torch.device('cuda'), lr=1e-3, weight_decay=0.01, gif_epochs=5,
        experiment_name='holodeck_maze_basic')


if __name__ == '__main__':
    main()

