import torch.nn as nn

from rl import *
from environments.gym import CartPoleEnvironmentFactory
import networks as nets


def main():
    factory = CartPoleEnvironmentFactory()
    embedding = nn.Sequential(nn.Linear(4, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU())
    policy = nets.DiscreteActionSpaceNetwork(nets.FourLayerMlp(100, 2, hidden_dim=10))
    value = nets.FourLayerMlp(100, 1, hidden_dim=10)

    ppo(factory, policy, value, multinomial_likelihood, embedding_net=embedding, epochs=1000, rollouts_per_epoch=100,
        max_episode_length=200, gamma=0.99, policy_epochs=5, batch_size=256, experiment_name='cartpole_shared')


if __name__ == '__main__':
    main()
