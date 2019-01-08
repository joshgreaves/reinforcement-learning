from torch import nn

import rl
from environments.gym import CartPoleEnvironmentFactory
import networks as nets


def train_ppo():
    factory = CartPoleEnvironmentFactory()
    policy = nn.Sequential(
        nets.FourLayerMlp(4, 2, hidden_dim=10),
        nets.MultinomialNetwork()
    )
    value = nets.FourLayerMlp(4, 1, hidden_dim=10)

    ppo = rl.PPO(factory, policy, value, experiment_name='cartpole_basic')
    ppo.train(100, rollouts_per_epoch=100, max_episode_length=200, policy_epochs=5, batch_size=256)


def train_dqn():
    factory = CartPoleEnvironmentFactory()
    action_value_network = nets.FourLayerMlp(4, 2, hidden_dim=10)

    # dqn = rl.DQN(factory, action_value_network)
    # dqn.train(100)


if __name__ == '__main__':
    train_ppo()
