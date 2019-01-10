from torch import nn

import rl
from environments.gym import HillClimbEnvironmentFactory
import networks as nets


def train_ppo():
    factory = HillClimbEnvironmentFactory()
    policy = nn.Sequential(
        nets.FourLayerMlp(2, 2, hidden_dim=10),
        nets.MultinomialNetwork()
    )
    value = nets.FourLayerMlp(2, 1, hidden_dim=10)
    intrinsic_net = nets.RND(2)

    ppo = rl.PPO(factory, policy, value, experiment_name='hillclimb', intrinsic_network=intrinsic_net)
    ppo.train(100, rollouts_per_epoch=100, max_episode_length=200, policy_epochs=5, batch_size=256)


if __name__ == '__main__':
    train_ppo()
