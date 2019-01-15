from torch import nn

import rl
from environments.gym import HillClimbEnvironmentFactory
import networks as nets


class HashCount:
    def __init__(self):
        self._counts = dict()

    def get_count(self, x):
        hashed = self._get_hash(x)
        if hashed not in self._counts:
            self._counts[hashed] = 1
        else:
            self._counts[hashed] += 1
        return self._counts[hashed]

    def _get_hash(self, x):
        return int(x[0] / .1), int(x[1] / .01)


def train_ppo():
    factory = HillClimbEnvironmentFactory()
    policy = nn.Sequential(
        nets.FourLayerMlp(2, 3, hidden_dim=10),
        nets.MultinomialNetwork()
    )
    value = nets.FourLayerMlp(2, 1, hidden_dim=10)
    counter = HashCount()

    ppo = rl.PPO(factory, policy, value, experiment_name='hillclimb', count_fun=counter.get_count)
    ppo.train(100, rollouts_per_epoch=100, max_episode_length=200, policy_epochs=5, batch_size=256)


if __name__ == '__main__':
    train_ppo()
