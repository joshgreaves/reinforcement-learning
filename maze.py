import torch
from torch import nn

import rl
from environments.holodeck import MazeWorldEnvironmentFactory
import networks as nets


class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute((0, 3, 1, 2))
        out = self._conv(x)
        return out.view((-1, 128 * 16 * 16))


def train_ppo():
    factory = MazeWorldEnvironmentFactory(res=(64, 64))
    # conv = nets.ConvNetwork128(128, 128, input_channels=4)
    conv = SimpleConv()
    policy = nn.Sequential(
        nets.FourLayerMlp(128 * 16 * 16, 4, hidden_dim=50),
        nets.MultinomialNetwork()
    )
    value = nets.FourLayerMlp(128 * 16 * 16, 1, hidden_dim=50)

    ppo = rl.PPO(factory, policy, value, embedding_network=conv, device=torch.device('cuda'), gamma=0.99,
                 experiment_name='holodeck_maze_basic3', gif_epochs=1, epsilon=0.05, backup_n=30, entropy_coefficient=1e-6)
    ppo.train(1000, rollouts_per_epoch=9, max_episode_length=300, environment_threads=9, data_loader_threads=10)


if __name__ == '__main__':
    train_ppo()
