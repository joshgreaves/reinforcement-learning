from copy import copy
import holodeck
from holodeck.sensors import Sensors
import numpy as np
import torch
import torch.nn as nn

from rl import *


class MazeWorldEnvironmentFactory(EnvironmentFactory):
    def __init__(self):
        super(MazeWorldEnvironmentFactory, self).__init__()

    def new(self):
        return MazeWorldEnvironment()


class MazeWorldEnvironment(RLEnvironment):
    def __init__(self, res=(128, 128), prev_frames=4):
        super(MazeWorldEnvironment, self).__init__()
        self._env = holodeck.make('MazeWorld', window_res=(256, 256), cam_res=res)
        self._res = res
        self._prev_frames = prev_frames
        self._imgs = np.zeros((res[0], res[1], prev_frames), dtype=np.float32)

    def step(self, action):
        s, r, t, _ = self._env.step(action.item())
        self._process_state(s)
        return self._imgs, np.array((r.item())), False

    def reset(self):
        self._imgs = np.zeros((self._res[0], self._res[1], self._prev_frames), dtype=np.float32)
        s, _, _, _ = self._env.reset()
        self._process_state(s)
        return self._imgs

    def _process_state(self, s):
        self._imgs = np.roll(self._imgs, 1, axis=2)
        self._imgs[:, :, 0] = np.mean(copy(s[Sensors.PIXEL_CAMERA][:, :, :3]).astype(np.float32) / 255, axis=2)


class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        # Input image: [n, 1, 128, 128]
        self._conv = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),  # -> [n, 32, 128, 128]
            nn.ReLU(),
            nn.Conv2d(64, 64, 2, stride=2),  # -> [n, 64, 64, 64]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  # -> [n, 64, 64, 64]
            nn.ReLU(),
            nn.Conv2d(64, 128, 2, stride=2),  # -> [n, 128, 32, 32]
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),  # -> [n, 128, 32, 32]
            nn.ReLU(),
            nn.Conv2d(128, 256, 2, stride=2),  # -> [n, 256, 16, 16]
        )

    def forward(self, x):
        x = x.permute((0, 3, 1, 2))
        activations = self._conv(x)
        return activations.view((-1, 256 * 16 * 16))


class PolicyNetwork(nn.Module):
    def __init__(self, action_dim=4, hidden_dim=100):
        super(PolicyNetwork, self).__init__()
        self._fc = nn.Sequential(
            nn.Linear(256 * 16 * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x, get_action=True):
        scores = self._fc(x)
        probs = self._softmax(scores)

        if not get_action:
            return probs

        batch_size = x.shape[0]
        actions = np.empty((batch_size, 1), dtype=np.uint8)
        probs_np = probs.cpu().detach().numpy()
        for i in range(batch_size):
            action_one_hot = np.random.multinomial(1, probs_np[i])
            action_idx = np.argmax(action_one_hot)
            actions[i, 0] = action_idx
        return probs, actions


class ValueNetwork(nn.Module):
    def __init__(self, hidden_dim=100):
        super(ValueNetwork, self).__init__()
        self._fc = nn.Sequential(
            nn.Linear(256 * 16 * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self._fc(x)


def main():
    factory = MazeWorldEnvironmentFactory()
    conv = ConvNetwork()
    policy = PolicyNetwork()
    value = ValueNetwork()
    ppo(factory, policy, value, multinomial_likelihood, embedding_net=conv, epochs=1000, rollouts_per_epoch=10,
        max_episode_length=300, gamma=0.999, policy_epochs=5, batch_size=256, epsilon=0.2, environment_threads=10,
        device=torch.device('cuda'), lr=1e-3, weight_decay=0.01)


if __name__ == '__main__':
    main()
