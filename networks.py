import numpy as np
import torch
from torch import nn


class FourLayerMlp(nn.Module):
    """Basic MLP with discrete actions"""

    def __init__(self, state_dim, action_dim, hidden_dim=100):
        super(FourLayerMlp, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self._net(x)


class DiscreteActionSpaceNetwork(nn.Module):
    def __init__(self, net):
        super(DiscreteActionSpaceNetwork, self).__init__()
        self._net = net
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x, get_action=True):
        """Receives input x of shape [batch, state_dim].
        Outputs action distribution (categorical distribution) of shape [batch, action_dim],
        as well as a sampled action (optional).
        """
        scores = self._net(x)
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


class ConvNetwork128(nn.Module):
    def __init__(self, input_height, input_width, input_channels):
        super(ConvNetwork128, self).__init__()
        self._input_height = input_height
        self._input_width = input_width
        output_height = input_height / 8
        output_width = input_width / 8
        self._output_height = int(output_height)
        self._output_width = int(output_width)

        if output_height != self._output_height or output_width != self._output_width:
            raise ValueError('Input height/width not compatible with ConvNetwork128')

        self._conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),  # -> [n, 32, 128, 128]
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
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute((0, 3, 1, 2))
        activations = self._conv(x)
        return activations.view((-1, 256 * self._output_height * self._output_width))

    def get_output_dim(self):
        return 256 * self._output_height * self._output_width
