import gym
import numpy as np
import torch
import torch.nn as nn

from rl import *


class CartPoleEnvironmentFactory(EnvironmentFactory):
    def __init__(self):
        super(CartPoleEnvironmentFactory, self).__init__()

    def new(self):
        return CartPoleEnvironment()


class CartPoleEnvironment(RLEnvironment):
    def __init__(self):
        super(CartPoleEnvironment, self).__init__()
        self._env = gym.make('CartPole-v0')

    def step(self, action):
        s, r, t, _ = self._env.step(action)
        return torch.from_numpy(s).float(), torch.tensor((r)).float().unsqueeze(0), t

    def reset(self):
        return torch.from_numpy(self._env.reset()).float()


class CartPolePolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super(CartPolePolicyNetwork, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(state_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, action_dim)
        )
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x, get_action=True):
        dim = len(x.shape)
        if dim == 1:
            x = x.unsqueeze(0)
        scores = self._net(x)
        probs = self._softmax(scores)
        if dim == 1:
            probs = probs.squeeze(0)
        if get_action:
            action_one_hot = np.random.multinomial(1, probs.detach().numpy().reshape((-1)))
            action_idx = np.argmax(action_one_hot)
            return probs, action_idx
        return probs


class CartPoleValueNetwork(nn.Module):
    def __init__(self, state_dim=4):
        super(CartPoleValueNetwork, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(state_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self._net(x)


def main():
    factory = CartPoleEnvironmentFactory()
    policy = CartPolePolicyNetwork()
    value = CartPoleValueNetwork()
    ppo(factory, policy, value, multinomial_likelihood, epochs=1000, rollouts_per_epoch=100, max_episode_length=200,
        gamma=0.99, policy_epochs=5, batch_size=256)


if __name__ == '__main__':
    main()

