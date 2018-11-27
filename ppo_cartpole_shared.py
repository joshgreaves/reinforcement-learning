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
        """action is type np.ndarray of shape [1] and type np.uint8.
        Returns observation (np.ndarray), r (float), t (boolean)
        """
        s, r, t, _ = self._env.step(action.item())
        return s, r, t

    def reset(self):
        """Returns observation (np.ndarray)"""
        return self._env.reset()


class CartPolePolicyNetwork(nn.Module):
    """Policy Network for CartPole."""
    def __init__(self, state_dim=4, action_dim=2):
        super(CartPolePolicyNetwork, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, action_dim)
        )
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


class CartPoleValueNetwork(nn.Module):
    """Approximates the value of a particular CartPole state."""
    def __init__(self, state_dim=4):
        super(CartPoleValueNetwork, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        """Receives an observation of shape [batch, state_dim].
        Returns the value of each state, in shape [batch, 1]
        """
        return self._net(x)


def main():
    factory = CartPoleEnvironmentFactory()
    policy = CartPolePolicyNetwork()
    embedding = nn.Sequential(nn.Linear(4, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU())
    value = CartPoleValueNetwork()
    ppo(factory, policy, value, multinomial_likelihood, embedding_net=embedding, epochs=1000, rollouts_per_epoch=100, max_episode_length=200,
        gamma=0.99, policy_epochs=5, batch_size=256)


if __name__ == '__main__':
    main()

