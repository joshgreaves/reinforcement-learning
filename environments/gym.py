from copy import copy
import gym
import numpy as np

from rl import EnvironmentFactory
from rl import RLEnvironment


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


class HillClimbEnvironmentFactory(EnvironmentFactory):
    def __init__(self):
        super(HillClimbEnvironmentFactory, self).__init__()

    def new(self):
        return HillClimbEnvironment()


class HillClimbEnvironment(RLEnvironment):
    def __init__(self):
        super(HillClimbEnvironment, self).__init__()
        self._env = gym.make('MountainCar-v0')

    def step(self, action):
        """action is type np.ndarray of shape [1] and type np.uint8.
        Returns observation (np.ndarray), r (float), t (boolean)
        """
        s, r, t, _ = self._env.step(action.item())
        return s, r, t

    def reset(self):
        """Returns observation (np.ndarray)"""
        return self._env.reset()
