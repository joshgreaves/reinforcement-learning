from copy import copy
import numpy as np
import holodeck
from holodeck.sensors import Sensors

from . import EnvironmentFactory, RLEnvironment


class MazeWorldEnvironmentFactory(EnvironmentFactory):
    def __init__(self, res=(128, 128)):
        super(MazeWorldEnvironmentFactory, self).__init__()
        self._res = res

    def new(self):
        return MazeWorldEnvironment(res=self._res)


class MazeWorldEnvironment(RLEnvironment):
    def __init__(self, res=(128, 128), prev_frames=4):
        super(MazeWorldEnvironment, self).__init__()
        self._env = holodeck.make('MazeWorld', window_res=(256, 256), cam_res=res)
        self._res = res
        self._prev_frames = prev_frames
        self._imgs = np.zeros((res[0], res[1], prev_frames), dtype=np.float32)
        self._interval = 1.0
        self._next_interval = self._interval

    def step(self, action):
        s, _, _, _ = self._env.step(action.item())

        r = 0.0
        if s[Sensors.LOCATION_SENSOR][0] >= self._next_interval:
            r += 1
            self._next_interval += self._interval

        self._process_state(s)
        return self._imgs, r, False

    def reset(self):
        self._imgs = np.zeros((self._res[0], self._res[1], self._prev_frames), dtype=np.float32)
        s, _, _, _ = self._env.reset()
        self._process_state(s)
        self._next_interval = self._interval
        return self._imgs

    def _process_state(self, s):
        self._imgs = np.roll(self._imgs, 1, axis=2)
        normalized = copy(s[Sensors.RGB_CAMERA][:, :, :3]).astype(np.float32) / 255.0

        self._imgs[:, :, 0] = 0.2126 * normalized[:, :, 0] + \
                              0.7152 * normalized[:, :, 1] + \
                              0.0722 * normalized[:, :, 2] - 0.5
