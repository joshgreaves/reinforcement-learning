from copy import copy
import numpy as np
import vizdoom as vzd

from . import EnvironmentFactory, RLEnvironment


class DoomHallwayFactory(EnvironmentFactory):
    def __init__(self):
        super(DoomHallwayFactory, self).__init__()

    def new(self):
        return DoomHallwayEnvironment()


class DoomHallwayEnvironment(RLEnvironment):
    def __init__(self, prev_frames=4, skiprate=1):
        super(DoomHallwayEnvironment, self).__init__()
        self._res = (120, 160)
        self._prev_frames = prev_frames
        self._skiprate = skiprate
        self._imgs = np.zeros((self._res[0], self._res[1], prev_frames), dtype=np.float32)

        self._game = vzd.DoomGame()
        self._game.set_doom_scenario_path('scenarios/deadly_corridor.wad')
        self._game.set_doom_map("map01")
        self._game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self._game.set_screen_format(vzd.ScreenFormat.RGB24)
        self._game.set_living_reward(0.0)
        self._game.set_doom_skill(5)

        self._game.add_available_button(vzd.Button.ATTACK)
        self._game.add_available_button(vzd.Button.MOVE_FORWARD)
        self._game.add_available_button(vzd.Button.MOVE_LEFT)
        self._game.add_available_button(vzd.Button.MOVE_RIGHT)
        self._game.add_available_button(vzd.Button.TURN_LEFT)
        self._game.add_available_button(vzd.Button.TURN_RIGHT)

        self._game.set_episode_timeout(3000)
        self._game.set_episode_start_time(10)
        self._game.set_window_visible(False)

        self._game.init()
        self._actions = [
            [True, False, False, False, False, False],
            [False, True, False, False, False, False],
            [False, False, True, False, False, False],
            [False, False, False, True, False, False],
            [False, False, False, False, True, False],
            [False, False, False, False, False, True],
        ]

    def step(self, action):
        action = action.item()
        r = self._game.make_action(self._actions[action], self._skiprate)
        # r = float(max(min(r, 1.0), -1.0))  # Clamp r to between -1 and 1
        r = float(r)

        t = self._game.is_episode_finished()

        if t:
            return self._imgs, -1.0, t

        state = self._game.get_state()
        screen_buf = state.screen_buffer
        self._process_state(screen_buf)

        if type(r) != float:
            print(type(r), r)
        return self._imgs, r, t

    def reset(self):
        self._imgs = np.zeros((self._res[0], self._res[1], self._prev_frames), dtype=np.float32)
        self._game.new_episode()
        state = self._game.get_state()
        screen_buf = state.screen_buffer
        self._process_state(screen_buf)
        return self._imgs

    def _process_state(self, s):
        self._imgs = np.roll(self._imgs, 1, axis=2)
        self._imgs[:, :, 0] = np.mean(s.astype(np.float32) / 255, axis=2)
