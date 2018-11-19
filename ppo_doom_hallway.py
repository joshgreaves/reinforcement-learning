import numpy as np
import torch
import torch.nn as nn
import vizdoom as vzd

from rl import *


class DoomEnvironmentFactory(EnvironmentFactory):
    def __init__(self):
        super(DoomEnvironmentFactory, self).__init__()

    def new(self):
        return DoomEnvironment()


class DoomEnvironment(RLEnvironment):
    def __init__(self, prev_frames=4, skiprate=1):
        super(DoomEnvironment, self).__init__()
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
        self._game.add_available_button(vzd.Button.MOVE_LEFT)
        self._game.add_available_button(vzd.Button.MOVE_RIGHT)
        self._game.add_available_button(vzd.Button.TURN_LEFT)
        self._game.add_available_button(vzd.Button.TURN_RIGHT)

        self._game.set_episode_timeout(3000)
        self._game.set_episode_start_time(10)
        self._game.set_window_visible(True)

        self._game.init()
        self._actions = [
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, True, False],
            [False, False, False, False, True]
        ]

    def step(self, action):
        action = action.item()
        r = self._game.make_action(self._actions[action], self._skiprate)
        r = float(max(min(r, 1.0), -1.0))  # Clamp r to between -1 and 1

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


class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        # Input image: [n, 1, 128, 128]
        self._conv = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),  # -> [n, 64, 120, 160]
            nn.ReLU(),
            nn.Conv2d(64, 64, 2, stride=2),  # -> [n, 64, 60, 80]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  # -> [n, 64, 60, 80]
            nn.ReLU(),
            nn.Conv2d(64, 128, 2, stride=2),  # -> [n, 128, 30, 40]
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),  # -> [n, 128, 30, 40]
            nn.ReLU(),
            nn.Conv2d(128, 256, 2, stride=2),  # -> [n, 256, 15, 20]
        )

    def forward(self, x):
        x = x.permute((0, 3, 1, 2))
        activations = self._conv(x)
        return activations.view((-1, 256 * 15 * 20))


class PolicyNetwork(nn.Module):
    def __init__(self, action_dim=5, hidden_dim=100):
        super(PolicyNetwork, self).__init__()
        self._fc = nn.Sequential(
            nn.Linear(256 * 15 * 20, hidden_dim),
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
            nn.Linear(256 * 15 * 20, hidden_dim),
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
    factory = DoomEnvironmentFactory()
    conv = ConvNetwork()
    policy = PolicyNetwork()
    value = ValueNetwork()
    ppo(factory, policy, value, multinomial_likelihood, embedding_net=conv, epochs=1000, rollouts_per_epoch=200,
        max_episode_length=3000, gamma=0.999, policy_epochs=1, batch_size=256, epsilon=0.2, environment_threads=5,
        data_loader_threads=10, device=torch.device('cuda'), lr=1e-3, weight_decay=0.0, gif_epochs=5)


if __name__ == '__main__':
    main()
