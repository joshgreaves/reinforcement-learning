from functools import reduce
from itertools import chain
from threading import Thread
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from queue import Queue


class RLEnvironment(object):
    def __init__(self):
        super(RLEnvironment, self).__init__()

    def step(self, x):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class EnvironmentFactory(object):
    def __init__(self):
        super(EnvironmentFactory, self).__init__()

    def new(self):
        raise NotImplementedError()


class AdvantageDataset(Dataset):
    def __init__(self, experience):
        super(AdvantageDataset, self).__init__()
        self._exp = experience
        self._num_runs = len(experience)
        self._length = reduce(lambda acc, x: acc + len(x), experience, 0)

    def __getitem__(self, index):
        idx = 0
        seen_data = 0
        current_exp = self._exp[0]
        while seen_data + len(current_exp) - 1 < index:
            seen_data += len(current_exp)
            idx += 1
            current_exp = self._exp[idx]
        chosen_exp = current_exp[index - seen_data]
        return chosen_exp[0], chosen_exp[4]

    def __len__(self):
        return self._length


class PolicyDataset(Dataset):
    def __init__(self, experience):
        super(PolicyDataset, self).__init__()
        self._exp = experience
        self._num_runs = len(experience)
        self._length = reduce(lambda acc, x: acc + len(x), experience, 0)

    def __getitem__(self, index):
        idx = 0
        seen_data = 0
        current_exp = self._exp[0]
        while seen_data + len(current_exp) - 1 < index:
            seen_data += len(current_exp)
            idx += 1
            current_exp = self._exp[idx]
        chosen_exp = current_exp[index - seen_data]
        return chosen_exp

    def __len__(self):
        return self._length


def multinomial_likelihood(dist, idx):
    return dist[range(dist.shape[0]), idx]


def ppo(env_factory, policy, value, likelihood_fn, epochs=1000, rollouts_per_epoch=100, max_episode_length=200,
        gamma=0.99, policy_epochs=5, batch_size=256, epsilon=0.2, environment_threads=1, device=torch.device('cpu'),
        policy_lr=1e-3, policy_weight_decay=0.01):
    policy = policy.to(device)
    value = value.to(device)

    policy_optim = optim.Adam(chain(policy.parameters(), value.parameters()), lr=policy_lr,
                              weight_decay=policy_weight_decay)
    value_criteria = nn.MSELoss()

    ppo_lower_bound = 1 - epsilon
    ppo_upper_bound = 1 + epsilon

    loop = tqdm(total=epochs, position=0, leave=False)

    environments = [env_factory.new() for _ in range(environment_threads)]
    rollouts_per_thread = rollouts_per_epoch // environment_threads
    remainder = rollouts_per_epoch % environment_threads
    rollout_nums = ([rollouts_per_thread + 1] * remainder) + ([rollouts_per_thread] * (environment_threads - remainder))

    for _ in range(epochs):
        experience_queue = Queue()
        reward_queue = Queue()
        threads = [Thread(target=_run_envs, args=(environments[i], policy, experience_queue,
                                                  reward_queue, rollout_nums[i],
                                                  max_episode_length, device)) for i in range(environment_threads)]
        for x in threads:
            x.start()
        for x in threads:
            x.join()
        rollouts = list(experience_queue.queue)
        avg_r = sum(reward_queue.queue) / reward_queue.qsize()
        loop.set_description('avg reward: % 6.2f' % (avg_r))

        _calculate_returns(rollouts, gamma)

        # Learn a policy
        policy_dataset = PolicyDataset(rollouts)
        policy_loader = DataLoader(policy_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        avg_policy_loss = 0
        avg_val_loss = 0
        for _ in range(policy_epochs):
            for state, old_action_dist, old_action, reward, ret in policy_loader:
                state = state.detach().to(device)
                old_action_dist = old_action_dist.detach().to(device)
                old_action = old_action.detach()
                reward = reward.detach()
                ret = ret.detach().to(device)

                policy_optim.zero_grad()
                advantage = (ret - value(state))

                current_action_dist = policy(state, False)
                current_likelihood = likelihood_fn(current_action_dist, old_action)
                old_likelihood = likelihood_fn(old_action_dist, old_action)
                ratio = (current_likelihood.to(device) / old_likelihood).unsqueeze(1)

                expected_returns = value(state.to(device))
                val_loss = value_criteria(expected_returns, ret)
                avg_val_loss += val_loss.item()

                lhs = ratio * advantage
                rhs = torch.clamp(ratio, ppo_lower_bound, ppo_upper_bound) * advantage
                policy_loss = -torch.mean(torch.min(lhs, rhs))

                avg_policy_loss += policy_loss.item()

                loss = policy_loss + val_loss
                loss.backward()
                policy_optim.step()
            avg_val_loss /= len(policy_loader)
            avg_policy_loss /= len(policy_loader)
            loop.set_description(
                'avg reward: % 6.2f, value loss: % 6.2f, policy loss: % 6.2f' % (avg_r, avg_val_loss, avg_policy_loss))
        print()
        loop.update(1)


def _calculate_returns(trajectories, gamma):
    for i, trajectory in enumerate(trajectories):
        current_return = 0
        for j in reversed(range(len(trajectory))):
            state, action_dist, action, reward = trajectory[j]
            ret = reward + gamma * current_return
            trajectories[i][j] = (state, action_dist, action, reward, ret)
            current_return = ret


def _run_envs(env, policy, experience_queue, reward_queue, num_rollouts, max_episode_length, device):
    for _ in range(num_rollouts):
        current_rollout = []
        s = env.reset()
        episode_reward = 0
        for _ in range(max_episode_length):
            action_dist, action = policy(s.to(device))
            s_prime, r, t = env.step(action)

            current_rollout.append((s, action_dist, action, r))
            episode_reward += r.item()
            if t:
                break
            s = s_prime
        experience_queue.put(current_rollout)
        reward_queue.put(episode_reward)
