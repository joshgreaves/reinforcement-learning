from functools import reduce
import imageio
from itertools import chain
from threading import Thread
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from queue import Queue


class RLEnvironment(object):
    """An RL Environment, used for wrapping environments to run PPO on."""
    def __init__(self):
        super(RLEnvironment, self).__init__()

    def step(self, x):
        """Takes an action x, which is the same format as the output from a policy network.
        Returns observation (np.ndarray), reward (float), terminal (boolean)
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the environment.
        Returns observation (np.ndarray)
        """
        raise NotImplementedError()


class EnvironmentFactory(object):
    """Creates new environment objects"""
    def __init__(self):
        super(EnvironmentFactory, self).__init__()

    def new(self):
        raise NotImplementedError()


class PolicyDataset(Dataset):
    def __init__(self, experience):
        super(PolicyDataset, self).__init__()
        self._exp = []
        for x in experience:
            self._exp.extend(x)
        self._length = len(self._exp)

    def __getitem__(self, index):
        return self._exp[index]

    def __len__(self):
        return self._length


def multinomial_likelihood(dist, idx):
    return dist[range(dist.shape[0]), idx.long()[:, 0]].unsqueeze(1)


def ppo(env_factory, policy, value, likelihood_fn, embedding_net=None, epochs=1000, rollouts_per_epoch=100,
        max_episode_length=200, gamma=0.99, policy_epochs=5, batch_size=256, epsilon=0.2, environment_threads=1,
        device=torch.device('cpu'), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, gif_epochs=0):
    # Move networks to the correct device
    policy = policy.to(device)
    value = value.to(device)
    embedding_net = embedding_net.to(device) if embedding_net else None

    # Collect parameters
    params = None
    if embedding_net:
        params = chain(policy.parameters(), value.parameters(), embedding_net.parameters())
    else:
        params = chain(policy.parameters(), value.parameters())

    # Set up optimization
    optimizer = optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    value_criteria = nn.MSELoss()

    # Calculate the upper and lower bound for PPO
    ppo_lower_bound = 1 - epsilon
    ppo_upper_bound = 1 + epsilon

    loop = tqdm(total=epochs, position=0, leave=False)

    # Prepare the environments
    environments = [env_factory.new() for _ in range(environment_threads)]
    rollouts_per_thread = rollouts_per_epoch // environment_threads
    remainder = rollouts_per_epoch % environment_threads
    rollout_nums = ([rollouts_per_thread + 1] * remainder) + ([rollouts_per_thread] * (environment_threads - remainder))

    for e in range(epochs):
        # Run the environments
        experience_queue = Queue()
        reward_queue = Queue()
        threads = [Thread(target=_run_envs, args=(environments[i], embedding_net, policy, experience_queue,
                                                  reward_queue, rollout_nums[i],
                                                  max_episode_length, device)) for i in range(environment_threads)]
        for x in threads:
            x.start()
        for x in threads:
            x.join()

        # Collect the experience
        rollouts = list(experience_queue.queue)
        avg_r = sum(reward_queue.queue) / reward_queue.qsize()
        loop.set_description('avg reward: % 6.2f' % (avg_r))
        _calculate_returns(rollouts, gamma)

        # Make gifs
        if gif_epochs and e % gif_epochs == 0:
            _make_gif(environments[0], embedding_net, policy, max_episode_length, device, '%d.gif' % e)

        # Update the policy
        experience_dataset = PolicyDataset(rollouts)
        data_loader = DataLoader(experience_dataset, num_workers=10, batch_size=batch_size, shuffle=True, pin_memory=True)
        avg_policy_loss = 0
        avg_val_loss = 0
        for _ in range(policy_epochs):
            for state, old_action_dist, old_action, reward, ret in data_loader:
                # Put all data on the correct device
                state = _prepare_tensor_batch(state, device)
                old_action_dist = _prepare_tensor_batch(old_action_dist, device)
                old_action = _prepare_tensor_batch(old_action, device)
                reward = _prepare_tensor_batch(reward, device).unsqueeze(1)
                ret = _prepare_tensor_batch(ret, device).unsqueeze(1)

                optimizer.zero_grad()
                
                # If there is an embedding net, carry out the embedding
                if embedding_net:
                    state = embedding_net(state)


                # Calculate the ratio term
                current_action_dist = policy(state, False)
                current_likelihood = likelihood_fn(current_action_dist, old_action)
                old_likelihood = likelihood_fn(old_action_dist, old_action)
                ratio = (current_likelihood / old_likelihood)

                # Calculate the value loss
                expected_returns = value(state)
                val_loss = value_criteria(expected_returns, ret)

                # Calculate the policy loss
                advantage = ret - expected_returns.detach()
                lhs = ratio * advantage
                rhs = torch.clamp(ratio, ppo_lower_bound, ppo_upper_bound) * advantage
                policy_loss = -torch.mean(torch.min(lhs, rhs))
                
                # For logging
                avg_val_loss += val_loss.item()
                avg_policy_loss += policy_loss.item()

                # Backpropagate
                loss = policy_loss + val_loss
                loss.backward()
                optimizer.step()

            # Log info
            avg_val_loss /= len(data_loader)
            avg_policy_loss /= len(data_loader)
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


def _run_envs(env, embedding_net, policy, experience_queue, reward_queue, num_rollouts, max_episode_length, device):
    for _ in range(num_rollouts):
        current_rollout = []
        s = env.reset()
        episode_reward = 0
        for _ in range(max_episode_length):
            input_state = s
            if embedding_net:
                input_state = embedding_net(_prepare_numpy(s, device))
            else:
                input_state = _prepare_numpy(s, device)

            action_dist, action = policy(input_state)
            action_dist, action = action_dist[0], action[0]  # Remove the batch dimension
            s_prime, r, t = env.step(action)

            current_rollout.append((s, action_dist.cpu().detach().numpy(), action, r))
            episode_reward += r
            if t:
                break
            s = s_prime
        experience_queue.put(current_rollout)
        reward_queue.put(episode_reward)


def _prepare_numpy(ndarray, device):
    return torch.from_numpy(ndarray).float().unsqueeze(0).to(device)


def _prepare_tensor_batch(tensor, device):
    return tensor.detach().float().to(device)


def _make_gif(env, embedding_net, policy, max_episode_length, device, filename):
    s = env.reset()
    with imageio.get_writer(filename, mode='I', duration=1 / 30) as writer:
        for _ in range(max_episode_length):
            writer.append_data((s[:, :, 0] * 255).astype(np.uint8))
            action_dist, action = policy(embedding_net(_prepare_numpy(s, device)))
            action_dist, action = action_dist[0], action[0]  # Remove the batch dimension
            s_prime, r, t = env.step(action)

            if t:
                break
            s = s_prime
