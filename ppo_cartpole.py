from rl import *
from environments import CartPoleEnvironmentFactory
import networks as net


def main():
    factory = CartPoleEnvironmentFactory()
    policy = net.DiscreteActionSpaceNetwork(net.FourLayerMlp(4, 2, hidden_dim=10))
    value = net.FourLayerMlp(4, 1, hidden_dim=10)

    ppo(factory, policy, value, multinomial_likelihood, epochs=1000, rollouts_per_epoch=100, max_episode_length=200,
        gamma=0.99, policy_epochs=5, batch_size=256, experiment_name='cartpole_basic')


if __name__ == '__main__':
    main()

