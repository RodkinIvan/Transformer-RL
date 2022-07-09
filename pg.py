# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque

from tests.gtrxl_gauss_policy_test import TransformerGaussianPolicy


# define policy network
class PolicyNet(nn.Module):
    def __init__(self, nS, nH, nA): # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
        super(PolicyNet, self).__init__()
        self.h = nn.Linear(nS, nH)
        self.out = nn.Linear(nH, nA)

    # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.softmax(self.out(x), dim=1)
        return x


Trans = True


def attempt(state, policy):
    if Trans:
        state = [state]
    probs = policy(torch.tensor(np.array(state)).unsqueeze(0).float())
    # sample an action from that set of probs

    if Trans:
        sampler = probs[0]
        action = sampler.sample()
    else:
        sampler = Categorical(probs)
        action = sampler.sample()

    return state, action, sampler


# create environment
env = gym.make("CartPole-v1")
# instantiate the policy
if Trans:
    policy = TransformerGaussianPolicy(env.observation_space.shape[0], 1, batch_sz=1)
else:
    policy = PolicyNet(env.observation_space.shape[0], 20, env.action_space.n)

# create an optimizer
optimizer = torch.optim.Adam(policy.parameters())
# initialize gamma and stats
gamma = 0.99
n_episode = 1
returns = deque(maxlen=100)
render_rate = 1000  # render every render_rate episodes

for p in policy.parameters():
    print(p)

for i in range(10000):
    rewards = []
    actions = []
    states = []
    # reset environment
    state = env.reset()
    while True:
        # render episode every render_rate epsiodes
        if n_episode % render_rate == 0:
            env.render()

        # calculate probabilities of taking each action
        state, action, sampler = attempt(state, policy)

        # use that action in the environment
        new_state, reward, done, info = env.step(action.item())
        # store state, action and reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = new_state
        if done:
            break

    # preprocess rewards
    rewards = np.array(rewards)
    # calculate rewards to go for less variance
    R = torch.tensor([np.sum(rewards[i:]*(gamma**np.array(range(i, len(rewards))))) for i in range(len(rewards))])
    # or uncomment following line for normal rewards
    # R = torch.sum(torch.tensor(rewards))

    # preprocess states and actions
    states = torch.tensor(np.array(states)).float()
    actions = torch.tensor(actions)

    # calculate gradient
    probs = policy(states)

    if Trans:
        sampler = probs[0]
    else:
        sampler = Categorical(probs)

    log_probs = -sampler.log_prob(actions)  # "-" because it was built to work with gradient descent,
    # but we are using gradient ascent
    pseudo_loss = torch.sum(log_probs * R)  # loss that when differentiated with autograd gives the gradient of J(θ)
    # update policy weights
    optimizer.zero_grad()
    pseudo_loss.backward()
    optimizer.step()

    # calculate average return and print it out
    returns.append(np.sum(rewards))
    print("Episode: {:6d}\tAvg. Return: {:6.2f}".format(n_episode, np.mean(returns)))
    n_episode += 1

# close environment
env.close()
for p in policy.parameters():
    print(p)