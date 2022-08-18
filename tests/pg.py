# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque

from GTrXL.gtrxl import GTrXL
from torch.distributions import Categorical

import wandb


# define policy network
class PolicyNet(nn.Module):
    def __init__(self, nS, nH, nA):  # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
        super(PolicyNet, self).__init__()
        self.h = nn.Linear(nS, nH)
        self.out = nn.Linear(nH, nA)

    # define forward pass with one hidden layer with ReLU activation and softmax after output layer
    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.softmax(self.out(x), dim=1)
        return x


class TransformerPolicy(torch.nn.Module):
    def __init__(self, state_dim, act_dim, batch_sz, n_transformer_layers=1, n_attn_heads=1):
        '''
            NOTE - I/P Shape : [seq_len, batch_size, state_dim]
        '''
        super(TransformerPolicy, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.transformer = GTrXL(input_dim=state_dim, head_num=n_attn_heads, layer_num=n_transformer_layers,
                                 embedding_dim=4)
        self.memory = None

        self.head_act_mean = torch.nn.Linear(state_dim, act_dim)

    def forward(self, state):
        trans_state = self.transformer(state)['logit']
        probs = F.softmax(self.head_act_mean(trans_state), dim=-1)

        return probs


Trans = True


def attempt(state, policy):
    if Trans:
        state = [state]
    probs = policy(torch.tensor(state).unsqueeze(0).float())
    # sample an action from that set of probs
    sampler = Categorical(probs)
    action = sampler.sample()
    return state, action, sampler


wandb.init(project='Transformer_RL', entity='irodkin')
# create environment
env = gym.make("CartPole-v1")
# instantiate the policy
if Trans:
    policy = TransformerPolicy(env.observation_space.shape[0], env.action_space.n, batch_sz=1)
else:
    policy = PolicyNet(env.observation_space.shape[0], 20, env.action_space.n)

# create an optimizer
optimizer = torch.optim.Adam(policy.parameters())
# initialize gamma and stats
gamma = 0.99
n_episode = 1
returns = deque(maxlen=100)
render_rate = 500  # render every render_rate episodes

for p in policy.parameters():
    print(p)

for i in range(2000):
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
    R = torch.tensor([np.sum(rewards[i:] * (gamma ** np.array(range(0, len(rewards)-i)))) for i in range(len(rewards))])
    # or uncomment following line for normal rewards
    # R = torch.sum(torch.tensor(rewards))

    # preprocess states and actions
    states = torch.tensor(np.array(states)).float()
    actions = torch.tensor(actions)

    # calculate gradient
    probs = policy(states)

    sampler = Categorical(probs[:, 0, :] if Trans else probs)

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
    wandb.log({'Avg reward': np.mean(returns)})
    n_episode += 1

# close environment
env.close()
wandb.finish()
