from tests.gtrxl_gauss_policy_test import TransformerGaussianPolicy
from torch.optim import SGD
import numpy as np
import torch


class BaseAgent:
    def __init__(self, **kwargs):
        raise NotImplementedError()

    def act(self, obs, reward):
        raise NotImplementedError()


class GridAgent:
    def __init__(self, state_dim, act_dim):
        self.policy = TransformerGaussianPolicy(state_dim, act_dim)
        self.criterion = torch.nn.MSELoss()

        self.optimizer = SGD(self.policy.parameters(), lr=0.01, momentum=0.9)

    def act(self, obs, reward):

        loss = self.criterion()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        policy = self.policy(obs)
        return policy[0].sample()
