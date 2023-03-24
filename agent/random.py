import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils


class Actor(nn.Module):
    def __init__(self, action_shape):
        super().__init__()
        self.low = -1.0 * torch.ones(action_shape[0])
        self.high = 1.0 * torch.ones(action_shape[0])

    def forward(self, obs):
        dist = torch.distributions.Uniform(self.low, self.high)
        return dist


class RandomAgent:
    def __init__(self, name, obs_type, obs_shape, action_shape, num_expl_steps,
                 batch_size, nstep, supervised, bonus):
        self.action_shape = action_shape
        self.actor = Actor(action_shape)
        self.train()

    def train(self, training=True):
        self.training = training

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, meta, step, eval_mode):
        dist = self.actor(obs)
        action = dist.sample()
        return action.cpu().numpy()

    def update(self, replay_iter, step):
        metrics = dict()
        return metrics
