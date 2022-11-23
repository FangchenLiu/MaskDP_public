import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import functools

import utils
from dm_control.utils import rewards


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(obs_dim*2, hidden_dim),
                                    nn.LayerNorm(hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, goal, std):
        x = torch.cat((obs, goal), dim=-1)
        mu = self.policy(x)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class GoalBCAgent:
    def __init__(self,
                 name,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 hidden_dim,
                 batch_size,
                 stddev_schedule,
                 use_tb,
                 transformer_cfg,
                 path=None,
                 has_next_action=False):
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.stddev_schedule = stddev_schedule
        self.use_tb = use_tb

        # models
        # init from snapshot
        payload = None
        if path is not None:
            print('loading existing model...')
            payload = torch.load(path)
            self.config = payload['cfg']
        else:
            self.config = transformer_cfg

        self.model = Actor(obs_shape[0], action_shape[0],
                           hidden_dim).to(device)
        print("number of parameters: %e", sum(p.numel() for p in self.model.parameters()))
        if path is not None:
            self.model.load_state_dict(payload['model'])

        # optimizers
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def act(self, obs, goal, step):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.model(obs, goal, stddev)
        action = policy.mean
        return action.cpu().numpy()[0]

    def update_actor(self, obs, action, step):
        metrics = dict()
        batch_size, T, _ = obs.size()
        # sample init state and goal
        start_t = np.random.randint(0, T-1)
        goal_t = np.random.randint(start_t, T)

        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.model(obs[:, start_t], obs[:, goal_t], stddev)

        log_prob = policy.log_prob(action[:, start_t]).sum(-1, keepdim=True)
        actor_loss = (-log_prob).mean()

        self.model_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.model_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_ent'] = policy.entropy().sum(dim=-1).mean().item()

        return metrics

    def eval_validation(self, val_iter, step=None):
        metrics = dict()
        batch = next(val_iter)
        obs, action, _, _, _, _ = utils.to_torch(
            batch, self.device)

        batch_size, T, _ = obs.size()
        start_t = np.random.randint(0, T-1)
        goal_t = np.random.randint(start_t, T)

        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.model(obs[:, start_t], obs[:, goal_t], stddev)

        log_prob = policy.log_prob(action[:, start_t]).sum(-1, keepdim=True)
        actor_loss = (-log_prob).mean()

        if self.use_tb:
            metrics['val_actor_loss'] = actor_loss.item()
            metrics['val_actor_ent'] = policy.entropy().sum(dim=-1).mean().item()
        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _ = utils.to_torch(
            batch, self.device)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update actor
        metrics.update(self.update_actor(obs, action, step))

        return metrics
