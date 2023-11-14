import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils
from dm_control.utils import rewards
from einops import rearrange, reduce, repeat
from agent.modules.attention import Block


class GoalDT(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        # MAE encoder specifics
        self.n_embd = config.n_embd
        self.max_len = config.traj_length
        self.state_embed = nn.Linear(obs_dim * 2, self.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, action_dim),
            nn.Tanh(),
        )  # decoder to patch
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = utils.get_1d_sincos_pos_embed_from_grid(self.n_embd, self.max_len)
        pe = torch.from_numpy(pos_embed).float().unsqueeze(0) / 2.0
        self.register_buffer("pos_embed", pe)
        self.register_buffer(
            "attn_mask",
            torch.tril(torch.ones(self.max_len, self.max_len))[None, None, ...],
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, obs, goal, std):
        batch_size, T, obs_dim = obs.size()
        goal = goal.repeat(1, T, 1)
        # goal: batch_size, obs_dim
        x = torch.cat((obs, goal), dim=-1)
        x = self.state_embed(x) + self.pos_embed[:, :T]
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, self.attn_mask)
        x = self.action_head(x)
        return x


class GoalDTAgent:
    def __init__(
        self,
        name,
        obs_shape,
        action_shape,
        device,
        lr,
        batch_size,
        stddev_schedule,
        use_tb,
        transformer_cfg,
        path=None,
    ):
        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.use_tb = use_tb
        self.stddev_schedule = stddev_schedule

        # init from snapshot
        payload = None
        if path is not None:
            print("loading existing model...")
            payload = torch.load(path)
            self.config = payload["cfg"]
        else:
            self.config = transformer_cfg
        self.model = GoalDT(obs_shape[0], action_shape[0], self.config).to(device)
        if path is not None:
            self.model.load_state_dict(payload["model"])
        # optimizers
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(
            "number of parameters: %e", sum(p.numel() for p in self.model.parameters())
        )
        self.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def act(self, obs, goal, step):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).unsqueeze(0).unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, step)
        action = self.model(obs, goal, stddev)
        action = action[:, -1]
        return action.cpu().numpy()[0]

    def update_actor(self, obs, action, step):
        metrics = dict()
        batch_size, T, _ = obs.size()
        # sample init state and goal
        goal = obs[:, -1].unsqueeze(1)
        obs = obs[:, :-1]
        stddev = utils.schedule(self.stddev_schedule, step)
        pred_action = self.model(obs, goal, stddev)
        actor_loss = ((pred_action - action[:, :-1]) ** 2).mean()
        self.opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()

        return metrics

    def eval_validation(self, val_iter, step=None):
        metrics = dict()
        batch = next(val_iter)
        obs, action, _, _, _, _ = utils.to_torch(batch, self.device)

        batch_size, T, _ = obs.size()
        goal = obs[:, -1].unsqueeze(1)
        obs = obs[:, :-1]
        stddev = utils.schedule(self.stddev_schedule, step)
        pred_action = self.model(obs, goal, stddev)

        actor_loss = ((pred_action - action[:, :-1]) ** 2).mean()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _ = utils.to_torch(batch, self.device)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update actor
        metrics.update(self.update_actor(obs, action, step))
        return metrics
