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


class GPT(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        # MAE encoder specifics
        self.n_embd = config.n_embd
        self.max_len = config.traj_length * 2
        self.state_embed = nn.Linear(obs_dim, self.n_embd)
        self.action_embed = nn.Linear(action_dim, self.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, action_dim),
            nn.Tanh(),
        )  # decoder to patch
        self.state_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, obs_dim),
        )
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = utils.get_1d_sincos_pos_embed_from_grid(self.n_embd, self.max_len)
        pe = torch.from_numpy(pos_embed).float().unsqueeze(0)
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

    def forward(self, obs, action):
        batch_size, T, obs_dim = obs.size()
        # goal: batch_size, obs_dim
        s = self.state_embed(obs)
        a = self.action_embed(action)
        x = (
            torch.stack([s, a], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * T, self.n_embd)
        )
        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, self.attn_mask)

        a = self.action_head(x[:, ::2])
        s = self.state_head(x[:, 1::2])
        # apply Transformer blocks
        return s, a


class GPTAgent:
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
        self.model = GPT(obs_shape[0], action_shape[0], self.config).to(device)
        if path is not None:
            self.model.load_state_dict(payload["model"])
        # optimizers
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.norm = self.config.norm
        print(
            "number of parameters: %e", sum(p.numel() for p in self.model.parameters())
        )
        self.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def act(self, obs, action, step):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        action = torch.as_tensor(action, device=self.device).unsqueeze(0)
        action = self.model(obs, action)[:, -1]
        return action.cpu().numpy()[0]

    def update_actor(self, obs, action, next_obs, step):
        metrics = dict()
        batch_size, T, _ = obs.size()
        # sample init state and goal
        pred_s, pred_a = self.model(obs, action)

        # no normalization
        if self.norm == "l2":
            next_obs = next_obs / torch.norm(next_obs, dim=-1, keepdim=True)
        elif self.norm == "mae":
            mean = next_obs.mean(dim=-1, keepdim=True)
            var = next_obs.var(dim=-1, keepdim=True)
            next_obs = (next_obs - mean) / (var + 1.0e-6) ** 0.5

        loss_s = ((pred_s - next_obs) ** 2).mean()
        loss_a = ((pred_a - action) ** 2).mean()
        loss = loss_a + loss_s

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        if self.use_tb:
            metrics["state_loss"] = loss_s.item()
            metrics["action_loss"] = loss_a.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _ = utils.to_torch(batch, self.device)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update actor
        metrics.update(self.update_actor(obs, action, next_obs, step))
        return metrics
