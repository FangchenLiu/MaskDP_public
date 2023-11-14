import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils
from dm_control.utils import rewards
from einops import rearrange, reduce, repeat
from agent.modules.attention import Block, CausalSelfAttention
from agent.mdp import MaskedDP


class MDPPromptAgent:
    def __init__(
        self,
        name,
        obs_shape,
        action_shape,
        device,
        lr,
        stddev_schedule,
        stddev_clip,
        batch_size,
        use_tb,
        context_length,
        forecast_length,
        transformer_cfg,
        path,
    ):
        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.use_tb = use_tb
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.context_length = context_length
        self.forecast_length = forecast_length

        # init from snapshot
        payload = None
        if path is not None:
            print("loading existing model...")
            payload = torch.load(path)
            self.config = payload["cfg"]
        else:
            self.config = transformer_cfg
        self.mdp = MaskedDP(obs_shape[0], action_shape[0], self.config).to(device)
        print("number of parameters: %e", sum(p.numel() for p in self.mdp.parameters()))
        if path is not None:
            self.mdp.load_state_dict(payload["model"])

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, self.context_length + self.forecast_length, self.config.n_embd
            ),
            requires_grad=False,
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1, self.context_length + self.forecast_length, self.config.n_embd
            ),
            requires_grad=False,
        )
        self.get_pe()
        self.train()

    def policy_dist(self, mean, std):
        std = torch.ones_like(mean) * std
        dist = utils.TruncatedNormal(mean, std)
        return dist

    def get_pe(self):
        T = self.context_length + self.forecast_length
        if 2 * T > self.mdp.pos_embed.shape[1]:
            self.pos_embed = utils.interpolate_pos_embed(self.mdp.pos_embed, 2 * T)
            self.decoder_pos_embed = utils.interpolate_pos_embed(
                self.mdp.decoder_pos_embed, 2 * T
            )
            self.attn_mask = torch.ones(2 * T, 2 * T)[None, None, ...].to(self.device)
        else:
            self.pos_embed = self.mdp.pos_embed
            self.decoder_pos_embed = self.mdp.decoder_pos_embed
            self.attn_mask = self.mdp.attn_mask

    def train(self, training=True):
        self.training = training
        self.mdp.train(training)

    def update(self, replay_iter):
        raise NotImplementedError

    def act(
        self, obs, context_s, context_a, global_step, eval_mode=True, generate_all=False
    ):
        obs = obs.unsqueeze(0).unsqueeze(0)
        T = context_s.shape[0]
        # timestep = torch.as_tensor(timestep, device=self.device).unsqueeze(0).unsqueeze(-1)
        stddev = utils.schedule(self.stddev_schedule, global_step)
        obs = self.mdp.state_embed(obs)
        context_state = self.mdp.state_embed(context_s).unsqueeze(0)
        context_action = self.mdp.action_embed(context_a).unsqueeze(0)
        context = (
            torch.stack([context_state, context_action], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(1, 2 * T, self.config.n_embd)
        )
        x = torch.cat((context, obs), dim=1) + self.pos_embed[:, : 2 * T + 1]

        for blk in self.mdp.encoder_blocks:
            x = blk(x, self.attn_mask)
        x = self.mdp.encoder_norm(x)

        # decoder
        mask_token = self.mdp.mask_token
        if generate_all is False:
            x = torch.cat([x, mask_token], dim=1)
            states = self.mdp.decoder_state_embed(x[:, ::2])
            actions = self.mdp.decoder_action_embed(x[:, 1::2])
        else:
            mask_states = self.mdp.mask_token.repeat(
                obs.shape[0], self.forecast_length - 1, 1
            )
            mask_actions = self.mdp.mask_token.repeat(
                obs.shape[0], self.forecast_length, 1
            )
            actions = torch.cat((x[:, 1::2], mask_actions), dim=1)
            states = torch.cat([x[:, ::2], mask_states], dim=1)
            assert self.context_length + self.forecast_length == actions.shape[1]
            assert self.context_length + self.forecast_length == states.shape[1]
            states = self.mdp.decoder_state_embed(states)
            actions = self.mdp.decoder_action_embed(actions)

        x = (
            torch.stack([states, actions], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(1, 2 * actions.shape[1], self.config.n_embd)
            + self.decoder_pos_embed[:, : 2 * actions.shape[1]]
        )

        # apply Transformer blocks
        for blk in self.mdp.decoder_blocks:
            x = blk(x, self.attn_mask)

        if generate_all is False:
            action_mean = self.mdp.action_head(x[:, -1])
        else:
            action_mean = self.mdp.action_head(x[:, 2 * T + 1 :: 2])
        policy = self.policy_dist(action_mean, stddev)

        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample(clip=self.stddev_clip)
        return action[0]

    def act_once(self, obs, context_s, context_a, global_step, budget, eval_mode=True):
        obs = obs.unsqueeze(0).unsqueeze(0)
        T = context_s.shape[0]
        # timestep = torch.as_tensor(timestep, device=self.device).unsqueeze(0).unsqueeze(-1)
        stddev = utils.schedule(self.stddev_schedule, global_step)
        obs = self.mdp.state_embed(obs)
        context_state = self.mdp.state_embed(context_s).unsqueeze(0)
        context_action = self.mdp.action_embed(context_a).unsqueeze(0)
        context = (
            torch.stack([context_state, context_action], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(1, 2 * T, self.config.n_embd)
        )
        x = torch.cat((context, obs), dim=1) + self.pos_embed[:, : 2 * T + 1]

        for blk in self.mdp.encoder_blocks:
            x = blk(x, self.attn_mask)
        x = self.mdp.encoder_norm(x)

        if budget > 1:
            mask_states = self.mdp.mask_token.repeat(obs.shape[0], budget - 1, 1)
            mask_actions = self.mdp.mask_token.repeat(obs.shape[0], budget, 1)
            actions = torch.cat((x[:, 1::2], mask_actions), dim=1)
            states = torch.cat([x[:, ::2], mask_states], dim=1)
        else:
            mask_token = self.mdp.mask_token
            x = (
                torch.cat([x, mask_token], dim=1)
                + self.decoder_pos_embed[:, : 2 * (T + 1)]
            )
            states = x[:, ::2]
            actions = x[:, 1::2]

        states = self.mdp.decoder_state_embed(states)
        actions = self.mdp.decoder_action_embed(actions)
        x = (
            torch.stack([states, actions], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(1, 2 * actions.shape[1], self.config.n_embd)
            + self.decoder_pos_embed[:, : 2 * actions.shape[1]]
        )

        # apply Transformer blocks
        for blk in self.mdp.decoder_blocks:
            x = blk(x, self.attn_mask)

        action_mean = self.mdp.action_head(x[:, 2 * T + 1])
        policy = self.policy_dist(action_mean, stddev)

        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample(clip=self.stddev_clip)
        return action[0]
