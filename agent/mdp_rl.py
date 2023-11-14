import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils
from dm_control.utils import rewards
from einops import rearrange, reduce, repeat
from agent.modules.attention import Block, CausalSelfAttention, mySequential
from agent.mdp import MaskedDP
import math


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, attention_length, finetune, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.max_len = attention_length
        print("attn length", self.max_len)
        self.mdp = MaskedDP(obs_dim, action_dim, config)
        self.ln = nn.LayerNorm(self.n_embd)
        self.action_head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.n_embd, self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, action_dim),
            nn.Tanh(),
        )
        self.config = config
        self.finetune = finetune
        self.apply(utils.weight_init)
        self.register_buffer(
            "attn_mask",
            torch.tril(torch.ones(self.max_len, self.max_len))[None, None, ...],
        )

    def freeze_layers(self):
        for param in self.mdp.parameters():
            param.requires_grad = True

        # if self.finetune == 'encoder':
        #     for b in self.mdp.encoder_blocks:
        #         for param in b.parameters():
        #             param.requires_grad = True

    def forward(self, obs_seq, std):
        batch_size, T, obs_dim = obs_seq.size()
        x = self.mdp.state_embed(obs_seq) + self.mdp.pos_embed[:, :T]
        # into attention
        for blk in self.mdp.encoder_blocks:
            x = blk(x, self.attn_mask)
        # x = self.mdp.encoder_norm(x)
        # # in decoder
        # x = self.mdp.decoder_state_embed(x) + self.mdp.decoder_pos_embed[:, :T]
        # for blk in self.mdp.decoder_blocks:
        #     x = blk(x, self.attn_mask)

        x = self.action_head(self.ln(x))

        std = torch.ones_like(x) * std
        dist = utils.TruncatedNormal(x, std)
        return dist

class TwinQHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.n_embd
        self.ln = nn.LayerNorm(hidden_dim)
        self.q1 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(utils.weight_init)

    def forward(self, x):
        # task = task.unsqueeze(1)
        # task = task.repeat(1, x.shape[1], 1)
        # x = torch.cat([self.ln(x), task], dim=-1)
        x = self.ln(x)
        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, attention_length, finetune, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.max_len = attention_length
        print("attn length", self.max_len)
        self.finetune = finetune
        self.mdp = MaskedDP(obs_dim, action_dim, config)
        # q1 and q2
        self.q = TwinQHead(config)
        self.target_q = TwinQHead(config)
        for param in self.target_q.parameters():
            param.requires_grad = False
        self.config = config
        self.register_buffer(
            "attn_mask",
            torch.tril(torch.ones(self.max_len * 2, self.max_len * 2))[None, None, ...],
        )
        self.apply(utils.weight_init)

    def freeze_layers(self):
        for param in self.mdp.parameters():
            param.requires_grad = True

        # if self.finetune == 'encoder':
        #     for b in self.mdp.encoder_blocks:
        #         for param in b.parameters():
        #             param.requires_grad = True

    def forward(self, obs_seq, action_seq, target=False):
        batch_size, T, obs_dim = obs_seq.size()
        state = self.mdp.state_embed(obs_seq)
        action = self.mdp.action_embed(action_seq)
        x = (
            torch.stack([state, action], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * T, self.config.n_embd)
        )
        x += self.mdp.pos_embed[:, : 2 * T]
        # encoder
        for blk in self.mdp.encoder_blocks:
            x = blk(x, self.attn_mask)
        # x = self.mdp.encoder_norm(x)
        #
        # # decoder preprocess
        # s = self.mdp.decoder_state_embed(x[:, ::2])
        # a = self.mdp.decoder_action_embed(x[:, 1::2])
        # x = torch.stack([s, a], dim=1).permute(0, 2, 1, 3).reshape_as(x)
        # x += self.mdp.decoder_pos_embed[:, :2*T]
        # # decoder
        # for blk in self.mdp.decoder_blocks:
        #     x = blk(x, self.attn_mask)
        if target is True:
            q1, q2 = self.target_q(x[:, 1::2])
            return q1.detach(), q2.detach(), "target"
        else:
            q1, q2 = self.q(x[:, 1::2])
            return q1, q2


class MTMDPAgent:
    def __init__(
        self,
        name,
        obs_shape,
        action_shape,
        device,
        lr,
        critic_target_tau,
        stddev_schedule,
        nstep,
        batch_size,
        stddev_clip,
        use_tb,
        finetune_actor,
        finetune_critic,
        attn_length,
        transformer_cfg,
        has_next_action=False,
        path=None,
    ):
        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        # models
        # init from snapshot
        payload = None
        if path is not None:
            print("loading existing model...")
            payload = torch.load(path)
            self.config = payload["cfg"]
        else:
            self.config = transformer_cfg
        self.attention_length = attn_length

        self.actor = Actor(
            obs_shape[0],
            action_shape[0],
            self.attention_length,
            finetune_actor,
            self.config,
        ).to(device)
        self.critic = Critic(
            obs_shape[0],
            action_shape[0],
            self.attention_length,
            finetune_critic,
            self.config,
        ).to(device)
        self.critic.target_q.load_state_dict(self.critic.q.state_dict())
        if path is not None:
            self.actor.mdp.load_state_dict(payload["model"])
            self.critic.mdp.load_state_dict(payload["model"])
        # optimizers
        self.actor.freeze_layers()
        self.critic.freeze_layers()
        self.actor_opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.actor.parameters()), lr=self.lr
        )
        self.critic_opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.critic.parameters()), lr=self.lr
        )
        actor_param = sum(
            p.numel()
            for p in filter(lambda p: p.requires_grad, self.actor.parameters())
        )
        critic_param = sum(
            p.numel()
            for p in filter(lambda p: p.requires_grad, self.critic.parameters())
        )
        print("number of parameters to be tuned: %e, %e", actor_param, critic_param)
        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, global_step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, global_step)
        policy = self.actor(obs, stddev)
        if eval_mode:
            action = policy.mean[:, -1]
        else:
            action = policy.sample(clip=None)[:, -1, :]
            if global_step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, next_obs, reward, discount, step):
        metrics = dict()
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2, _ = self.critic(next_obs, next_action, target=True)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action, target=False)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    def update_actor(self, obs, action, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, stddev)
        Q1, Q2 = self.critic(obs, policy.sample(clip=self.stddev_clip), target=False)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_ent"] = policy.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _ = utils.to_torch(batch, self.device)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, next_obs, reward, discount, step)
        )

        # update actor
        metrics.update(self.update_actor(obs, action, step))

        # update critic target
        utils.soft_update_params(
            self.critic.q, self.critic.target_q, self.critic_target_tau
        )

        return metrics
