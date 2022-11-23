import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils
from dm_control.utils import rewards
from einops import rearrange, reduce, repeat
from agent.modules.attention import Block, CausalSelfAttention, TwinQ, mySequential
import math

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.max_len = config.traj_length
        self.embed_state = nn.Linear(obs_dim, self.n_embd)
        self.blocks = mySequential(*[Block(config) for _ in range(config.n_layer)])
        self.policy_head = nn.Sequential(nn.Tanh(), nn.Linear(self.n_embd, action_dim))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.config = config
        self.apply(utils.weight_init)
        if self.config.mask_type == 'causal':
            self.register_buffer('actor_mask', torch.tril(torch.ones(self.max_len, self.max_len))[None, None, ...])
        elif self.config.mask_type == 'markovian':
            self.register_buffer('actor_mask', torch.diag(torch.ones(self.max_len))[None, None, ...])
        elif self.config.mask_type == 'random':
            # actor should always be at least causal
            self.register_buffer('actor_mask', torch.tril(torch.ones(self.max_len, self.max_len))[None, None, ...])
        else:
            raise NotImplementedError

    def forward(self, obs_seq, std, pos_emb):
        batch_size, T, obs_dim = obs_seq.size()
        state_emb = self.embed_state(obs_seq)
        x = self.blocks(self.drop(state_emb + pos_emb[:, :T]), self.actor_mask)
        mu = self.policy_head(self.ln_f(x))
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.max_len = config.traj_length
        self.embed_state = nn.Linear(obs_dim, self.n_embd)
        self.embed_action = nn.Linear(action_dim, self.n_embd)
        # q1 and q2
        self.q = TwinQ(config)
        self.target_q = TwinQ(config)
        self.config = config
        self._init_mask()
        self.apply(utils.weight_init)

    def _init_mask(self):
        if self.config.mask_type == 'causal':
            self.register_buffer('critic_mask', torch.tril(torch.ones(self.max_len*2, self.max_len*2))[None, None, ...])
        elif self.config.mask_type == 'markovian':
            mask = torch.diag(torch.ones(self.max_len*2))
            for i in range(self.max_len):
                mask[2*i+1, 2*i] = 1 #a_t is able to see s_t
            self.register_buffer('critic_mask', mask[None, None, ...])
        else:
            raise NotImplementedError

    def critic_forward(self, obs_seq, action_seq, pred_action_seq, pos_emb):
        batch_size, T, obs_dim = obs_seq.size()
        state_emb = self.embed_state(obs_seq) + pos_emb
        action_emb = self.embed_action(action_seq) + pos_emb
        pred_action_emb = self.embed_action(pred_action_seq) + pos_emb
        state_action_tokens = torch.stack([state_emb, action_emb], dim=1).permute(0, 2, 1, 3).reshape(batch_size, 2*T, self.config.n_embd)
        next_state_action_tokens = torch.stack([state_emb, pred_action_emb], dim=1).permute(0, 2, 1, 3).reshape(batch_size, 2*T, self.config.n_embd)
        q1, q2 = self.q(state_action_tokens, self.critic_mask)
        tq1, tq2 = self.target_q(next_state_action_tokens, self.critic_mask)
        return q1[:, 1::2], q2[:, 1::2], tq1[:, 1::2].detach(), tq2[:, 1::2].detach()

    def actor_forward(self, obs_seq, pred_action_seq, pos_emb):
        batch_size, T, obs_dim = obs_seq.size()
        state_emb = self.embed_state(obs_seq) + pos_emb
        pred_action_emb = self.embed_action(pred_action_seq) + pos_emb
        state_action_tokens = torch.stack([state_emb, pred_action_emb], dim=1).permute(0, 2, 1, 3).reshape(batch_size, 2*T, self.config.n_embd)
        q1, q2 = self.q(state_action_tokens, self.critic_mask)
        return q1[:, 1::2], q2[:, 1::2]

class ValueTransformer(nn.Module):
    def __init__(self, obs_shape, action_shape, config):
        super().__init__()
        self.config = config
        self.actor = Actor(obs_shape[0], action_shape[0], self.config)

        self.critic = Critic(obs_shape[0], action_shape[0], self.config)
        self.critic.target_q.load_state_dict(self.critic.q.state_dict())
        self._init_pe()

    def _init_pe(self):
        if self.config.pe_type == 'absolute':
            position = torch.arange(self.config.episode_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.config.n_embd, 2) * (-math.log(10000.0) / self.config.n_embd))
            pe = torch.zeros(self.config.episode_len, 1, self.config.n_embd)
            pe[:, 0, 0::2] = torch.sin(position * div_term) / 20.
            pe[:, 0, 1::2] = torch.cos(position * div_term) / 20.
            pe = torch.permute(pe, (1, 0, 2))  # (1, max_len, d)
            self.register_buffer('pos_emb', pe)
        elif self.config.pe_type == 'relative':
            position = torch.arange(self.config.traj_length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.config.n_embd, 2) * (-math.log(10000.0) / self.config.n_embd))
            pe = torch.zeros(self.config.traj_length, 1, self.config.n_embd)
            pe[:, 0, 0::2] = torch.sin(position * div_term) / 20.
            pe[:, 0, 1::2] = torch.cos(position * div_term) / 20.
            pe = torch.permute(pe, (1, 0, 2))  # (1, max_len, d)
            self.register_buffer('pos_emb', pe)
        else:
            # no pe, zero input
            pe = torch.zeros(1, self.config.traj_length, self.config.n_embd, requires_grad=False)
            self.register_buffer('pos_emb', pe)

    def _get_pe(self, timestep):
        if self.config.pe_type == 'absolute':
            pos_ind = repeat(timestep, "b t 1 -> b t h", h=self.config.n_embd)
            pos_emb = torch.gather(repeat(self.pos_emb, "1 t h -> b t h", b=pos_ind.shape[0]), 1, pos_ind)
        else:
            pos_emb = self.pos_emb
        return pos_emb

    def get_action(self, obs_seq, std, timestep):
        pos_emb = self._get_pe(timestep)
        assert timestep.shape[0] == obs_seq.shape[0], f"same shape expected"
        return self.actor.forward(obs_seq, std, pos_emb)

    def get_value(self, obs_seq, action_seq, pred_action_seq, timestep, critic_mode=True):
        pos_emb = self._get_pe(timestep)
        assert timestep.shape[0] == obs_seq.shape[0], f"same shape expected"
        if critic_mode is True:
            return self.critic.critic_forward(obs_seq, action_seq, pred_action_seq, pos_emb)
        else:
            return self.critic.actor_forward(obs_seq, pred_action_seq, pos_emb)

class ValueTransformerAgent:
    def __init__(self,
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
                 transformer_cfg,
                 has_next_action=False):
        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.config = transformer_cfg
        # models
        self.transformer = ValueTransformer(obs_shape, action_shape, self.config).to(device)
        # optimizers
        self.actor_opt = torch.optim.Adam(self.transformer.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.transformer.critic.parameters(), lr=lr)
        self.train()

    def train(self, training=True):
        self.training = training
        self.transformer.train(training)

    def act(self, obs, timestep, global_step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        timestep = torch.as_tensor(timestep, device=self.device).unsqueeze(0).unsqueeze(-1)
        stddev = utils.schedule(self.stddev_schedule, global_step)
        policy = self.transformer.get_action(obs, stddev, timestep)
        if eval_mode:
            action = policy.mean[:, -1]
        else:
            action = policy.sample(clip=None)[:, -1, :]
            if global_step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, timestep, step):
        metrics = dict()
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.transformer.get_action(obs, stddev, timestep)
        pred_action = dist.sample(clip=self.stddev_clip).detach()
        Q1, Q2, target_Q1, target_Q2 = self.transformer.get_value(obs, action, pred_action, timestep, critic_mode=True)
        with torch.no_grad():
            target_V = torch.min(target_Q1, target_Q2)[:, 1:]
            target_Q = reward[:, :-1] + (discount[:, :-1] * target_V)
        critic_loss = F.mse_loss(Q1[:, :-1], target_Q) + F.mse_loss(Q2[:, :-1], target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    def update_actor(self, obs, action, timestep, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.transformer.get_action(obs, stddev, timestep)
        Q1, Q2 = self.transformer.get_value(obs, action, policy.sample(clip=self.stddev_clip), timestep, critic_mode=False)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_ent'] = policy.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, timestep = utils.to_torch(
            batch, self.device)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, timestep, step))

        # update actor
        metrics.update(self.update_actor(obs, action, timestep, step))

        # update critic target
        utils.soft_update_params(self.transformer.critic.q, self.transformer.critic.target_q,
                                 self.critic_target_tau)

        return metrics
