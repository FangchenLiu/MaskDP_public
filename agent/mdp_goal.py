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

class MDPGoalAgent:
    def __init__(self,
                 name,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 batch_size,
                 use_tb,
                 finetune,
                 transformer_cfg,
                 path=None):

        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.use_tb = use_tb

        # init from snapshot
        payload = None
        if path is not None:
            print('loading existing model...')
            payload = torch.load(path)
            self.config = payload['cfg']
        else:
            self.config = transformer_cfg
        self.mdp = MaskedDP(obs_shape[0], action_shape[0], self.config).to(device)
        print("number of parameters: %e", sum(p.numel() for p in self.mdp.parameters()))
        if path is not None:
            self.mdp.load_state_dict(payload['model'])

        self.finetune = finetune
        self._freeze_layers()
        self.train()

    def _freeze_layers(self):
        frozen_layers = [self.mdp.pos_embed, self.mdp.decoder_pos_embed, self.mdp.state_embed, self.mdp.action_embed,
                         self.mdp.mask_token]

        if self.finetune == 'decoder':
            frozen_layers += [self.mdp.encoder_blocks, self.mdp.encoder_norm]

        if self.finetune == 'linear':
            frozen_layers += [self.mdp.decoder_blocks, self.mdp.decoder_state_embed, self.mdp.decoder_action_embed]

        for m in frozen_layers:
            if isinstance(m, nn.Module):
                for param in m.parameters():
                    param.requires_grad = False
            elif isinstance(m, nn.Parameter):
                m.requires_grad = False
        # optimizers
        print("number of parameters to be tuned: %e", sum(p.numel() for p in filter(lambda p: p.requires_grad, self.mdp.parameters())))
        self.opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.mdp.parameters()), lr=self.lr)

    def train(self, training=True):
        self.training = training
        self.mdp.train(training)

    def multi_goal_act(self, obs, goal, time_budgets):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).unsqueeze(0)

        assert goal.shape[1] == len(time_budgets)
        T = time_budgets[-1]
        if 2*(T+1) > self.mdp.pos_embed.shape[1]:
            pos_embed = utils.interpolate_pos_embed(self.mdp.pos_embed, 2*(T+1))
            decoder_pos_embed = utils.interpolate_pos_embed(self.mdp.decoder_pos_embed, 2*(T+1))
            attn_mask = torch.ones(2*(T+1), 2*(T+1))[None, None, ...].to(self.device)
        else:
            pos_embed = self.mdp.pos_embed
            decoder_pos_embed = self.mdp.decoder_pos_embed
            attn_mask = self.mdp.attn_mask

        s_emb = self.mdp.state_embed(obs) + pos_embed[:, 0]
        g_emb = self.mdp.state_embed(goal) + pos_embed[:, time_budgets*2]
        # encoder
        x = torch.cat([s_emb, g_emb], dim=1)
        for blk in self.mdp.encoder_blocks:
            x = blk(x, attn_mask)
        x = self.mdp.encoder_norm(x)

        if T > 1:
            obs = self.mdp.mask_token.repeat(obs.shape[0], T+1, 1)
            obs[:, 0] = x[:, 0]
            obs[:, time_budgets] = x[:, 1:]
        else:
            obs = x

        mask_actions = self.mdp.mask_token.repeat(obs.shape[0], T+1, 1)
        obs = self.mdp.decoder_state_embed(obs)
        mask_actions = self.mdp.decoder_action_embed(mask_actions)

        x = torch.stack([obs, mask_actions], dim=1).permute(0, 2, 1, 3).reshape(obs.shape[0], 2*(T+1), self.config.n_embd)
        x += decoder_pos_embed[:, :2*(T+1)]
        # apply Transformer blocks
        for blk in self.mdp.decoder_blocks:
            x = blk(x, attn_mask)
        actions = self.mdp.action_head(x[:, 1::2])[:, :-1]
        return actions.cpu().numpy()[0]

    def act(self, obs, goal, T):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).unsqueeze(0)
        if 2*(T+1) > self.mdp.pos_embed.shape[1]:
            pos_embed = utils.interpolate_pos_embed(self.mdp.pos_embed, 2*(T+1))
            decoder_pos_embed = utils.interpolate_pos_embed(self.mdp.decoder_pos_embed, 2*(T+1))
            attn_mask = torch.ones(2*(T+1), 2*(T+1))[None, None, ...].to(self.device)
        else:
            pos_embed = self.mdp.pos_embed
            decoder_pos_embed = self.mdp.decoder_pos_embed
            attn_mask = self.mdp.attn_mask

        s_emb = self.mdp.state_embed(obs) + pos_embed[:, 0]
        g_emb = self.mdp.state_embed(goal) + pos_embed[:, 2*T]
        # encoder
        x = torch.cat([s_emb, g_emb], dim=1)
        for blk in self.mdp.encoder_blocks:
            x = blk(x, attn_mask)
        x = self.mdp.encoder_norm(x)

        if T > 1:
            mask_states = self.mdp.mask_token.repeat(obs.shape[0], T-1, 1)
            obs = torch.cat([x[:, 0].unsqueeze(1), mask_states], dim=1)
            obs = torch.cat([obs, x[:, -1].unsqueeze(1)], dim=1)
        else:
            obs = x

        mask_actions = self.mdp.mask_token.repeat(obs.shape[0], T+1, 1)
        obs = self.mdp.decoder_state_embed(obs)
        mask_actions = self.mdp.decoder_action_embed(mask_actions)

        x = torch.stack([obs, mask_actions], dim=1).permute(0, 2, 1, 3).reshape(obs.shape[0], 2*(T+1), self.config.n_embd)
        x += decoder_pos_embed[:, :2*(T+1)]
        # apply Transformer blocks
        for blk in self.mdp.decoder_blocks:
            x = blk(x, attn_mask)
        actions = self.mdp.action_head(x[:, 1::2])[:, :-1]
        return actions.cpu().numpy()[0]
