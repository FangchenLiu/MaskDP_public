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

class MaskedDP(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        # MAE encoder specifics
        self.n_embd = config.n_embd
        self.max_len = config.traj_length * 2
        #self.mask_ratio = config.mask_ratio
        self.pe = config.pe
        self.norm = config.norm
        print('norm', self.norm)
        self.state_embed = nn.Linear(obs_dim, self.n_embd)
        self.action_embed = nn.Linear(action_dim, self.n_embd)
        self.encoder_blocks = nn.ModuleList([Block(config) for _ in range(config.n_enc_layer)])
        self.encoder_norm = nn.LayerNorm(self.n_embd)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_state_embed = nn.Linear(self.n_embd, self.n_embd)
        self.decoder_action_embed = nn.Linear(self.n_embd, self.n_embd)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.n_embd))

        self.decoder_blocks = nn.ModuleList([Block(config) for _ in range(config.n_dec_layer)])

        self.action_head = nn.Sequential(nn.LayerNorm(self.n_embd), nn.ReLU(inplace=True), nn.Linear(self.n_embd, action_dim), nn.Tanh()) # decoder to patch
        self.state_head = nn.Sequential(nn.LayerNorm(self.n_embd), nn.ReLU(inplace=True), nn.Linear(self.n_embd, obs_dim))
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = utils.get_1d_sincos_pos_embed_from_grid(self.n_embd, self.max_len)
        pe = torch.from_numpy(pos_embed).float().unsqueeze(0) / 2.
        self.register_buffer('pos_embed', pe)
        self.register_buffer('decoder_pos_embed', pe)
        self.register_buffer('attn_mask', torch.ones(self.max_len, self.max_len)[None, None, ...])
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, states, actions, mask_ratio):
        batch_size, T, obs_dim = states.size()
        s_emb = self.state_embed(states)
        a_emb = self.action_embed(actions)

        x = torch.stack([s_emb, a_emb], dim=1).permute(0, 2, 1, 3).reshape(batch_size, 2*T, self.n_embd)
        x = x + self.pos_embed
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x, self.attn_mask)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        s = self.decoder_state_embed(x[:, ::2])
        a = self.decoder_action_embed(x[:, 1::2])

        x = torch.stack([s, a], dim=1).permute(0, 2, 1, 3).reshape_as(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, self.attn_mask)

        # predictor projection
        s = self.state_head(x[:, ::2])
        a = self.action_head(x[:, 1::2])

        return s, a

    def forward_loss(self, target_s, target_a, pred_s, pred_a, mask):
        batch_size, T, _ = target_s.size()
        # apply normalization
        if self.norm == 'l2':
            target_s = target_s / torch.norm(target_s, dim=-1, keepdim=True)
        elif self.norm == 'mae':
            mean = target_s.mean(dim=-1, keepdim=True)
            var = target_s.var(dim=-1, keepdim=True)
            target_s = (target_s - mean) / (var + 1.e-6)**.5

        loss_s = (pred_s - target_s) ** 2
        loss_a = (pred_a - target_a) ** 2
        loss = torch.stack([loss_s.mean(dim=-1), loss_a.mean(dim=-1)], dim=1).permute(0, 2, 1).reshape(batch_size, 2*T)
        masked_loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss_s = loss_s.mean()
        loss_a = loss_a.mean()
        return masked_loss, loss_s, loss_a


class MaskedDPAgent:
    def __init__(self,
                 name,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 batch_size,
                 use_tb,
                 mask_ratio,
                 transformer_cfg):

        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.use_tb = use_tb
        self.config = transformer_cfg

        # models
        self.model = MaskedDP(obs_shape[0], action_shape[0], transformer_cfg).to(device)
        self.mask_ratio = mask_ratio
        # optimizers
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        print("number of parameters: %e", sum(p.numel() for p in self.model.parameters()))

        self.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def update_mdp(self, states, actions):
        metrics = dict()
        mask_ratio = np.random.choice(self.mask_ratio)
        latent, mask, ids_restore = self.model.forward_encoder(states, actions, mask_ratio)
        pred_s, pred_a = self.model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        mask_loss, state_loss, action_loss = self.model.forward_loss(states, actions, pred_s, pred_a, mask)
        if self.config.loss == 'masked':
            loss = mask_loss
        elif self.config.loss == 'total':
            loss = state_loss + action_loss
        else:
            raise NotImplementedError

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        if self.use_tb:
            metrics['mask_loss'] = mask_loss.item()
            metrics['state_loss'] = state_loss.item()
            metrics['action_loss'] = action_loss.item()

        return metrics

    def eval_validation(self, val_iter, step=None):
        metrics = dict()
        batch = next(val_iter)
        obs, action, _, _, _, _ = utils.to_torch(
            batch, self.device)
        mask_ratio = np.random.choice(self.mask_ratio)
        latent, mask, ids_restore = self.model.forward_encoder(obs, action, mask_ratio)
        pred_s, pred_a = self.model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        mask_loss, state_loss, action_loss = self.model.forward_loss(obs, action, pred_s, pred_a, mask)

        if self.use_tb:
            metrics['val_mask_loss'] = mask_loss.item()
            metrics['val_state_loss'] = state_loss.item()
            metrics['val_action_loss'] = action_loss.item()

        return metrics

    def update(self, replay_iter, step=None):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, _, _, _, _ = utils.to_torch(
            batch, self.device)

        # update critic
        metrics.update(self.update_mdp(obs, action))

        return metrics
