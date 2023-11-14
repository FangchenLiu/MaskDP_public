#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python pretrain.py \
    agent=mdp \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.loss="masked" \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_enc_layer=3 \
    agent.transformer_cfg.n_dec_layer=2 \
    agent.transformer_cfg.norm='l2' \
    num_grad_steps=400010 \
    task=quadruped_run \
    snapshot_dir=snapshot \
    resume=false\
    exp_name='ablation'\
    project=mdp_ablation \
    use_wandb=True &
sleep 30

CUDA_VISIBLE_DEVICES=5 python pretrain.py \
    agent=mdp \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.loss="total" \
    agent.mask_ratio=[0.95]\
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_enc_layer=3 \
    agent.transformer_cfg.n_dec_layer=2 \
    agent.transformer_cfg.norm='l2' \
    num_grad_steps=400010 \
    task=quadruped_run \
    snapshot_dir=snapshot \
    resume=false\
    exp_name='ablation'\
    project=mdp_ablation \
    use_wandb=True &
sleep 30

CUDA_VISIBLE_DEVICES=6 python pretrain.py \
    agent=mdp \
    agent.batch_size=384 \
    agent.mask_ratio=[0.15]\
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.loss="total" \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_enc_layer=3 \
    agent.transformer_cfg.n_dec_layer=2 \
    agent.transformer_cfg.norm='l2' \
    num_grad_steps=400010 \
    task=quadruped_run \
    snapshot_dir=snapshot \
    resume=false\
    exp_name='ablation'\
    project=mdp_ablation \
    use_wandb=True &
sleep 30
CUDA_VISIBLE_DEVICES=7 python pretrain.py \
    agent=mdp \
    agent.batch_size=384 \
    agent.mask_ratio=[0.5]\
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.loss="total" \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_enc_layer=3 \
    agent.transformer_cfg.n_dec_layer=2 \
    agent.transformer_cfg.norm='l2' \
    num_grad_steps=400010 \
    task=quadruped_run \
    snapshot_dir=snapshot \
    resume=false\
    exp_name='ablation'\
    project=mdp_ablation \
    use_wandb=True &
sleep 30