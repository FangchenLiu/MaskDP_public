#!/usr/bin/env bash
# bc goal baselines
CUDA_VISIBLE_DEVICES=1 python pretrain.py \
    agent=bc_goal \
    agent.batch_size=1024 \
    agent.transformer_cfg.traj_length=64 \
    task=quadruped_run \
    snapshot_dir=snapshot \
    num_grad_steps=400010 \
    exp_name='baseline'\
    project=mdp-baseline \
    use_wandb=True &
sleep 30
CUDA_VISIBLE_DEVICES=2 python pretrain.py \
    agent=bc_goal \
    agent.batch_size=1024 \
    agent.transformer_cfg.traj_length=64 \
    task=walker_run \
    snapshot_dir=snapshot \
    num_grad_steps=400010 \
    exp_name='baseline'\
    project=mdp-baseline \
    use_wandb=True &
sleep 30
CUDA_VISIBLE_DEVICES=6 python pretrain.py \
    agent=bc_goal \
    agent.batch_size=1024 \
    agent.transformer_cfg.traj_length=64 \
    task=cheetah_run \
    snapshot_dir=snapshot \
    num_grad_steps=400010 \
    exp_name='baseline'\
    project=mdp-baseline \
    use_wandb=True &
sleep 30
# goal GPT baselines
################ assigned ####################
CUDA_VISIBLE_DEVICES=4 python pretrain.py \
    agent=seq_goal \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_layer=5 \
    task=quadruped_run \
    snapshot_dir=snapshot \
    exp_name='baseline'\
    project=mdp-baseline \
    use_wandb=True &
sleep 30
CUDA_VISIBLE_DEVICES=1 python pretrain.py \
    agent=seq_goal \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_layer=3 \
    task=quadruped_run \
    snapshot_dir=snapshot \
    exp_name='baseline'\
    project=mdp-baseline \
    use_wandb=True &
sleep 30
CUDA_VISIBLE_DEVICES=5 python pretrain.py \
    agent=seq_goal \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_layer=5 \
    task=walker_run \
    snapshot_dir=snapshot \
    num_grad_steps=400010 \
    exp_name='baseline'\
    project=mdp-baseline \
    use_wandb=True &
sleep 30
CUDA_VISIBLE_DEVICES=3 python pretrain.py \
    agent=seq_goal \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_layer=3 \
    task=walker_run \
    snapshot_dir=snapshot \
    num_grad_steps=400010 \
    exp_name='baseline'\
    project=mdp-baseline \
    use_wandb=True &
sleep 30
############### assigned ####################

CUDA_VISIBLE_DEVICES=6 python pretrain.py \
    agent=seq_goal \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_layer=5 \
    task=cheetah_run \
    snapshot_dir=snapshot \
    num_grad_steps=400010 \
    exp_name='baseline'\
    project=mdp-baseline \
    use_wandb=True &
sleep 30
CUDA_VISIBLE_DEVICES=5 python pretrain.py \
    agent=seq_goal \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_layer=3 \
    task=cheetah_run \
    snapshot_dir=snapshot \
    num_grad_steps=400010 \
    exp_name='baseline'\
    project=mdp-baseline \
    use_wandb=True &
sleep 30
CUDA_VISIBLE_DEVICES=6 python pretrain.py \
    agent=gpt \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_layer=5 \
    agent.transformer_cfg.norm='l2' \
    task=quadruped_run \
    snapshot_dir=snapshot \
    num_grad_steps=400010 \
    exp_name='baseline'\
    project=mdp-baseline \
    use_wandb=True &
sleep 30
CUDA_VISIBLE_DEVICES=7 python pretrain.py \
    agent=gpt \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_layer=5 \
    agent.transformer_cfg.norm='l2' \
    task=walker_run \
    snapshot_dir=snapshot \
    num_grad_steps=400010 \
    exp_name='baseline'\
    project=mdp-baseline \
    use_wandb=True &
sleep 30
CUDA_VISIBLE_DEVICES=7 python pretrain.py \
    agent=gpt \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_layer=5 \
    agent.transformer_cfg.norm='l2' \
    task=cheetah_run \
    snapshot_dir=snapshot \
    num_grad_steps=400010 \
    exp_name='baseline'\
    project=mdp-baseline \
    use_wandb=True &
sleep 30