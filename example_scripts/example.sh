#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python pretrain_mdp_mt.py \
    agent=mdp \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.loss="total" \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_enc_layer=3 \
    agent.transformer_cfg.n_dec_layer=2 \
    agent.transformer_cfg.shared_proj=false \
    agent.transformer_cfg.type_embed=false \
    num_grad_steps=400010 \
    task=quadruped_run \
    snapshot_dir=snapshot \
    resume=false\
    exp_name='neurips'\
    project=mdp-neurips \
    use_wandb=True &

# eval single/multi goal
CUDA_VISIBLE_DEVICES=1 python eval_singlegoal.py \
        agent=mdp \
        agent.batch_size=384 \
        task=walker_walk \
        snapshot_base_dir=/path/to/pretrained_model \
        goal_buffer_dir=/path/to/goal \
        snapshot_ts=400000 \
        project=goal-eval \
        replan=False \
        use_wandb=True &
# eval prompt
CUDA_VISIBLE_DEVICES=2 python eval_prompt.py \
            agent=mdp_prompt \
            agent.batch_size=128 \
            agent.context_length=5\
            agent.forecast_length=60\
            task=walker_walk \
            snapshot_base_dir=/path/to/pretrained_model \
            goal_buffer_dir=/path/to/goal \
            snapshot_ts=400000 \
            num_eval_episodes=100 \
            project=eval-prompt \
            mt=true \
            replan=true \
            use_wandb=True &

# finetune RL
CUDA_VISIBLE_DEVICES=3 python finetune_rl.py \
        pretrained_data=expert \
        finetuned_data=${data_dir} \
        agent=mdp_multitask \
        agent.batch_size=256 \
        task=walker_walk \
        snapshot_base_dir=/path/to/pretrained_model \
        replay_buffer_dir=/path/to/replay_buffer \
        snapshot_ts=400000 \
        replay_buffer_size=1000000\
        project=eval-rl \
        mt=true\
        use_wandb=True &