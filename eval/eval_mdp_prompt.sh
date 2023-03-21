#!/usr/bin/env bash
exp_mt_ff=("2022.05.12/051229_mdp" "2022.05.12/051229_mdp" "2022.05.12/051130_mdp" "2022.05.12/051130_mdp" "2022.05.12/051130_mdp" \
     "2022.05.12/051329_mdp" "2022.05.12/051329_mdp")

env=('quadruped_run' 'quadruped_walk' 'walker_run' 'walker_walk' 'walker_stand' 'cheetah_run' 'cheetah_run_backward')


for i in "${!exp_mt_ff[@]}"
do
    CUDA_VISIBLE_DEVICES=$((i+1)) python eval_prompt.py \
            pretrained_data=expert \
            agent=mdp_prompt \
            agent.batch_size=128 \
            agent.context_length=5\
            agent.forecast_length=60\
            task=${env[$i]} \
            snapshot_base_dir=../../../output/${exp_mt_ff[$i]}/snapshot \
            replay_buffer_dir=/shared/fangchen/rl_data_collection/expert \
            goal_buffer_dir=/shared/fangchen/rl_data_collection/expert \
            snapshot_ts=400000 \
            num_eval_episodes=100 \
            project=fangchen-eval-prompt \
            mt=true \
            replan=true \
            use_wandb=True &
    sleep 30
    CUDA_VISIBLE_DEVICES=$((i+1)) python eval_prompt.py \
            pretrained_data=expert \
            agent=mdp_prompt \
            agent.batch_size=128 \
            agent.context_length=5\
            agent.forecast_length=40\
            task=${env[$i]} \
            snapshot_base_dir=../../../output/${exp_mt_ff[$i]}/snapshot \
            replay_buffer_dir=/shared/fangchen/rl_data_collection/expert \
            goal_buffer_dir=/shared/fangchen/rl_data_collection/expert \
            snapshot_ts=400000 \
            num_eval_episodes=100 \
            project=fangchen-eval-prompt \
            mt=true \
            replan=false \
            use_wandb=True &
    sleep 30
done