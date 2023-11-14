#!/usr/bin/env bash
exp_gpt=("2022.05.13/062302_gpt" "2022.05.13/062302_gpt"  "2022.05.13/062332_gpt" "2022.05.13/062332_gpt"  "2022.05.13/062332_gpt" \
                "2022.05.13/062402_gpt" "2022.05.13/062402_gpt")

env=('quadruped_run' 'quadruped_walk' 'walker_run' 'walker_walk' 'walker_stand' 'cheetah_run' 'cheetah_run_backward')

for i in "${!exp_gpt[@]}"
do
    CUDA_VISIBLE_DEVICES=$((i+1)) python eval_prompt.py \
            agent=gpt_prompt \
            agent.batch_size=128 \
            agent.context_length=5\
            agent.forecast_length=40\
            task=${env[$i]} \
            snapshot_base_dir=../../../output/${exp_gpt[$i]}/snapshot \
            goal_buffer_dir=/path/to/maskdp_eval/expert \
            snapshot_ts=400000 \
            num_eval_episodes=100 \
            project=fangchen-eval-prompt \
            replan=true \
            use_wandb=True &
    sleep 30
    CUDA_VISIBLE_DEVICES=$((i+1)) python eval_prompt.py \
            agent=gpt_prompt \
            agent.batch_size=128 \
            agent.context_length=5\
            agent.forecast_length=60\
            task=${env[$i]} \
            snapshot_base_dir=../../../output/${exp_gpt[$i]}/snapshot \
            goal_buffer_dir=/path/to/maskdp_eval/expert \
            snapshot_ts=400000 \
            num_eval_episodes=100 \
            project=fangchen-eval-prompt \
            replan=true \
            use_wandb=True &
    sleep 30
done