#!/usr/bin/env bash
exp_mt_ff=("2022.05.12/051229_mdp" "2022.05.12/051130_mdp" "2022.05.12/051329_mdp")
data=(unsup)

for i in "${!data[@]}"
do
    CUDA_VISIBLE_DEVICES=2 python finetune_multitask.py \
        pretrained_data=mixed \
        finetuned_data=${data[$i]} \
        agent=mdp_multitask \
        agent.batch_size=256 \
        task=walker_run \
        seed=2\
        snapshot_base_dir=../../../output/${exp_mt_ff[1]}/snapshot \
        replay_buffer_dir=/shared/fangchen/dataset/exorl_mdp/${data[$i]} \
        snapshot_ts=300000 \
        replay_buffer_size=1000000\
        project=fangchen-eval-mdp-freeze \
        mt=true\
        use_wandb=True &
    sleep 30
    CUDA_VISIBLE_DEVICES=3 python finetune_multitask.py \
        pretrained_data=mixed \
        finetuned_data=${data[$i]} \
        agent=mdp_multitask \
        agent.batch_size=256 \
        task=walker_run \
        snapshot_base_dir=../../../output/${exp_mt_ff[1]}/snapshot \
        replay_buffer_dir=/shared/fangchen/dataset/exorl_mdp/${data[$i]} \
        snapshot_ts=300000 \
        replay_buffer_size=1000000\
        project=fangchen-eval-mdp-freeze \
        mt=true\
        use_wandb=True &
    sleep 30
    CUDA_VISIBLE_DEVICES=4 python finetune_multitask.py \
        pretrained_data=mixed \
        finetuned_data=${data[$i]} \
        agent=mdp_multitask \
        agent.batch_size=256 \
        task=walker_walk \
        snapshot_base_dir=../../../output/${exp_mt_ff[1]}/snapshot \
        replay_buffer_dir=/shared/fangchen/dataset/exorl_mdp/${data[$i]} \
        snapshot_ts=400000 \
        replay_buffer_size=1000000\
        project=fangchen-eval-mdp-freeze \
        mt=true\
        use_wandb=True &
    sleep 30
done