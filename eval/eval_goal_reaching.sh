#!/usr/bin/env bash
exp_mdp=("path/to/your/snapshot1", "path/to/your/snapshot2", "path/to/your/snapshot3","path/to/your/snapshot4","path/to/your/snapshot5","path/to/your/snapshot6","path/to/your/snapshot7")
env=('quadruped_run' 'quadruped_walk' 'walker_run' 'walker_walk' 'walker_stand' 'cheetah_run' 'cheetah_run_backward')

for i in "${!env[@]}"
do
   CUDA_VISIBLE_DEVICES=6 python eval_multigoal.py \
        pretrained_data=expert \
        agent=mdp_goal \
        agent.batch_size=384 \
        seed=2\
        num_eval_episodes=100 \
        task=${env[$i]} \
        snapshot_base_dir=../../../your_basedir_name/${exp_mdp[$i]}/snapshot \
        replay_buffer_dir=/path/to/your/replay/buffer \
        goal_buffer_dir=/path/to/your/replay/buffer \
        snapshot_ts=300000 \
        project=eval-multi-goal \
        mt=true \
        replan=false \
        use_wandb=True &
        sleep 30

   CUDA_VISIBLE_DEVICES=7 python eval_multigoal.py \
        pretrained_data=expert \
        agent=mdp_goal \
        agent.batch_size=384 \
        seed=2\
        num_eval_episodes=100 \
        task=${env[$i]} \
        snapshot_base_dir=../../../your_basedir_name/${exp_mdp[$i]}/snapshot \
        replay_buffer_dir=/path/to/your/replay/buffer \
        goal_buffer_dir=/path/to/your/replay/buffer \
        snapshot_ts=300000 \
        project=eval-multi-goal \
        mt=true \
        replan=true \
        use_wandb=True &
        sleep 30
done