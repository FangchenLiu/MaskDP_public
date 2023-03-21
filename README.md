

# Masked Autoencoding for Scalable and Generalizable Decision Making

This codebase is a pre-released implementation of [MaskDP](https://openreview.net/forum?id=lNokkSaUbfV).

## Prerequisites

Install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Download MuJoCo binaries [here](https://mujoco.org/download).
* Unzip the downloaded archive into `~/.mujoco/`.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 unzip
```

Install dependencies:
```sh
conda env create -f conda_env.yml
conda activate maskdp
```

## Dataset
This branch consists codes about data collection
```sh
# collecting per-domain unsupervised data (only intrinsic rewards), walker_stand is just used to specify a domain
CUDA_VISIBLE_DEVICES=7 python pretrain.py \
    seed=42 \
    bonus=0.0 \
    task=walker_stand \
    supervised=false \
    agent=proto \
    agent.nstep=1 \
    agent.batch_size=1024 \
    obs_type=states \
    action_repeat=1 \
    num_train_frames=2000010 \
    replay_buffer_size=100000 \
    replay_buffer_num_workers=4 \
    save_replay_buffer=True \
    project=MaskDP_state_data \
    use_wandb=True &
```
```sh
# collecting semi-supervised data (mix intrinsic and extrinsic rewards) for each task
CUDA_VISIBLE_DEVICES=0 python pretrain.py \
    seed=42 \
    bonus=0.5 \
    task=walker_stand \
    supervised=true \
    agent=proto \
    agent.nstep=1 \
    agent.batch_size=1024 \
    obs_type=states \
    action_repeat=1 \
    num_train_frames=2000010 \
    replay_buffer_size=100000 \
    replay_buffer_num_workers=4 \
    save_replay_buffer=True \
    project=MaskDP_state_data \
    use_wandb=True &
```
```sh
# collecting supervised data (only extrinsic rewards) for each task
CUDA_VISIBLE_DEVICES=0 python pretrain.py \
    seed=42 \
    bonus=0.5 \
    task=walker_stand \
    supervised=true \
    agent=ddpg \
    agent.nstep=1 \
    agent.batch_size=1024 \
    obs_type=states \
    action_repeat=1 \
    num_train_frames=2000010 \
    replay_buffer_size=100000 \
    replay_buffer_num_workers=4 \
    save_replay_buffer=True \
    project=MaskDP_state_data \
    use_wandb=True &
```