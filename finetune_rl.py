import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
from video import VideoRecorder
import wandb
import omegaconf

torch.backends.cudnn.benchmark = True


def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1


def get_dir(cfg):
    snapshot_base_dir = Path(cfg.snapshot_base_dir)
    snapshot_dir = snapshot_base_dir / get_domain(cfg.task)
    snapshot = snapshot_dir / str(1) / f"snapshot_{cfg.snapshot_ts}.pt"
    return snapshot


def eval(global_step, agent, env, logger, num_eval_episodes, video_recorder, max_len):
    step, episode, episode_rewards = 0, 0, []
    eval_until_episode = utils.Until(num_eval_episodes)
    while eval_until_episode(episode):
        total_rewards = 0
        time_step = env.reset()
        video_recorder.init(env, enabled=(episode == 0))
        observations = []
        while not time_step.last():
            observations.append(time_step.observation)
            obs = np.asarray(observations)
            if len(obs) > max_len:
                obs = obs[-max_len:]
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(obs, global_step, eval_mode=True)
            time_step = env.step(action)
            video_recorder.record(env)
            total_rewards += time_step.reward
            step += 1
        episode_rewards.append(total_rewards)
        episode += 1
        video_recorder.save(f"{global_step}.mp4")

    episode_rewards = np.array(episode_rewards)
    with logger.log_and_dump_ctx(global_step, ty="eval") as log:
        log("reward", np.mean(episode_rewards))
        log("std", np.std(episode_rewards))
        # if len(total_rewards) > 1:
        #     log('reward2', total_rewards[1])
        # if len(total_rewards) > 2:
        #     log('reward3', total_rewards[2])
        log("step", global_step)


@hydra.main(config_path=".", config_name="finetune_rl")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create envs
    env = dmc.make(cfg.task, seed=cfg.seed)
    # create logger
    cfg.agent.obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = env.action_spec().shape
    # create agent
    path = get_dir(cfg)
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape,
        path=path,
    )

    cfg.agent.transformer_cfg = agent.config

    exp_name = "_".join(
        [cfg.agent.name, get_domain(cfg.task), cfg.finetuned_data, str(cfg.seed)]
    )
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=cfg.project,
        entity="maskdp",
        name=exp_name,
        config=wandb_config,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
        ),
        mode="online" if cfg.use_wandb else "offline",
        notes=cfg.notes,
    )

    logger = Logger(
        work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb, mode="offline-rl"
    )

    # create replay buffer
    data_specs = (
        env.observation_spec(),
        env.action_spec(),
        env.reward_spec(),
        env.discount_spec(),
    )

    # create data storage
    domain = get_domain(cfg.task)
    datasets_dir = work_dir / cfg.replay_buffer_dir
    if str(cfg.finetuned_data) == "unsup":
        print("using unsupervised data")
        replay_dir = datasets_dir.resolve() / domain
    else:
        print("using supervised data")
        replay_dir = datasets_dir.resolve() / domain / cfg.task
    print(replay_dir)
    replay_loader = make_replay_loader(
        env,
        replay_dir,
        cfg.replay_buffer_size,
        cfg.batch_size,
        cfg.replay_buffer_num_workers,
        cfg.discount,
        cfg.agent.attn_length,
        relabel=True,
    )
    replay_iter = iter(replay_loader)

    # create video recorders
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    timer = utils.Timer()

    global_step = 0

    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    log_every_step = utils.Every(cfg.log_every_steps)
    max_len = cfg.agent.attn_length

    while train_until_step(global_step):
        # try to evaluate
        if eval_every_step(global_step):
            logger.log("eval_total_time", timer.total_time(), global_step)
            eval(
                global_step,
                agent,
                env,
                logger,
                cfg.num_eval_episodes,
                video_recorder,
                max_len,
            )

        metrics = agent.update(replay_iter, global_step)
        logger.log_metrics(metrics, global_step, ty="train")
        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty="train") as log:
                log("fps", cfg.log_every_steps / elapsed_time)
                log("total_time", total_time)
                log("step", global_step)

        global_step += 1


if __name__ == "__main__":
    main()
