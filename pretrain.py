import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

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


def get_dir(cfg):
    resume_dir = Path(cfg.resume_dir)
    snapshot = resume_dir / str(
        cfg.seed) / f'snapshot_{cfg.resume_step}.pt'
    print('loading from', snapshot)
    return snapshot

def get_domain(task):
    if task.startswith('point_mass_maze'):
        return 'point_mass_maze'
    return task.split('_', 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1



@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create envs
    env = dmc.make(cfg.task, seed=cfg.seed)

    # create agent
    agent = hydra.utils.instantiate(cfg.agent,
                                    obs_shape=env.observation_spec().shape,
                                    action_shape=env.action_spec().shape)

    if cfg.resume is True:
        resume_dir = get_dir(cfg)
        payload = torch.load(resume_dir)
        agent.model.load_state_dict(payload['model'])

    domain = get_domain(cfg.task)
    snapshot_dir = work_dir / Path(cfg.snapshot_dir) / domain / str(cfg.seed)
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    # create logger
    cfg.agent.obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = env.action_spec().shape
    exp_name = '_'.join([cfg.agent.name,domain,str(cfg.seed)])
    wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.project,
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
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    replay_train_dir = Path(cfg.replay_buffer_dir) / domain
    print(f'replay dir: {replay_train_dir}')
    train_loader = make_replay_loader(env, replay_train_dir, cfg.replay_buffer_size,
                                      cfg.batch_size,
                                      cfg.replay_buffer_num_workers,
                                      cfg.discount,
                                      domain,
                                      cfg.agent.transformer_cfg.traj_length,
                                      relabel=False)
    train_iter = iter(train_loader)
    # create video recorders

    timer = utils.Timer()

    global_step = cfg.resume_step

    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    log_every_step = utils.Every(cfg.log_every_steps)

    while train_until_step(global_step):
        # try to evaluate
        metrics = agent.update(train_iter, global_step)
        logger.log_metrics(metrics, global_step, ty='train')
        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty='train') as log:
                log('fps', cfg.log_every_steps / elapsed_time)
                log('total_time', total_time)
                log('step', global_step)

        if global_step in cfg.snapshots:
            snapshot = snapshot_dir / f'snapshot_{global_step}.pt'
            payload = {'model': agent.model.state_dict(), 'cfg': cfg.agent.transformer_cfg}
            with snapshot.open('wb') as f:
                torch.save(payload, f)

        global_step += 1



if __name__ == '__main__':
    main()
