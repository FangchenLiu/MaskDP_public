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


def get_domain(task):
    if task.startswith('point_mass_maze'):
        return 'point_mass_maze'
    return task.split('_', 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1

def get_dir(cfg):
    snapshot_base_dir = Path(cfg.snapshot_base_dir)
    snapshot_dir = snapshot_base_dir / get_domain(cfg.task)
    snapshot = snapshot_dir / str(1) / f'snapshot_{cfg.snapshot_ts}.pt'
    return snapshot

def eval_seq_bc(global_step, agent, env, logger, goal_iter, device, num_eval_episodes, video_recorder):
    step, episode, total_dist2goal = 0, 0, []
    eval_until_episode = utils.Until(num_eval_episodes)
    batch = next(goal_iter)
    start_obs, start_physics, goal, goal_physics, time_budget = utils.to_torch(
        batch, device)

    while eval_until_episode(episode):
        time_step = env.reset()
        with env.physics.reset_context():
            env.physics.set_state(start_physics[episode].cpu())
        video_recorder.init(env, enabled=True)
        epi_budget = time_budget[episode]
        obs = start_obs[episode].unsqueeze(0)
        episode_dist = []
        for i in range(epi_budget.shape[0]):
            dist2goal = 1e6
            if obs.shape[0] > 1:
                obs = obs[-1].unsqueeze(0)
            current_goal = goal[episode, i]
            if i == 0:
                current_budget = epi_budget[i] + 2
            else:
                current_budget = epi_budget[i] - epi_budget[i-1] + 2
            for _ in range(current_budget):
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(obs,
                                       current_goal,
                                       global_step)
                time_step = env.step(action)
                obs_t = np.asarray(time_step.observation)
                obs_t = torch.as_tensor(obs_t, device=device)
                obs = torch.cat((obs, obs_t.unsqueeze(0)), dim=0)

                dist = np.linalg.norm(time_step.observation - current_goal.cpu().numpy())
                dist2goal = min(dist2goal, dist)
                video_recorder.record(env)
            episode_dist.append(dist2goal)

        total_dist2goal.append(episode_dist)
        video_recorder.save(f'{global_step}.mp4')
        video_recorder.render_goal(env, goal_physics[episode, -1])
        episode += 1

    total_dist2goal = np.array(total_dist2goal)
    with logger.log_and_dump_ctx(global_step, ty='eval') as log:
        log('dist2goal1', np.mean(total_dist2goal[:, 0]))
        log('dist2goal2', np.mean(total_dist2goal[:, 1]))
        log('dist2goal3', np.mean(total_dist2goal[:, 2]))
        log('dist2goal4', np.mean(total_dist2goal[:, 3]))
        log('dist2goal5', np.mean(total_dist2goal[:, 4]))
        log('step', global_step)

def eval_bc(global_step, agent, env, logger, goal_iter, device, num_eval_episodes, video_recorder):
    step, episode, total_dist2goal = 0, 0, []
    eval_until_episode = utils.Until(num_eval_episodes)
    batch = next(goal_iter)
    start_obs, start_physics, goal, goal_physics, time_budget = utils.to_torch(
        batch, device)

    while eval_until_episode(episode):
        time_step = env.reset()
        with env.physics.reset_context():
            env.physics.set_state(start_physics[episode].cpu())
        video_recorder.init(env, enabled=True)
        epi_budget = time_budget[episode]
        obs = start_obs[episode]
        episode_dist = []
        for i in range(epi_budget.shape[0]):
            dist2goal = 1e6
            current_goal = goal[episode, i]
            if i == 0:
                current_budget = epi_budget[i] + 2
            else:
                current_budget = epi_budget[i] - epi_budget[i-1] + 2
            for _ in range(current_budget):
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(obs,
                                       current_goal,
                                       global_step)
                time_step = env.step(action)
                obs = np.asarray(time_step.observation)
                obs = torch.as_tensor(obs, device=device)
                dist = np.linalg.norm(time_step.observation - current_goal.cpu().numpy())
                dist2goal = min(dist2goal, dist)
                video_recorder.record(env)
            episode_dist.append(dist2goal)

        total_dist2goal.append(episode_dist)
        video_recorder.save(f'{global_step}.mp4')
        video_recorder.render_goal(env, goal_physics[episode, -1])
        episode += 1

    total_dist2goal = np.array(total_dist2goal)
    with logger.log_and_dump_ctx(global_step, ty='eval') as log:
        log('dist2goal1', np.mean(total_dist2goal[:, 0]))
        log('dist2goal2', np.mean(total_dist2goal[:, 1]))
        log('dist2goal3', np.mean(total_dist2goal[:, 2]))
        log('dist2goal4', np.mean(total_dist2goal[:, 3]))
        log('dist2goal5', np.mean(total_dist2goal[:, 4]))
        log('step', global_step)

def eval_mdp(global_step, agent, env, logger, goal_iter, device, num_eval_episodes, video_recorder, replan=False):
    step, episode, total_dist2goal = 0, 0, []
    eval_until_episode = utils.Until(num_eval_episodes)
    batch = next(goal_iter)
    start_obs, start_physics, goal, goal_physics, time_budget = utils.to_torch(
        batch, device)

    while eval_until_episode(episode):
        time_step = env.reset()
        with env.physics.reset_context():
            env.physics.set_state(start_physics[episode].cpu())
        dist2goal = 1e6
        video_recorder.init(env, enabled=True)
        if replan is False:
            with torch.no_grad(), utils.eval_mode(agent):
                actions = agent.multi_goal_act(start_obs[episode].unsqueeze(0), goal[episode], time_budget[episode])

            states = []
            for a in actions:
                time_step = env.step(a)
                video_recorder.record(env)
                states.append(np.asarray(time_step.observation))
            states = np.array(states)
            episode_dist = []
            episode_budget = time_budget[episode]

            for i in range(len(episode_budget)):
                dist2goal = 1e5
                current_goal = goal[episode, i]
                for t in range(len(states)):
                    dist = np.linalg.norm(states[t] - current_goal.cpu().numpy())
                    dist2goal = min(dist2goal, dist)

                episode_dist.append(dist2goal)

            video_recorder.save(f'{global_step}.mp4')
            video_recorder.render_goal(env, goal_physics[episode, -1])
            episode += 1
            total_dist2goal.append(episode_dist)
        else:
            obs = start_obs[episode]
            episode_budget = time_budget[episode]
            total_episode_budget = episode_budget[-1]
            goal_index = 0
            states = []
            for i in range(total_episode_budget):
                if i == episode_budget[goal_index]:
                    goal_index += 1
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.multi_goal_act(obs.unsqueeze(0), goal[episode, goal_index:], time_budget[episode, goal_index:]-i)[0, ...]
                    time_step = env.step(action)
                    video_recorder.record(env)
                    obs = np.asarray(time_step.observation)
                    obs = torch.as_tensor(obs, device=device)
                    states.append(np.asarray(time_step.observation))

            states = np.array(states)
            episode_dist = []
            episode_budget = time_budget[episode]

            for i in range(len(episode_budget)):
                dist2goal = 1e5
                current_goal = goal[episode, i]
                for t in range(len(states)):
                    dist = np.linalg.norm(states[t] - current_goal.cpu().numpy())
                    dist2goal = min(dist2goal, dist)

                episode_dist.append(dist2goal)

            video_recorder.save(f'{global_step}.mp4')
            video_recorder.render_goal(env, goal_physics[episode, -1])
            episode += 1
            total_dist2goal.append(episode_dist)


    total_dist2goal = np.array(total_dist2goal)
    with logger.log_and_dump_ctx(global_step, ty='eval') as log:
        log('dist2goal1', np.mean(total_dist2goal[:, 0]))
        log('dist2goal2', np.mean(total_dist2goal[:, 1]))
        log('dist2goal3', np.mean(total_dist2goal[:, 2]))
        log('dist2goal4', np.mean(total_dist2goal[:, 3]))
        log('dist2goal5', np.mean(total_dist2goal[:, 4]))
        log('step', global_step)


@hydra.main(config_path='.', config_name='eval')
def main(cfg):
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create envs
    env = dmc.make(cfg.task, seed=cfg.seed)

    # create agent
    path = get_dir(cfg)
    agent = hydra.utils.instantiate(cfg.agent,
                                    obs_shape=env.observation_spec().shape,
                                    action_shape=env.action_spec().shape,
                                    path=path)

    # create logger
    cfg.agent.obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = env.action_spec().shape
    cfg.agent.transformer_cfg = agent.config
    exp_name = '_'.join([cfg.agent.name,cfg.task, str(cfg.replan),str(cfg.seed)])
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
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb, mode='multi_goal')

    # create replay buffer
    data_specs = (env.observation_spec(), env.action_spec(), env.reward_spec(),
                  env.discount_spec())

    # create data storage
    domain = get_domain(cfg.task)

    goal_dir = Path(cfg.goal_buffer_dir) / domain / cfg.task

    print(f'goal dir: {goal_dir}')

    goal_loader = make_replay_loader(env, goal_dir, cfg.goal_buffer_size,
                                     cfg.num_eval_episodes,
                                     cfg.goal_buffer_num_workers,
                                     cfg.discount,
                                     domain=domain,
                                     traj_length=1,
                                     mode='multi_goal',
                                     cfg=agent.config,
                                     relabel=False)
    goal_iter = iter(goal_loader)

    # create video recorders
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    timer = utils.Timer()

    global_step = 0
    eval_every_step = utils.Every(cfg.eval_every_steps)

    if eval_every_step(global_step):
        logger.log('eval_total_time', timer.total_time(), global_step)
        if cfg.agent.name == 'mdp_goal':
            eval_mdp(global_step, agent, env, logger, goal_iter, device, cfg.num_eval_episodes,
                     video_recorder, replan=cfg.replan)
        elif cfg.agent.name == 'bc_goal':
            eval_bc(global_step, agent, env, logger, goal_iter, device, cfg.num_eval_episodes,
                    video_recorder)
        elif cfg.agent.name == 'seq_goal':
            eval_seq_bc(global_step, agent, env, logger, goal_iter, device, cfg.num_eval_episodes,
                        video_recorder)
        else:
            raise NotImplementedError



if __name__ == '__main__':
    main()
