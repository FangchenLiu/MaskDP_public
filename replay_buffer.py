import datetime
import io
import random
import traceback
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from utils import get_norm

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn, domain, obs):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

def relable_episode(env, episode):
    rewards = []
    reward_spec = env.reward_spec()
    states = episode['physics']
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        reward = env.task.get_reward(env.physics)
        reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
        rewards.append(reward)
    episode['reward'] = np.array(rewards, dtype=reward_spec.dtype)
    return episode


class OfflineReplayBuffer(IterableDataset):
    def __init__(self, env, replay_dir, max_size, num_workers, discount, domain, traj_length, mode, cfg, relabel, obs):
        self._env = env
        self._replay_dir = replay_dir
        self._domain = domain
        self._mode = mode
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._discount = discount
        self._loaded = False
        self._traj_length = traj_length
        self._cfg = cfg
        self._relabel = relabel
        self._obs = obs
        # print('seed', np.random.get_state()[1][0])
        # random.seed(np.random.get_state()[1][0])

    def _load(self, relable=True):
        if relable:
            print('Labeling data...')
        else:
            print('loading reward free data...')
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.rglob('*.npz')) # get all episodes recursively
        for eps_fn in eps_fns:
            if self._size > self._max_size:
                print('over size', self._max_size)
                break
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            episode = load_episode(eps_fn, self._domain, self._obs)
            if relable:
                episode = self._relable_reward(episode)
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)

    def _sample_episode(self):
        if not self._loaded:
            self._load(self._relabel)
            self._loaded = True
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _relable_reward(self, episode):
        return relable_episode(self._env, episode)

    def _sample(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._traj_length + 1) + 1
        obs = episode['observation'][idx - 1:idx-1+self._traj_length]
        action = episode['action'][idx: idx+self._traj_length]
        next_obs = episode['observation'][idx: idx+self._traj_length]
        reward = episode['reward'][idx: idx+self._traj_length]
        discount = episode['discount'][idx: idx+self._traj_length] * self._discount
        timestep = np.arange(idx-1, idx+self._traj_length-1)[:, np.newaxis]
        return (obs, action, reward, discount, next_obs, 0)

    def _sample_goal(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        start_idx = np.random.randint(0, 900)
        length = np.random.randint(15, 20)
        start_obs = episode['observation'][start_idx]
        start_physics = episode['physics'][start_idx]
        goal_obs = episode['observation'][start_idx+length-1]
        goal_physics = episode['physics'][start_idx+length-1]
        timestep = length - 1
        #print(action.shape)
        return (start_obs, start_physics, goal_obs, goal_physics, timestep)

    def _sample_multiple_goal(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        start_idx = np.random.randint(0, 850)
        time_budget = np.array([12, 24, 36, 48, 60])

        start_obs = episode['observation'][start_idx]
        start_physics = episode['physics'][start_idx]

        goal = episode['observation'][start_idx + time_budget]
        goal_physics = episode['physics'][start_idx + time_budget]

        #print(action.shape)
        return (start_obs, start_physics, goal, goal_physics, time_budget)

    def _sample_context(self):
        episode = self._sample_episode()
        context_length = self._cfg.context_length
        forecast_length = self._cfg.forecast_length
        # add +1 for the first dummy transition
        #idx = np.random.randint(0, 50 - context_length+ 1) + 1
        start_idx = np.random.randint(100, 850)
        obs = episode['observation'][start_idx-1: start_idx+context_length] # last state is the initial obs
        action = episode['action'][start_idx: start_idx+context_length]
        reward = episode['reward'][start_idx+context_length: start_idx+context_length+forecast_length]
        physics = episode['physics'][start_idx-1: start_idx+context_length]
        remaining = episode['action'][start_idx+context_length: start_idx+context_length+forecast_length]
        return (obs, action, physics, reward, remaining)

    def __iter__(self):
        while True:
            if self._mode is None:
                yield self._sample()
            elif self._mode == 'goal':
                yield self._sample_goal()
            elif self._mode == 'multi_goal':
                yield self._sample_multiple_goal()
            elif self._mode == 'prompt':
                yield self._sample_context()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(env, replay_dir, max_size, batch_size, num_workers,
                       discount, domain, traj_length=1, mode=None, cfg=None, multi_task=False, relabel=True, obs='states'):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = OfflineReplayBuffer(env, replay_dir, max_size_per_worker,
                                       num_workers, discount, domain, traj_length, mode, cfg, relabel, obs)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader
