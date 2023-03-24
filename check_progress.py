import argparse
from pathlib import Path
from omegaconf import OmegaConf
from collections import defaultdict
import shutil
import re

from dmc_benchmark import DOMAINS

TARGET_DIR = Path(
    '/private/home/denisy/workspace/research/unsupervised_data_collection/datasets'
)

NON_RUN_DIRS = ['code', 'slurm', 'multirun']


def parse_overrides(run_dir):
    overrides_path = run_dir / '.hydra' / 'overrides.yaml'
    overrides = OmegaConf.load(str(overrides_path))
    kvs = dict()
    for override in overrides:
        key, value = override.split('=', 1)
        kvs[key] = value
    return kvs


def get_progress(src_dir, file_re):
    progress_step = 0
    for file_path in src_dir.iterdir():
        match = re.match(file_re, file_path.stem)
        if match:
            step = int(match.groups()[0])
            progress_step = max(progress_step, step)
    return progress_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_dir', type=str)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    stats = defaultdict(defaultdict)

    for run_dir in sweep_dir.iterdir():
        if run_dir.stem in NON_RUN_DIRS:
            continue
        #import ipdb; ipdb.set_trace()
        overrides = parse_overrides(run_dir)

        progress = get_progress(run_dir, r'actor_weights_(.+)')

        stats[overrides['agent']][overrides['seed']] = progress

    all_done = True
    for agent in stats.keys():
        seeds = stats[agent]
        done_seeds, running_seeds = [], []
        for seed, progress in seeds.items():
            if progress == 1000000:
                done_seeds.append(seed)
            else:
                running_seeds.append(f'{seed}:{progress}')
        print(agent)
        done_seeds = ','.join(done_seeds)
        running_seeds = ','.join(running_seeds)
        print(f'done seeds: {done_seeds}')
        print(f'running seeds: {running_seeds}')
        if len(running_seeds) == 0:
            print('OK!')
        else:
            all_done = False
        print()

    if all_done:
        print('Everything is ready!')
    else:
        print('Some unfinished runs!')


if __name__ == '__main__':
    main()
