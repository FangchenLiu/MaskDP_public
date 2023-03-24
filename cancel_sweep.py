import os
import argparse
from subprocess import Popen, DEVNULL
from pathlib import Path


def cancel_job(job_id):
    print(f'canceling {job_id}')
    cmds = ['scancel', job_id]
    env = os.environ.copy()
    Popen(cmds, env=env).communicate()


def cancel_jobs(slurm_dir):
    num_jobs = 0
    for job in slurm_dir.glob('*.pkl'):
        if job.stem.endswith('_submitted'):
            job_id = job.stem[:-len('_submitted')]
            cancel_job(job_id)
            num_jobs += 1
    print(f'canceled {num_jobs} jobs')


def remove_snapshots(sweep_dir):
    for snapshot in sweep_dir.glob('**/snapshot.pt'):
        snapshot.unlink(missing_ok=True)
        print(f'removing {snapshot}')

    for snapshot in sweep_dir.glob('**/buffer/*.npz'):
        snapshot.unlink(missing_ok=True)
        print(f'removing {snapshot}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_dir', type=str)
    parser.add_argument('--remove',
                        dest='remove_snapshots',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    slurm_dir = sweep_dir / 'slurm'

    cancel_jobs(slurm_dir)
    if args.remove_snapshots:
        remove_snapshots(sweep_dir)


if __name__ == '__main__':
    main()
