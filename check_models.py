import argparse
from pathlib import Path

from dmc_benchmark import DOMAINS

OBS_TYPES = ['states', 'pixels']
TIMES = [100000, 500000, 1000000, 2000000]


def find_all_agents(agents_dir):
    agents_dir = Path(agents_dir)
    agents = []
    for agent_file in agents_dir.glob('*.py'):
        agent = agent_file.stem.split('.')[0]
        if agent not in ['ddpg', 'protov2']:
            agents.append(agent)
    return agents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models_dir', type=str)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    agents = find_all_agents('./agent')

    for agent in agents:
        errors = []
        total, ready = 0, 0
        print(f'checking {agent}...')
        for obs_type in OBS_TYPES:
            for domain in DOMAINS:
                for t in TIMES:
                    avail_seeds = []
                    total += 1
                    for seed in range(1, 11):
                        snapshot = models_dir / obs_type / domain / agent / str(
                            seed) / f'snapshot_{t}.pt'
                        if snapshot.exists():
                            avail_seeds.append(seed)

                    if len(avail_seeds) == 0:
                        errors += [
                            f'{obs_type}/{domain}/{agent}/{seed}/snapshot_{t}.pt'
                        ]
                    else:
                        ready += 1

        if len(errors) > 0:
            print(f'missing {total - ready}:')
        else:
            print(f'all done {ready}/{total}')
        for error in errors:
            print(error)
        print(f'status {ready}/{total}\n')


if __name__ == '__main__':
    main()
