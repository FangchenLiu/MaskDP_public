DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
]

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'point_mass_maze': 'point_mass_reach_multitask',
    'cheetah': 'cheetah_run',
    'cartpole': 'cartpole_balance'
}


TASK_TO_ID = {
    'walker_stand': 0,
    'walker_walk': 1,
    'walker_run': 2,
    'walker_flip': 3,
    'jaco_reach_top_left': 0,
    'jaco_reach_top_right': 1,
    'jaco_reach_bottom_left': 2,
    'jaco_reach_botoom_right': 3,
    'quadruped_stand': 0,
    'quadruped_walk': 1,
    'quadruped_run': 2,
    'quadruped_jump': 3,
    'point_mass_reach_top_left': 0,
    'point_mass_reach_top_right': 1,
    'point_mass_reach_bottom_left': 2,
    'point_mass_reach_botoom_right': 3,
}

def get_reward_id(task):
    return TASK_TO_ID.get(task, 0)