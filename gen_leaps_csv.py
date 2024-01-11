import sys
import os

from concurrent.futures import ProcessPoolExecutor

import h5py
import pandas as pd

from test_karel import get_reward

sys.path.insert(0, '.')

from prl_gym.exec_env import ExecEnv2
from pretrain.get_karel_config import get_karel_task_config

datadir = sys.argv[1]
hdf5_file = h5py.File(os.path.join(datadir, 'data.hdf5'), 'r')
with open(os.path.join(datadir, 'id.txt'), 'r') as id_file:
    id_list = id_file.readlines()
id_list = [id.strip() for id in id_list]

LIMIT = None

dummy = ExecEnv2(get_karel_task_config('topOff', 0)['args'])

prog_list = []
prog_lens = []
for prog_id in id_list[:LIMIT]:
    prog_seq = hdf5_file[prog_id]['program'][()]
    prog_len = int(prog_id.split('_')[4])
    prog_str = dummy.dsl.intseq2str(prog_seq)

    prog_list.append(prog_str)
    prog_lens.append(prog_len)

tasks = ('cleanHouse', 'harvester', 'fourCorners', 'randomMaze', 'stairClimber', 'topOff')
seed = 75092

def find_task_rewards(task):
    mdp_config = get_karel_task_config(task, seed, num_demo_per_program=10)
    rewards = []
    for prog in prog_list[:LIMIT]:
        reward = get_reward(prog, mdp_config)
        rewards.append(reward) 
    return rewards

num_processes = len(tasks)

with ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = [executor.submit(find_task_rewards, task) for task in tasks]
    results = [future.result() for future in futures]
    rewards = {task: rewards for (task, rewards) in zip(tasks, results)}

df = pd.DataFrame({
    'program': prog_list,
    'prog_len': prog_lens,
    **{f'{task}_reward': rewards[task] for task in tasks}
})

df.to_csv(sys.argv[2], index_label='indices')