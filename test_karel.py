"""Test script to run a program on a Karel environment."""

import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))) # hacky path manipulation to allow LEAPS code to be imported
from prl_gym.exec_env import ExecEnv2
from rl.envs import make_vec_envs
from pretrain.get_karel_config import get_karel_task_config


def get_reward(program_text, mdp_config):
    program = program_text.replace('\\', '').replace('\'', '')
    
    cfg_rl = mdp_config['rl']
    cfg_envs = mdp_config['rl']['envs']

    custom = True if "karel" or "CartPoleDiscrete" in cfg_envs['executable']['name'] else False
    test_env = make_vec_envs(cfg_envs['executable']['name'], mdp_config['seed'], 1,
                                cfg_rl['gamma'], '/tmp', mdp_config['device'], False,
                                custom_env=custom, custom_env_type='program', custom_kwargs={'config': mdp_config['args']})
    test_env.reset()

    program_seq = ExecEnv2(mdp_config['args']).dsl.str2intseq(program)[1:mdp_config['max_program_len']]
    program_seq += [50] * (mdp_config['max_program_len'] - len(program_seq))
    program_tensor = torch.tensor(program_seq, dtype=torch.int8, device=mdp_config['device']).unsqueeze(0)
    _, reward, _, _ = test_env.step(program_tensor)
    reward = reward.item()

    return reward

def main():
    program_text = 'DEF run m( IFELSE c( frontIsClear c) i( move turnRight i) ELSE e( WHILE c( rightIsClear c) w( pickMarker w) e) IF c( frontIsClear c) i( move putMarker i) m)'
    seed = 100
    task = 'topOff'
    mdp_config = get_karel_task_config(task, seed)

    reward = main(program_text, mdp_config)

    print(reward)


if __name__ == '__main__':
    main()