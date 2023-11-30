"""Test script to run a program on a Karel environment."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))) # hacky path manipulation to allow LEAPS code to be imported

from prl_gym.exec_env import ExecEnv2
from pretrain.get_karel_config import get_karel_task_config


if __name__ == '__main__':
    
    program_text = 'DEF run m( WHILE c( noMarkersPresent c) w( turnRight move w) putMarker move WHILE c( noMarkersPresent c) w( turnRight move w) putMarker move WHILE c( noMarkersPresent c) w( turnRight move w) turnRight move putMarker m)'
    mdp_config = get_karel_task_config('topOff')

    program = program_text.replace('\\', '').replace('\'', '')
    
    karel_env = ExecEnv2(mdp_config['args'])
    reward, pred_program = karel_env.reward(program, is_program_str=True)

    states_0 = pred_program['s_h'][0]
    actions_0 = pred_program['a_h'][0]
    rewards_0 = pred_program['reward_h'][0]

    actions_0_len = pred_program['a_h_len'][0]

    for i in range(actions_0_len):
        karel_env._world.print_state(states_0[i])
        print(actions_0[i], rewards_0[i])
    karel_env._world.print_state(states_0[actions_0_len])

    print(reward)