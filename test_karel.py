"""Test script to run a program on a Karel environment."""

import os
import sys

import torch

sys.path.insert(
    0, os.path.dirname(os.path.realpath(__file__))
)  # hacky path manipulation to allow LEAPS code to be imported
from pretrain.get_karel_config import get_karel_task_config
from prl_gym.exec_env import ExecEnv2
from rl.envs import make_vec_envs

# Max length of program in the LEAPS dataset is 50
MAX_LEN = 50


def get_reward(program_text, mdp_config):
    program = program_text.replace("\\", "").replace("'", "")

    cfg_rl = mdp_config["rl"]
    cfg_envs = mdp_config["rl"]["envs"]

    custom = (
        True
        if "karel" or "CartPoleDiscrete" in cfg_envs["executable"]["name"]
        else False
    )
    test_env = make_vec_envs(
        cfg_envs["executable"]["name"],
        mdp_config["seed"],
        1,
        cfg_rl["gamma"],
        "/tmp",
        mdp_config["device"],
        False,
        custom_env=custom,
        custom_env_type="program",
        custom_kwargs={"config": mdp_config["args"]},
    )
    test_env.reset()

    program_seq = ExecEnv2(mdp_config["args"]).dsl.str2intseq(program)[1 : MAX_LEN + 1]
    program_seq += [MAX_LEN] * (MAX_LEN - len(program_seq))
    program_tensor = torch.tensor(
        program_seq, dtype=torch.int8, device=mdp_config["device"]
    ).unsqueeze(0)
    _, reward, _, _ = test_env.step(program_tensor)
    reward = reward.item()

    return reward


def main():
    tests = {
        "cleanHouse": [
            "DEF run m( IFELSE c( markersPresent c) i( IFELSE c( frontIsClear c) i( move i) ELSE e( move e) i) ELSE e( move putMarker e) IFELSE c( frontIsClear c) i( pickMarker i) ELSE e( turnRight e) IF c( not c( noMarkersPresent c) c) i( turnLeft i) m)",
        ],
        "fourCorners": [
            "DEF run m( IF c( markersPresent c) i( move i) WHILE c( frontIsClear c) w( turnLeft w) turnRight putMarker move putMarker m)"
        ],
        "harvester": [
            "DEF run m( turnRight turnRight turnRight turnRight pickMarker move turnLeft pickMarker move pickMarker move pickMarker move pickMarker move pickMarker move turnRight pickMarker move pickMarker move pickMarker move pickMarker move turnRight pickMarker move pickMarker move pickMarker move pickMarker move turnRight pickMarker move pickMarker move pickMarker move m)"
        ],
        "randomMaze": [
            "DEF run m( WHILE c( noMarkersPresent c) w( WHILE c( not c( markersPresent c) c) w( turnRight move w) w) m)"
        ],
        "stairClimber": [
            "DEF run m( WHILE c( noMarkersPresent c) w( turnRight move w) IF c( rightIsClear c) i( WHILE c( frontIsClear c) w( move w) i) m)"
        ],
        "topOff": [
            "DEF run m( WHILE c( noMarkersPresent c) w( move w) putMarker move turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight turnRight WHILE c( noMarkersPresent c) w( move w) turnRight turnRight turnRight turnRight putMarker turnRight m)",
            "DEF run m( WHILE c( not c( markersPresent c) c) w( move w) putMarker move WHILE c( not c( markersPresent c) c) w( move w) putMarker move WHILE c( not c( markersPresent c) c) w( move w) turnRight move turnRight putMarker turnRight move turnRight turnRight m)",
        ],
    }
    seed = 75092  # 28236
    for task in tests:
        mdp_config = get_karel_task_config(task, seed, num_demo_per_program=10)
        for program_text in tests[task]:
            reward = get_reward(program_text, mdp_config)

            print(task, reward, program_text)


if __name__ == "__main__":
    main()
